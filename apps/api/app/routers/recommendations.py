from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.exc import ProgrammingError

from packages.core.db import db_session, get_database_url
from packages.db.models import MlModel, RecommendationExplanation, RecommendationRun
from packages.domain.universe import DEFAULT_ETF_SYMBOLS
from packages.services.explanations import generate_recommendation_explanation
from packages.services.recommendation import run_recommendation_backtest_from_files


router = APIRouter(prefix="/recommendations", tags=["recommendations"])


class BacktestRequest(BaseModel):
    symbols: list[str] | None = Field(default=None, description="ETF symbols, e.g. ['VT','VTI']")
    symbols_csv: str | None = Field(default=None, description="Comma-separated symbols, e.g. VT,VTI")
    data_source: str = Field(default="file", description="file|db")
    model_path: str = Field(default="artifacts/return_model/model.joblib")
    max_weight: float = Field(default=0.35, ge=0.01, le=1.0)
    cov_lookback_months: int = Field(default=36, ge=12, le=120)
    rf_annual: float = Field(default=0.0, ge=-0.5, le=0.5)
    transaction_cost_bps: float = Field(default=10.0, ge=0.0, le=200.0)
    include_equity: bool = False
    include_weight_history: bool = False


class BacktestResponse(BaseModel):
    run_id: str | None = None
    symbols: list[str]
    metrics: dict[str, float]
    benchmarks: dict[str, dict[str, float]] | None = None
    latest_weights: dict[str, float]
    equity_curve: list[dict] | None = None
    weight_history: list[dict] | None = None


@router.post("/backtest", response_model=BacktestResponse)
def backtest(req: BacktestRequest) -> BacktestResponse:
    symbols = req.symbols
    if (not symbols) and req.symbols_csv:
        symbols = [s.strip().upper() for s in req.symbols_csv.split(",") if s.strip()]
    if not symbols:
        symbols = list(DEFAULT_ETF_SYMBOLS)

    allowed = set(DEFAULT_ETF_SYMBOLS)
    invalid = [s for s in symbols if s not in allowed]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Unsupported symbols: {invalid}. Allowed: {sorted(allowed)}")
    if len(symbols) > 15:
        raise HTTPException(status_code=400, detail="Too many symbols (max 15)")
    if req.data_source not in ("file", "db"):
        raise HTTPException(status_code=400, detail="Invalid data_source (use file|db)")

    try:
        result = run_recommendation_backtest_from_files(
            symbols=symbols,
            data_source=req.data_source,
            model_path=req.model_path,
            include_equity=req.include_equity,
            include_weight_history=req.include_weight_history,
            max_weight=req.max_weight,
            cov_lookback_months=req.cov_lookback_months,
            rf_annual=req.rf_annual,
            transaction_cost_bps=req.transaction_cost_bps,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return BacktestResponse(
        run_id=result.run_id,
        symbols=result.symbols,
        metrics=result.metrics,
        benchmarks=result.benchmarks,
        latest_weights=result.latest_weights,
        equity_curve=result.equity_curve,
        weight_history=result.weight_history,
    )


class RunSummary(BaseModel):
    id: str
    created_at: str
    symbols: list
    metrics: dict
    latest_weights: dict
    benchmarks: dict | None = None


class ModelSummary(BaseModel):
    id: str
    created_at: str
    name: str
    task: str
    data_frequency: str
    feature_schema: str
    metrics: dict
    artifact_path: str
    artifact_sha256: str


@router.get("/models", response_model=list[ModelSummary])
def list_models(limit: int = 20) -> list[ModelSummary]:
    try:
        _ = get_database_url()
    except Exception:
        return []
    limit = max(1, min(int(limit), 200))
    with db_session() as session:
        try:
            models = session.query(MlModel).order_by(MlModel.created_at.desc()).limit(limit).all()
        except ProgrammingError:
            raise HTTPException(status_code=503, detail="Database migrations not applied")
        return [
            ModelSummary(
                id=str(m.id),
                created_at=m.created_at.isoformat(),
                name=m.name,
                task=m.task,
                data_frequency=m.data_frequency,
                feature_schema=m.feature_schema,
                metrics=m.metrics,
                artifact_path=m.artifact_path,
                artifact_sha256=m.artifact_sha256,
            )
            for m in models
        ]


@router.get("/runs", response_model=list[RunSummary])
def list_runs(limit: int = 20) -> list[RunSummary]:
    try:
        _ = get_database_url()
    except Exception:
        return []

    limit = max(1, min(int(limit), 200))
    with db_session() as session:
        try:
            runs = (
                session.query(RecommendationRun)
                .order_by(RecommendationRun.created_at.desc())
                .limit(limit)
                .all()
            )
        except ProgrammingError:
            raise HTTPException(status_code=503, detail="Database migrations not applied")
        return [
            RunSummary(
                id=str(r.id),
                created_at=r.created_at.isoformat(),
                symbols=r.symbols,
                metrics=r.metrics,
                latest_weights=r.latest_weights,
                benchmarks=getattr(r, "benchmarks", None),
            )
            for r in runs
        ]


@router.get("/runs/{run_id}", response_model=RunSummary)
def get_run(run_id: str) -> RunSummary:
    try:
        _ = get_database_url()
    except Exception:
        raise HTTPException(status_code=400, detail="DATABASE_URL is not configured")

    try:
        run_uuid = uuid.UUID(run_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid run_id")

    with db_session() as session:
        try:
            run = session.get(RecommendationRun, run_uuid)
        except ProgrammingError:
            raise HTTPException(status_code=503, detail="Database migrations not applied")
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return RunSummary(
            id=str(run.id),
            created_at=run.created_at.isoformat(),
            symbols=run.symbols,
            metrics=run.metrics,
            latest_weights=run.latest_weights,
            benchmarks=getattr(run, "benchmarks", None),
        )


class ExplanationResponse(BaseModel):
    id: str
    run_id: str
    created_at: str
    provider: str
    model: str
    prompt_version: str
    content_type: str
    content: str


@router.get("/runs/{run_id}/explanation", response_model=ExplanationResponse)
def get_explanation(run_id: str) -> ExplanationResponse:
    try:
        _ = get_database_url()
    except Exception:
        raise HTTPException(status_code=400, detail="DATABASE_URL is not configured")

    try:
        run_uuid = uuid.UUID(run_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid run_id")

    with db_session() as session:
        try:
            expl = (
                session.query(RecommendationExplanation)
                .filter(RecommendationExplanation.run_id == run_uuid)
                .order_by(RecommendationExplanation.created_at.desc())
                .first()
            )
        except ProgrammingError:
            raise HTTPException(status_code=503, detail="Database migrations not applied")
        if expl is None:
            raise HTTPException(status_code=404, detail="Explanation not found")
        return ExplanationResponse(
            id=str(expl.id),
            run_id=str(expl.run_id),
            created_at=expl.created_at.isoformat(),
            provider=expl.provider,
            model=expl.model,
            prompt_version=expl.prompt_version,
            content_type=expl.content_type,
            content=expl.content,
        )


class ExplainRequest(BaseModel):
    force: bool = False


@router.post("/runs/{run_id}/explain", response_model=ExplanationResponse)
def explain_run(run_id: str, req: ExplainRequest) -> ExplanationResponse:
    try:
        _ = get_database_url()
    except Exception:
        raise HTTPException(status_code=400, detail="DATABASE_URL is not configured")

    try:
        run_uuid = uuid.UUID(run_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid run_id")

    with db_session() as session:
        try:
            run = session.get(RecommendationRun, run_uuid)
        except ProgrammingError:
            raise HTTPException(status_code=503, detail="Database migrations not applied")

        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        if not req.force:
            existing = (
                session.query(RecommendationExplanation)
                .filter(RecommendationExplanation.run_id == run_uuid)
                .order_by(RecommendationExplanation.created_at.desc())
                .first()
            )
            if existing is not None:
                return ExplanationResponse(
                    id=str(existing.id),
                    run_id=str(existing.run_id),
                    created_at=existing.created_at.isoformat(),
                    provider=existing.provider,
                    model=existing.model,
                    prompt_version=existing.prompt_version,
                    content_type=existing.content_type,
                    content=existing.content,
                )

        run_payload = {
            "id": str(run.id),
            "created_at": run.created_at.isoformat(),
            "data_source": run.data_source,
            "data_frequency": run.data_frequency,
            "model_path": run.model_path,
            "request": run.request,
            "symbols": run.symbols,
            "metrics": run.metrics,
            "latest_weights": run.latest_weights,
        }

        try:
            generated = generate_recommendation_explanation(run_payload=run_payload)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        expl = RecommendationExplanation(
            run_id=run.id,
            provider=generated.provider,
            model=generated.model,
            prompt_version=generated.prompt_version,
            content_type=generated.content_type,
            content=generated.content,
            request=generated.request,
            response=generated.response,
        )
        session.add(expl)
        session.commit()
        session.refresh(expl)

        return ExplanationResponse(
            id=str(expl.id),
            run_id=str(expl.run_id),
            created_at=expl.created_at.isoformat(),
            provider=expl.provider,
            model=expl.model,
            prompt_version=expl.prompt_version,
            content_type=expl.content_type,
            content=expl.content,
        )
