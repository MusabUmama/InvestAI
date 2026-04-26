from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.exc import ProgrammingError

from packages.core.db import db_session, get_database_url
from packages.db.models import RecommendationRun
from packages.services.recommendation import run_recommendation_backtest_from_files


router = APIRouter(prefix="/recommendations", tags=["recommendations"])


class BacktestRequest(BaseModel):
    symbols: list[str] | None = Field(default=None, description="ETF symbols, e.g. ['VT','VTI']")
    symbols_csv: str | None = Field(default=None, description="Comma-separated symbols, e.g. VT,VTI")
    model_path: str = Field(default="artifacts/return_model/model.joblib")
    max_weight: float = Field(default=0.35, ge=0.01, le=1.0)
    cov_lookback_months: int = Field(default=36, ge=12, le=120)
    rf_annual: float = Field(default=0.0, ge=-0.5, le=0.5)
    include_equity: bool = False
    include_weight_history: bool = False


class BacktestResponse(BaseModel):
    run_id: str | None = None
    symbols: list[str]
    metrics: dict[str, float]
    latest_weights: dict[str, float]
    equity_curve: list[dict] | None = None
    weight_history: list[dict] | None = None


@router.post("/backtest", response_model=BacktestResponse)
def backtest(req: BacktestRequest) -> BacktestResponse:
    symbols = req.symbols
    if (not symbols) and req.symbols_csv:
        symbols = [s.strip().upper() for s in req.symbols_csv.split(",") if s.strip()]

    try:
        result = run_recommendation_backtest_from_files(
            symbols=symbols,
            model_path=req.model_path,
            include_equity=req.include_equity,
            include_weight_history=req.include_weight_history,
            max_weight=req.max_weight,
            cov_lookback_months=req.cov_lookback_months,
            rf_annual=req.rf_annual,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return BacktestResponse(
        run_id=result.run_id,
        symbols=result.symbols,
        metrics=result.metrics,
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
        )
