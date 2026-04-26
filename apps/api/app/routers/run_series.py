from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy.exc import ProgrammingError

from packages.core.db import db_session, get_database_url
from packages.db.models import RecommendationBacktestPoint, RecommendationWeightHistory


router = APIRouter(prefix="/recommendations", tags=["recommendations"])


class EquityPoint(BaseModel):
    date: str
    portfolio_value: float
    portfolio_return: float


class RunSeriesResponse(BaseModel):
    equity_curve: list[EquityPoint]
    weight_history: list[dict]


@router.get("/runs/{run_id}/series", response_model=RunSeriesResponse)
def get_run_series(run_id: str) -> RunSeriesResponse:
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
            points = (
                session.query(RecommendationBacktestPoint)
                .filter(RecommendationBacktestPoint.run_id == run_uuid)
                .order_by(RecommendationBacktestPoint.date.asc())
                .all()
            )
            weights = (
                session.query(RecommendationWeightHistory)
                .filter(RecommendationWeightHistory.run_id == run_uuid)
                .order_by(RecommendationWeightHistory.rebalance_date.asc())
                .all()
            )
        except ProgrammingError:
            raise HTTPException(status_code=503, detail="Database migrations not applied")

    equity_curve = [
        EquityPoint(
            date=p.date.isoformat(),
            portfolio_value=float(p.portfolio_value),
            portfolio_return=float(p.portfolio_return),
        )
        for p in points
    ]
    weight_history = []
    for w in weights:
        item = {"rebalance_date": w.rebalance_date.isoformat()}
        item.update({k: float(v) for k, v in w.weights.items()})
        weight_history.append(item)

    return RunSeriesResponse(equity_curve=equity_curve, weight_history=weight_history)

