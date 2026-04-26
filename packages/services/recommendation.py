from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from packages.core.db import db_session, get_database_url
from packages.db.models import RecommendationBacktestPoint, RecommendationRun, RecommendationWeightHistory
from packages.domain.constraints import DEFAULT_CONSTRAINTS
from packages.domain.universe import DEFAULT_ETF_SYMBOLS
from packages.market_data.file_store import LocalPriceStore
from packages.ml.features import add_basic_features, to_monthly_dataset
from packages.ml.modeling import prepare_monthly_dataset, predict_expected_returns, train_return_model
from packages.quant.backtest import BacktestResult, walk_forward_monthly_backtest
from packages.quant.prices import load_month_end_price_panel


@dataclass(frozen=True)
class RecommendationBacktestResponse:
    run_id: str | None
    symbols: list[str]
    metrics: dict[str, float]
    latest_weights: dict[str, float]
    equity_curve: list[dict[str, float | str]] | None
    weight_history: list[dict[str, float | str]] | None


def _build_monthly_feature_rows(store: LocalPriceStore, symbols: list[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for sym in symbols:
        daily = store.read_symbol_bars(sym)
        daily = add_basic_features(daily)
        monthly = to_monthly_dataset(daily)
        parts.append(monthly)
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    return df


def _expected_return_fn_from_model(
    *,
    model,
    monthly_feature_df: pd.DataFrame,
) -> callable:
    def fn(rebalance_date: date, symbols: list[str]) -> np.ndarray:
        rows = monthly_feature_df[monthly_feature_df["date"].dt.date == rebalance_date].copy()
        rows = rows.set_index("symbol").reindex(symbols).reset_index()
        preds = predict_expected_returns(model, rows)
        return preds

    return fn


def run_recommendation_backtest_from_files(
    *,
    symbols: list[str] | None = None,
    model_path: str = "artifacts/return_model/model.joblib",
    include_equity: bool = False,
    include_weight_history: bool = False,
    max_weight: float = DEFAULT_CONSTRAINTS.max_weight,
    cov_lookback_months: int = 36,
    rf_annual: float = 0.0,
) -> RecommendationBacktestResponse:
    symbols = list(DEFAULT_ETF_SYMBOLS) if not symbols else [s.strip().upper() for s in symbols if s.strip()]
    if not symbols:
        raise ValueError("No symbols provided")

    store = LocalPriceStore.default()
    panel = load_month_end_price_panel(store, symbols, min_common_months=cov_lookback_months + 12)

    monthly_feature_df = _build_monthly_feature_rows(store, symbols)
    monthly_feature_df = prepare_monthly_dataset(monthly_feature_df)

    model_file = Path(model_path)
    if model_file.exists():
        model = joblib.load(model_file)
        expected_return_fn = _expected_return_fn_from_model(model=model, monthly_feature_df=monthly_feature_df)
    else:
        # Lightweight fallback: train once on all available data (still time-safe for backtest if data already ended).
        model = train_return_model(monthly_feature_df)
        expected_return_fn = _expected_return_fn_from_model(model=model, monthly_feature_df=monthly_feature_df)

    bt: BacktestResult = walk_forward_monthly_backtest(
        month_end_prices=panel.prices,
        expected_return_fn=expected_return_fn,
        max_weight=float(max_weight),
        cov_lookback_months=int(cov_lookback_months),
        rf_annual=float(rf_annual),
    )

    latest_weights: dict[str, float] = {}
    if len(bt.weights) > 0:
        latest = bt.weights.iloc[-1]
        latest_weights = {k: float(v) for k, v in latest.sort_values(ascending=False).items() if float(v) > 0}

    equity_curve = None
    if include_equity:
        equity_curve = [
            {
                "date": str(pd.to_datetime(row["date"]).date()),
                "portfolio_value": float(row["portfolio_value"]),
                "portfolio_return": float(row["portfolio_return"]),
            }
            for _, row in bt.equity_curve.iterrows()
        ]

    weight_history = None
    if include_weight_history and len(bt.weights) > 0:
        weight_history = []
        for idx, row in bt.weights.iterrows():
            item = {"rebalance_date": str(pd.to_datetime(idx).date())}
            for sym, w in row.items():
                item[sym] = float(w)
            weight_history.append(item)

    run_id: str | None = None
    # Persist the run if DB is available (Docker Compose path).
    try:
        _ = get_database_url()
        with db_session() as session:
            run = RecommendationRun(
                data_source="file",
                data_frequency="monthly",
                model_path=model_path,
                request={
                    "max_weight": float(max_weight),
                    "cov_lookback_months": int(cov_lookback_months),
                    "rf_annual": float(rf_annual),
                    "include_equity": bool(include_equity),
                    "include_weight_history": bool(include_weight_history),
                },
                symbols=symbols,
                metrics={k: float(v) for k, v in bt.metrics.items()},
                latest_weights=latest_weights,
            )
            session.add(run)
            session.flush()
            run_id = str(run.id)

            # Always persist the equity curve + weight history for replay/debugging.
            points = []
            for _, row in bt.equity_curve.iterrows():
                points.append(
                    RecommendationBacktestPoint(
                        run_id=run.id,
                        date=pd.to_datetime(row["date"]).date(),
                        portfolio_value=row["portfolio_value"],
                        portfolio_return=row["portfolio_return"],
                    )
                )
            session.add_all(points)

            hist_rows = []
            for reb_date, row in bt.weights.iterrows():
                hist_rows.append(
                    RecommendationWeightHistory(
                        run_id=run.id,
                        rebalance_date=pd.to_datetime(reb_date).date(),
                        weights={sym: float(w) for sym, w in row.items()},
                    )
                )
            session.add_all(hist_rows)
            session.commit()
    except Exception:
        # Persistence is best-effort; the API result should still be returned.
        run_id = run_id

    return RecommendationBacktestResponse(
        run_id=run_id,
        symbols=symbols,
        metrics={k: float(v) for k, v in bt.metrics.items()},
        latest_weights=latest_weights,
        equity_curve=equity_curve,
        weight_history=weight_history,
    )
