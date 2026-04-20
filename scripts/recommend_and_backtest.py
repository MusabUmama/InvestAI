from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
import joblib
import numpy as np
import pandas as pd

from packages.domain.constraints import DEFAULT_CONSTRAINTS
from packages.domain.universe import DEFAULT_ETF_SYMBOLS
from packages.market_data.file_store import LocalPriceStore
from packages.ml.features import add_basic_features, to_monthly_dataset
from packages.ml.modeling import prepare_monthly_dataset, predict_expected_returns, train_return_model
from packages.quant.backtest import walk_forward_monthly_backtest
from packages.quant.prices import load_month_end_price_panel


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


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Recommend Sharpe-optimized ETF weights and run walk-forward backtest.")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols (default: built-in ETF universe).")
    parser.add_argument("--max-weight", type=float, default=DEFAULT_CONSTRAINTS.max_weight)
    parser.add_argument("--cov-lookback-months", type=int, default=36)
    parser.add_argument("--rf-annual", type=float, default=0.0)
    parser.add_argument(
        "--model",
        default="artifacts/return_model/model.joblib",
        help="Path to a pre-trained model.joblib. If missing, the script trains walk-forward each rebalance.",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/backtests/latest",
        help="Directory to write metrics, equity curve, and weights.",
    )
    args = parser.parse_args()

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols.strip()
        else list(DEFAULT_ETF_SYMBOLS)
    )

    store = LocalPriceStore.default()
    panel = load_month_end_price_panel(store, symbols, min_common_months=args.cov_lookback_months + 12)

    monthly_feature_df = _build_monthly_feature_rows(store, symbols)
    monthly_feature_df = prepare_monthly_dataset(monthly_feature_df)

    model_path = Path(args.model)
    if model_path.exists():
        model = joblib.load(model_path)

        def expected_return_fn(rebalance_date: date, sym_list: list[str]) -> np.ndarray:
            rows = monthly_feature_df[monthly_feature_df["date"].dt.date == rebalance_date].copy()
            rows = rows[rows["symbol"].isin(sym_list)]
            rows = rows.sort_values("symbol")
            # Align to sym_list order
            rows = rows.set_index("symbol").reindex(sym_list).reset_index()
            preds = predict_expected_returns(model, rows)
            return preds

    else:
        def expected_return_fn(rebalance_date: date, sym_list: list[str]) -> np.ndarray:
            # Train on all data available up to previous month-end to avoid leakage.
            cutoff = pd.Timestamp(rebalance_date) - pd.offsets.MonthEnd(1)
            train_df = monthly_feature_df[monthly_feature_df["date"] <= cutoff].copy()
            if len(train_df) < 100:
                # Not enough training data; fall back to simple momentum proxy.
                rows = monthly_feature_df[monthly_feature_df["date"].dt.date == rebalance_date].copy()
                rows = rows.set_index("symbol").reindex(sym_list).reset_index()
                return np.nan_to_num(rows["mom_63d"].to_numpy(dtype=float), nan=0.0)
            model = train_return_model(train_df)
            rows = monthly_feature_df[monthly_feature_df["date"].dt.date == rebalance_date].copy()
            rows = rows.set_index("symbol").reindex(sym_list).reset_index()
            return predict_expected_returns(model, rows)

    bt = walk_forward_monthly_backtest(
        month_end_prices=panel.prices,
        expected_return_fn=expected_return_fn,
        max_weight=float(args.max_weight),
        cov_lookback_months=int(args.cov_lookback_months),
        rf_annual=float(args.rf_annual),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "metrics.json").write_text(json.dumps(bt.metrics, indent=2), encoding="utf-8")
    bt.equity_curve.to_csv(out_dir / "equity_curve.csv", index=False)
    bt.weights.to_csv(out_dir / "weights.csv")

    # Also write the latest recommended weights (last rebalance row).
    if len(bt.weights) > 0:
        latest = bt.weights.iloc[-1].sort_values(ascending=False)
        latest.to_csv(out_dir / "latest_weights.csv", header=["weight"])

    print(json.dumps(bt.metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

