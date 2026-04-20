from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd

from packages.quant.optimizer import maximize_sharpe_via_mean_variance_sweep


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.DataFrame  # columns: date, portfolio_value, portfolio_return
    weights: pd.DataFrame       # index: rebalance date, columns: symbols
    metrics: dict[str, float]


def _annualized_sharpe(returns: np.ndarray, rf_month: float = 0.0) -> float:
    r = returns - rf_month
    mean_m = float(np.mean(r))
    std_m = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    if std_m <= 0:
        return float("nan")
    return (mean_m * 12.0) / (std_m * np.sqrt(12.0))


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(np.min(dd))


def walk_forward_monthly_backtest(
    *,
    month_end_prices: pd.DataFrame,   # index: month_end date, columns: symbols
    expected_return_fn: Callable[[date, list[str]], np.ndarray],
    max_weight: float = 0.35,
    cov_lookback_months: int = 36,
    rf_annual: float = 0.0,
) -> BacktestResult:
    """
    Monthly walk-forward:
    - At month-end t, compute expected returns for t->t+1
    - Estimate covariance from trailing monthly returns over `cov_lookback_months`
    - Optimize constrained Sharpe (approx via mean-variance sweep)
    - Apply weights to realized returns in month t+1
    """
    prices = month_end_prices.sort_index()
    symbols = list(prices.columns)
    rets = prices.pct_change().dropna(how="any")

    if len(rets) < cov_lookback_months + 6:
        raise RuntimeError("Not enough months for backtest + covariance lookback")

    rf_month = (1.0 + rf_annual) ** (1.0 / 12.0) - 1.0

    weight_rows = []
    equity_dates = []
    equity_vals = []
    port_rets = []

    value = 1.0

    # Iterate over rebalance dates that have a forward month to realize.
    for i in range(cov_lookback_months, len(rets) - 1):
        rebalance_date = rets.index[i]  # month-end for which we predict next month
        next_month_date = rets.index[i + 1]

        window = rets.iloc[i - cov_lookback_months : i]
        sigma = np.cov(window.to_numpy().T, ddof=1)
        # Light shrinkage towards diagonal for stability.
        diag = np.diag(np.diag(sigma))
        sigma = 0.9 * sigma + 0.1 * diag

        mu = expected_return_fn(rebalance_date, symbols)
        mu = np.asarray(mu, dtype=float)
        if mu.shape != (len(symbols),):
            raise RuntimeError("expected_return_fn must return (n_symbols,) vector")

        opt = maximize_sharpe_via_mean_variance_sweep(mu=mu, sigma=sigma, rf=rf_month, max_weight=max_weight)
        w = opt.weights

        realized = rets.iloc[i + 1].to_numpy(dtype=float)
        port_r = float(w @ realized)
        value *= (1.0 + port_r)

        weight_rows.append((rebalance_date, w))
        equity_dates.append(next_month_date)
        equity_vals.append(value)
        port_rets.append(port_r)

    weights_df = pd.DataFrame(
        data=[w for _, w in weight_rows],
        index=pd.Index([d for d, _ in weight_rows], name="rebalance_date"),
        columns=symbols,
    )

    equity_curve = pd.DataFrame(
        {
            "date": pd.to_datetime(equity_dates),
            "portfolio_value": equity_vals,
            "portfolio_return": port_rets,
        }
    )

    ret_arr = np.asarray(port_rets, dtype=float)
    metrics = {
        "months": float(len(ret_arr)),
        "total_return": float(equity_vals[-1] - 1.0) if equity_vals else float("nan"),
        "cagr": float(equity_vals[-1] ** (12.0 / len(ret_arr)) - 1.0) if len(ret_arr) > 0 else float("nan"),
        "vol_annual": float(np.std(ret_arr, ddof=1) * np.sqrt(12.0)) if len(ret_arr) > 1 else float("nan"),
        "sharpe_annual": _annualized_sharpe(ret_arr, rf_month=rf_month),
        "max_drawdown": _max_drawdown(np.asarray(equity_vals, dtype=float)) if equity_vals else float("nan"),
    }

    return BacktestResult(equity_curve=equity_curve, weights=weights_df, metrics=metrics)

