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
    benchmarks: dict[str, dict[str, float]]


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


def _basic_metrics(*, returns: np.ndarray, equity_vals: np.ndarray, rf_month: float) -> dict[str, float]:
    months = float(len(returns))
    total_return = float(equity_vals[-1] - 1.0) if len(equity_vals) else float("nan")
    cagr = float(equity_vals[-1] ** (12.0 / len(returns)) - 1.0) if len(returns) > 0 else float("nan")
    vol_annual = float(np.std(returns, ddof=1) * np.sqrt(12.0)) if len(returns) > 1 else float("nan")
    sharpe_annual = _annualized_sharpe(returns, rf_month=rf_month)
    max_dd = _max_drawdown(equity_vals) if len(equity_vals) else float("nan")
    return {
        "months": months,
        "total_return": total_return,
        "cagr": cagr,
        "vol_annual": vol_annual,
        "sharpe_annual": sharpe_annual,
        "max_drawdown": max_dd,
    }


def walk_forward_monthly_backtest(
    *,
    month_end_prices: pd.DataFrame,   # index: month_end date, columns: symbols
    expected_return_fn: Callable[[date, list[str]], np.ndarray],
    max_weight: float = 0.35,
    cov_lookback_months: int = 36,
    rf_annual: float = 0.0,
    transaction_cost_bps: float = 10.0,
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
    cost_rate = float(transaction_cost_bps) / 10000.0

    weight_rows = []
    equity_dates = []
    equity_vals = []
    port_rets = []
    gross_rets = []
    turnovers = []
    costs = []

    value = 1.0
    prev_w = None

    # Benchmark equity (no costs)
    bench = {
        "equal_weight": 1.0,
    }
    if "VT" in symbols:
        bench["VT"] = 1.0
    if "VTI" in symbols and "BND" in symbols:
        bench["60_40_VTI_BND"] = 1.0
    bench_rets: dict[str, list[float]] = {k: [] for k in bench}
    bench_vals: dict[str, list[float]] = {k: [] for k in bench}

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
        port_r_gross = float(w @ realized)

        turnover = 0.0
        if prev_w is not None:
            turnover = 0.5 * float(np.abs(w - prev_w).sum())
        cost = cost_rate * turnover
        port_r = port_r_gross - cost

        value *= (1.0 + port_r)

        weight_rows.append((rebalance_date, w))
        equity_dates.append(next_month_date)
        equity_vals.append(value)
        port_rets.append(port_r)
        gross_rets.append(port_r_gross)
        turnovers.append(turnover)
        costs.append(cost)
        prev_w = w

        # Benchmarks for the same realized month
        eq_r = float(np.mean(realized))
        bench["equal_weight"] *= (1.0 + eq_r)
        bench_rets["equal_weight"].append(eq_r)
        bench_vals["equal_weight"].append(bench["equal_weight"])

        if "VT" in symbols:
            vt_idx = symbols.index("VT")
            vt_r = float(realized[vt_idx])
            bench["VT"] *= (1.0 + vt_r)
            bench_rets["VT"].append(vt_r)
            bench_vals["VT"].append(bench["VT"])

        if "VTI" in symbols and "BND" in symbols:
            vti_r = float(realized[symbols.index("VTI")])
            bnd_r = float(realized[symbols.index("BND")])
            r6040 = 0.6 * vti_r + 0.4 * bnd_r
            bench["60_40_VTI_BND"] *= (1.0 + r6040)
            bench_rets["60_40_VTI_BND"].append(float(r6040))
            bench_vals["60_40_VTI_BND"].append(bench["60_40_VTI_BND"])

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
            "portfolio_return_gross": gross_rets,
            "turnover": turnovers,
            "transaction_cost": costs,
        }
    )

    ret_arr = np.asarray(port_rets, dtype=float)
    equity_arr = np.asarray(equity_vals, dtype=float)
    metrics = _basic_metrics(returns=ret_arr, equity_vals=equity_arr, rf_month=rf_month)
    metrics["turnover_mean"] = float(np.mean(turnovers)) if len(turnovers) else float("nan")
    metrics["turnover_total"] = float(np.sum(turnovers)) if len(turnovers) else float("nan")
    metrics["transaction_cost_bps"] = float(transaction_cost_bps)

    benchmarks: dict[str, dict[str, float]] = {}
    for name in bench_rets:
        r = np.asarray(bench_rets[name], dtype=float)
        v = np.asarray(bench_vals[name], dtype=float)
        benchmarks[name] = _basic_metrics(returns=r, equity_vals=v, rf_month=rf_month)

    return BacktestResult(equity_curve=equity_curve, weights=weights_df, metrics=metrics, benchmarks=benchmarks)
