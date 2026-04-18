from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConstraints:
    # Long-only portfolio (no shorting, no leverage).
    long_only: bool = True

    # Fully-invested by default (sum of weights == 1.0).
    target_gross_exposure: float = 1.0

    # Per-asset bounds.
    min_weight: float = 0.0
    max_weight: float = 0.35

    # Rebalance cadence used by backtests and walk-forward evaluation.
    rebalance_frequency: str = "monthly"

    # Optimization objective (v1).
    objective: str = "max_sharpe"


DEFAULT_CONSTRAINTS = PortfolioConstraints()

