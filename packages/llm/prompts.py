from __future__ import annotations

from typing import Any


PROMPT_VERSION = "v1"


def build_recommendation_explanation_messages(*, run: dict[str, Any]) -> list[dict[str, str]]:
    """
    Grounded explanation prompt: only uses computed weights/metrics and explicit assumptions.
    """
    symbols = run.get("symbols") or []
    metrics = run.get("metrics") or {}
    weights = run.get("latest_weights") or {}
    request = run.get("request") or {}

    system = (
        "You are a careful investment assistant. "
        "Explain the portfolio and backtest results clearly, "
        "without claiming certainty or giving personalized financial advice. "
        "Use only the provided data. If something is missing, say so. "
        "Use plain ASCII only (no Unicode symbols like × or −)."
    )

    user = {
        "symbols": symbols,
        "latest_weights": weights,
        "metrics": metrics,
        "assumptions": {
            "asset_universe": "ETFs only",
            "objective": "Maximize Sharpe ratio (constrained long-only)",
            "rebalance_frequency": "monthly",
            "max_weight": request.get("max_weight"),
            "cov_lookback_months": request.get("cov_lookback_months"),
            "risk_free_rate_annual": request.get("rf_annual"),
            "data_frequency": run.get("data_frequency"),
            "data_source": run.get("data_source"),
        },
        "requirements": {
            "format": "markdown",
            "sections": [
                "Summary (2-3 sentences)",
                "Portfolio Breakdown (bullets)",
                "Backtest Interpretation (bullets)",
                "Key Risks (bullets)",
                "Assumptions & Limitations (bullets)",
                "Disclaimer (1 sentence)",
            ],
        },
    }

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Explain this run:\n{user}"},
    ]
