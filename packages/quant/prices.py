from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

from packages.market_data.store import PriceStore


@dataclass(frozen=True)
class PricePanel:
    prices: pd.DataFrame  # index: month_end date, columns: symbols, values: adjusted_close

    @property
    def symbols(self) -> list[str]:
        return list(self.prices.columns)

    def monthly_returns(self) -> pd.DataFrame:
        return self.prices.pct_change().dropna(how="any")


def _month_end_prices(daily: pd.DataFrame) -> pd.Series:
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "adjusted_close" in df.columns and pd.to_numeric(df["adjusted_close"], errors="coerce").notna().any():
        price_col = "adjusted_close"
    else:
        price_col = "close"
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])

    df["month"] = df["date"].dt.to_period("M")
    last = df.groupby("month", as_index=False).tail(1)
    last = last.sort_values("date")
    series = pd.Series(last[price_col].to_numpy(), index=last["date"].dt.date.to_numpy())
    series.index.name = "date"
    return series


def load_month_end_price_panel(
    store: PriceStore,
    symbols: Iterable[str],
    *,
    min_common_months: int = 36,
) -> PricePanel:
    """
    Loads daily CSVs per symbol and reduces them into month-end adjusted_close prices.
    Returns an aligned panel with a common date index (intersection across symbols).
    """
    series_by_symbol: dict[str, pd.Series] = {}
    for sym in symbols:
        daily = store.read_symbol_bars(sym)
        series_by_symbol[sym] = _month_end_prices(daily)

    prices = pd.DataFrame(series_by_symbol).sort_index()

    # Keep only months where every symbol has a price (common start date).
    prices = prices.dropna(how="any")
    if len(prices) < min_common_months:
        raise RuntimeError(
            f"Not enough common month-end points across symbols ({len(prices)}). "
            f"Try fewer symbols or ingest more history."
        )

    # Ensure strictly increasing index and no duplicates.
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.sort_index()

    # Basic sanity checks.
    if not np.isfinite(prices.to_numpy()).all():
        raise RuntimeError("Non-finite values in price panel after alignment")

    return PricePanel(prices=prices)
