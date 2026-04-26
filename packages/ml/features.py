from __future__ import annotations

import pandas as pd


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: bars for a single symbol with columns at least: date, adjusted_close (or close).
    With our current Alpha Vantage free-tier path, bars are typically monthly.
    Output: same rows with basic technical features.
    """
    out = df.copy()
    out = out.sort_values("date")

    if "adjusted_close" in out.columns and pd.to_numeric(out["adjusted_close"], errors="coerce").notna().any():
        price_col = "adjusted_close"
    else:
        price_col = "close"
    if price_col not in out.columns:
        raise ValueError("Expected adjusted_close or close column")

    price = pd.to_numeric(out[price_col], errors="coerce")
    # Periods are "rows" (monthly if using TIME_SERIES_MONTHLY).
    out["ret_1p"] = price.pct_change(1)
    out["ret_3p"] = price.pct_change(3)
    out["ret_6p"] = price.pct_change(6)
    out["ret_12p"] = price.pct_change(12)

    out["vol_12p"] = out["ret_1p"].rolling(12).std()
    out["vol_36p"] = out["ret_1p"].rolling(36).std()

    out["mom_12p"] = price.pct_change(12)
    out["mom_24p"] = price.pct_change(24)

    # Simple moving average ratios (trend proxy)
    sma_12 = price.rolling(12).mean()
    sma_36 = price.rolling(36).mean()
    out["sma12_ratio"] = price / sma_12 - 1.0
    out["sma36_ratio"] = price / sma_36 - 1.0

    return out


def to_monthly_dataset(daily: pd.DataFrame, horizon_days: int = 21) -> pd.DataFrame:
    """
    Converts daily features to a monthly-ish supervised dataset:
    - uses the last available row per calendar month as the feature row
    - target is forward return to the next feature row (next month)
    """
    df = daily.copy()
    df = df.sort_values(["symbol", "date"])

    if "adjusted_close" in df.columns and pd.to_numeric(df["adjusted_close"], errors="coerce").notna().any():
        price_col = "adjusted_close"
    else:
        price_col = "close"
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    month_last = df.groupby(["symbol", "month"], as_index=False).tail(1)

    # Forward return target: next period return from feature date
    month_last["target_fwd_ret"] = (
        month_last.groupby("symbol")[price_col].shift(-1) / month_last[price_col] - 1.0
    )
    # Also include a volatility-like target for later models
    month_last["target_fwd_vol"] = month_last.groupby("symbol")["ret_1p"].shift(-1).rolling(3).std()

    month_last = month_last.drop(columns=["month"])
    return month_last
