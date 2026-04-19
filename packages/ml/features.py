from __future__ import annotations

import pandas as pd


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: daily bars for a single symbol with columns at least: date, adjusted_close (or close)
    Output: same rows with basic technical features.
    """
    out = df.copy()
    out = out.sort_values("date")

    price_col = "adjusted_close" if "adjusted_close" in out.columns else "close"
    if price_col not in out.columns:
        raise ValueError("Expected adjusted_close or close column")

    price = pd.to_numeric(out[price_col], errors="coerce")
    out["ret_1d"] = price.pct_change(1)
    out["ret_5d"] = price.pct_change(5)
    out["ret_21d"] = price.pct_change(21)

    out["vol_21d"] = out["ret_1d"].rolling(21).std()
    out["vol_63d"] = out["ret_1d"].rolling(63).std()

    out["mom_63d"] = price.pct_change(63)
    out["mom_126d"] = price.pct_change(126)

    # Simple moving average ratios (trend proxy)
    sma_21 = price.rolling(21).mean()
    sma_63 = price.rolling(63).mean()
    out["sma21_ratio"] = price / sma_21 - 1.0
    out["sma63_ratio"] = price / sma_63 - 1.0

    return out


def to_monthly_dataset(daily: pd.DataFrame, horizon_days: int = 21) -> pd.DataFrame:
    """
    Converts daily features to a monthly-ish supervised dataset:
    - uses the last available row per calendar month as the feature row
    - target is forward return over `horizon_days` on adjusted_close
    """
    df = daily.copy()
    df = df.sort_values(["symbol", "date"])

    price_col = "adjusted_close" if "adjusted_close" in df.columns else "close"
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    month_last = df.groupby(["symbol", "month"], as_index=False).tail(1)

    # Forward return target: next period return from feature date
    month_last["target_fwd_ret"] = (
        month_last.groupby("symbol")[price_col].shift(-1) / month_last[price_col] - 1.0
    )
    # Also include a volatility-like target for later models
    month_last["target_fwd_vol"] = month_last.groupby("symbol")["ret_1d"].shift(-1).rolling(3).std()

    month_last = month_last.drop(columns=["month"])
    return month_last

