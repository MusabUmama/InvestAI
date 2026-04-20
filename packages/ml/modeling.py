from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_21d",
    "vol_21d",
    "vol_63d",
    "mom_63d",
    "mom_126d",
    "sma21_ratio",
    "sma63_ratio",
]


def prepare_monthly_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].astype(str)

    out = out.dropna(subset=["target_fwd_ret"])
    for c in FEATURE_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=FEATURE_COLUMNS)
    return out


def train_return_model(train_df: pd.DataFrame) -> Pipeline:
    X = train_df[["symbol"] + FEATURE_COLUMNS]
    y = train_df["target_fwd_ret"].astype(float)

    pre = ColumnTransformer(
        transformers=[
            ("sym", OneHotEncoder(handle_unknown="ignore"), ["symbol"]),
            ("num", "passthrough", FEATURE_COLUMNS),
        ]
    )

    model = Pipeline(
        steps=[
            ("pre", pre),
            ("reg", Ridge(alpha=1.0, random_state=0)),
        ]
    )
    model.fit(X, y)
    return model


def predict_expected_returns(model: Pipeline, feature_rows: pd.DataFrame) -> np.ndarray:
    X = feature_rows[["symbol"] + FEATURE_COLUMNS].copy()
    X["symbol"] = X["symbol"].astype(str)
    for c in FEATURE_COLUMNS:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0.0)
    pred = model.predict(X)
    return np.asarray(pred, dtype=float)

