from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
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


def time_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["date", "symbol"])
    unique_dates = sorted(pd.to_datetime(df["date"]).unique())
    if len(unique_dates) < 20:
        raise RuntimeError("Not enough dates for a reasonable time split")
    cutoff_idx = int(len(unique_dates) * train_ratio)
    cutoff_date = unique_dates[cutoff_idx]
    train = df[pd.to_datetime(df["date"]) <= cutoff_date]
    test = df[pd.to_datetime(df["date"]) > cutoff_date]
    return train, test


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Train a simple return prediction model from monthly_dataset.csv.")
    parser.add_argument("--data", default="data/features/monthly_dataset.csv")
    parser.add_argument("--out-dir", default="artifacts/return_model")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)

    # Basic cleanup
    df = df.dropna(subset=["target_fwd_ret"])
    for c in FEATURE_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURE_COLUMNS)

    train, test = time_split(df, train_ratio=0.8)

    X_train = train[["symbol"] + FEATURE_COLUMNS]
    y_train = train["target_fwd_ret"].astype(float)
    X_test = test[["symbol"] + FEATURE_COLUMNS]
    y_test = test["target_fwd_ret"].astype(float)

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

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred)),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

