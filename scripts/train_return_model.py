from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline


from packages.ml.modeling import FEATURE_COLUMNS, prepare_monthly_dataset, train_return_model
from packages.core.db import db_session, get_database_url
from packages.db.models import MlModel


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

    df = prepare_monthly_dataset(pd.read_csv(args.data))
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            "monthly_dataset.csv is missing expected feature columns. "
            f"Missing: {missing}. Re-run scripts/build_features.py."
        )

    train, test = time_split(df, train_ratio=0.8)

    X_train = train[["symbol"] + FEATURE_COLUMNS]
    y_train = train["target_fwd_ret"].astype(float)
    X_test = test[["symbol"] + FEATURE_COLUMNS]
    y_test = test["target_fwd_ret"].astype(float)

    model: Pipeline = train_return_model(train)
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
    model_path = out_dir / "model.joblib"
    joblib.dump(model, model_path)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Compute artifact hash for registry
    sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()

    # Best-effort: write to DB model registry if available.
    try:
        _ = get_database_url()
        with db_session() as session:
            # Infer symbols from dataset
            symbols = sorted(df["symbol"].unique().tolist())
            reg = MlModel(
                name="return_model",
                task="predict_next_month_return",
                data_frequency="monthly",
                feature_schema=str(df["feature_schema"].iloc[0]) if "feature_schema" in df.columns else "monthly_v2",
                symbols=symbols,
                train_params={"model": "Ridge", "alpha": 1.0, "train_ratio": 0.8},
                metrics=metrics,
                artifact_path=str(model_path).replace("\\", "/"),
                artifact_sha256=sha256,
            )
            session.add(reg)
            session.commit()
            print(f"Registered model in DB: {reg.id}")
    except Exception:
        pass

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
