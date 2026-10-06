#!/usr/bin/env python3
"""
Train the Random Forest intrusion-detection model and save deployable
artifacts to model/artifacts/.

Usage:
    python model/train.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.preprocessing import (  # noqa: E402
    CLASS_LABELS, FINAL_FEATURE_ORDER, RAW_COLUMNS, dataframe_to_features_and_labels,
)

ARTIFACTS_DIR = ROOT / "model" / "artifacts"
NSL_DATA_PATH = ROOT / "data" / "KDDTrain+.txt"
REAL_DATA_PATH = ROOT / "data" / "kddcup.data_10_percent_corrected"

N_ESTIMATORS = 40
MAX_DEPTH = 18


def load_raw_dataframe() -> tuple[pd.DataFrame, str]:
    if NSL_DATA_PATH.exists():
        print(f"Loading benchmark dataset from {NSL_DATA_PATH}")
        df = pd.read_csv(NSL_DATA_PATH, header=None)
        if df.shape[1] == 43:
            df.columns = RAW_COLUMNS + ["difficulty"]
            df = df.drop(columns=["difficulty"])
        else:
            df.columns = RAW_COLUMNS
        return df, "NSL-KDD (125K+ benchmark)"

    if REAL_DATA_PATH.exists():
        print(f"Loading real dataset from {REAL_DATA_PATH}")
        df = pd.read_csv(REAL_DATA_PATH, names=RAW_COLUMNS)
        return df, "KDD Cup 99 (real data)"

    print(
        "Real dataset not found — generating a synthetic "
        "schema-accurate dataset instead (see data/README.md)."
    )
    from model.synthetic import generate_synthetic_raw_dataframe

    df = generate_synthetic_raw_dataframe(n_per_class=4000)
    return df, "synthetic"


def main() -> None:
    raw_df, source = load_raw_dataframe()
    X, y = dataframe_to_features_and_labels(raw_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify=y
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    start = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start

    train_acc = accuracy_score(y_train, model.predict(X_train_scaled)) * 100
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled)) * 100

    print(f"Trained on {source} data: {len(X_train)} train / {len(X_test)} test rows")
    print(f"Train accuracy: {train_acc:.2f}%  Test accuracy: {test_acc:.2f}%  ({train_time:.2f}s)")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS_DIR / "model.joblib", compress=3)
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib", compress=3)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_accuracy_pct": round(train_acc, 2),
        "test_accuracy_pct": round(test_acc, 2),
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "feature_order": FINAL_FEATURE_ORDER,
        "class_labels": sorted(set(y.tolist())) or CLASS_LABELS,
        "sklearn_version": sklearn.__version__,
        "model_size_bytes": (ARTIFACTS_DIR / "model.joblib").stat().st_size,
    }
    with open(ARTIFACTS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved artifacts to {ARTIFACTS_DIR}")
    print(f"model.joblib size: {metadata['model_size_bytes'] / 1024:.1f} KB")


if __name__ == "__main__":
    main()
