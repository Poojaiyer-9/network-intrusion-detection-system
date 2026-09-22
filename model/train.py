#!/usr/bin/env python3
"""
Train the Random Forest intrusion-detection model and save deployable
artifacts to model/artifacts/.

Usage:
    python model/train.py

Data source (checked in this order):
  1. Real KDD Cup 99 10%-subset, if present at:
       data/kddcup.data_10_percent_corrected
     (download from http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
     — see data/README.md)
  2. Otherwise, falls back to a synthetic, schema-accurate dataset
     (model/synthetic.py) so the pipeline still runs end-to-end. Artifacts
     built this way are tagged "source": "synthetic" in metadata.json and
     are a DEMO ONLY — retrain on the real dataset before trusting accuracy
     numbers.
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
REAL_DATA_PATH = ROOT / "data" / "kddcup.data_10_percent_corrected"

# Capped so the serialized model stays small enough for a serverless
# deployment bundle. Raise these once training on the full real dataset if
# bundle size allows (see README's "Model size & Vercel limits" section).
N_ESTIMATORS = 30
MAX_DEPTH = 14


def load_raw_dataframe() -> tuple[pd.DataFrame, str]:
    if REAL_DATA_PATH.exists():
        print(f"Loading real dataset from {REAL_DATA_PATH}")
        df = pd.read_csv(REAL_DATA_PATH, names=RAW_COLUMNS)
        return df, "real"

    print(
        "Real dataset not found at "
        f"{REAL_DATA_PATH.relative_to(ROOT)} — generating a synthetic "
        "schema-accurate dataset instead (demo only, see data/README.md)."
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
    # Fit on plain ndarrays (not the DataFrame) so the scaler doesn't record
    # feature names — inference sends a positional vector, not a DataFrame,
    # and a name mismatch there would otherwise emit an sklearn UserWarning
    # on every request.
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
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
    if source == "synthetic":
        print(
            "\nNOTE: this model was trained on SYNTHETIC data and is a demo "
            "only. Retrain with the real KDD Cup 99 dataset before relying "
            "on it (see data/README.md)."
        )


if __name__ == "__main__":
    main()
