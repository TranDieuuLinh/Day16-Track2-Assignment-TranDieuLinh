#!/usr/bin/env python3
"""LightGBM benchmark on Credit Card Fraud Detection dataset."""

import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

DATA_DIR = Path(os.environ.get("BENCHMARK_DATA_DIR", Path(__file__).parent))
DATA_FILE = DATA_DIR / "creditcard.csv"
OUTPUT_FILE = DATA_DIR / "benchmark_result.json"
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "t3.micro")


def find_data_file() -> Path:
    candidates = [
        DATA_DIR / "creditcard.csv",
        DATA_DIR / "ml-benchmark" / "creditcard.csv",
        Path.home() / "ml-benchmark" / "creditcard.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find creditcard.csv. Checked: {[str(p) for p in candidates]}"
    )


def main() -> None:
    data_path = find_data_file()
    print(f"Dataset: {data_path}")
    print(f"Instance type: {INSTANCE_TYPE}")
    print("-" * 50)

    load_start = time.perf_counter()
    df = pd.read_csv(data_path)
    load_time = time.perf_counter() - load_start

    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": 42,
        "num_threads": os.cpu_count() or 1,
    }

    train_start = time.perf_counter()
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    train_time = time.perf_counter() - train_start
    best_iteration = model.best_iteration

    y_pred_proba = model.predict(X_test, num_iteration=best_iteration)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc_roc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    single_row = X_test.iloc[[0]]
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        model.predict(single_row, num_iteration=best_iteration)
        latencies.append((time.perf_counter() - start) * 1000)
    inference_latency_1_row_ms = float(np.median(latencies))

    batch = X_test.iloc[:1000]
    throughput_start = time.perf_counter()
    model.predict(batch, num_iteration=best_iteration)
    throughput_time = time.perf_counter() - throughput_start
    inference_throughput_1000_rows = 1000 / throughput_time

    results = {
        "instance_type": INSTANCE_TYPE,
        "dataset": "mlg-ulb/creditcardfraud",
        "num_rows": len(df),
        "num_features": X.shape[1],
        "data_load_time_seconds": round(load_time, 4),
        "training_time_seconds": round(train_time, 4),
        "best_iteration": int(best_iteration),
        "auc_roc": round(float(auc_roc), 6),
        "accuracy": round(float(accuracy), 6),
        "f1_score": round(float(f1), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "inference_latency_1_row_ms": round(inference_latency_1_row_ms, 4),
        "inference_throughput_1000_rows_per_sec": round(
            float(inference_throughput_1000_rows), 2
        ),
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Thời gian load data:              {results['data_load_time_seconds']} s")
    print(f"Thời gian training:              {results['training_time_seconds']} s")
    print(f"Best iteration:                  {results['best_iteration']}")
    print(f"AUC-ROC:                         {results['auc_roc']}")
    print(f"Accuracy:                        {results['accuracy']}")
    print(f"F1-Score:                        {results['f1_score']}")
    print(f"Precision:                       {results['precision']}")
    print(f"Recall:                          {results['recall']}")
    print(
        f"Inference latency (1 row):       {results['inference_latency_1_row_ms']} ms"
    )
    print(
        "Inference throughput (1000 rows): "
        f"{results['inference_throughput_1000_rows_per_sec']} rows/s"
    )
    print("-" * 50)
    print(f"Saved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
