"""Chronological model training, evaluation and artifact persistence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .diagnostics import feature_diagnostics, residual_autocorrelation
from .features import FEATURE_COLUMNS
from .models import make_model


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mse = float(mean_squared_error(actual, predicted))
    baseline_mse = float(np.mean(actual**2))
    corr = float(np.corrcoef(actual, predicted)[0, 1]) if len(actual) > 2 and np.std(predicted) > 0 and np.std(actual) > 0 else 0.0
    return {
        "rows": int(len(actual)),
        "mse": mse,
        "zero_forecast_mse": baseline_mse,
        "oos_r2_vs_zero": float(1.0 - mse / baseline_mse) if baseline_mse > 0 else 0.0,
        "mae": float(mean_absolute_error(actual, predicted)),
        "directional_accuracy": float(np.mean(np.sign(actual) == np.sign(predicted))),
        "prediction_return_correlation": corr,
    }


def train_and_evaluate(
    data_path: str | Path,
    model_name: str = "ridge",
    artifact_path: str | Path = "artifacts/hpl1.joblib",
    predictions_path: str | Path = "artifacts/test_predictions.csv",
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    ridge_alpha: float = 10.0,
) -> dict[str, Any]:
    """Fit on first 60%, inspect next 20%, then refit on first 80% and test last 20%.

    Hyperparameters/model family should be fixed before inspecting final test metrics.
    The final artifact is trained on train+validation and never sees test labels.
    """
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1 and train_fraction + validation_fraction < 1):
        raise ValueError("train_fraction and validation_fraction must be positive and sum to less than 1")
    df = pd.read_csv(data_path)
    needed = set(FEATURE_COLUMNS + ["target_return", "timestamp", "spread_bps"])
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for col in FEATURE_COLUMNS + ["target_return", "spread_bps"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS + ["target_return", "spread_bps", "timestamp"])
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    n = len(df)
    if n < 100:
        raise ValueError(f"Only {n} complete rows. At least 100 are required for a basic chronological experiment; more is strongly recommended.")

    train_end = int(n * train_fraction)
    valid_end = int(n * (train_fraction + validation_fraction))
    # Each row t uses events through the end of second t and labels the return to
    # second t+1. Purge one row at split boundaries so a training label cannot
    # depend on the first unseen bucket at the time a frozen model would be fit.
    validation_start, validation_stop = train_end, valid_end - 1
    test_start = valid_end
    if train_end < 31 or validation_stop - validation_start < 10 or n - test_start < 10:
        raise ValueError("Not enough rows for the requested chronological splits after boundary purging")

    X = df[FEATURE_COLUMNS]
    y = df["target_return"].to_numpy(dtype=float)
    model = make_model(model_name, alpha=ridge_alpha)
    model.fit(X.iloc[:train_end - 1], y[:train_end - 1])
    X_validation = X.iloc[validation_start:validation_stop]
    validation_predictions = model.predict(X_validation)
    validation_actual = y[validation_start:validation_stop]
    validation_metrics = _metrics(validation_actual, validation_predictions)
    diagnostics = feature_diagnostics(
        model, model_name, list(FEATURE_COLUMNS), X_validation, validation_actual
    )
    diagnostics_path = Path(artifact_path).with_suffix(".feature_diagnostics.csv")
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(diagnostics_path, index=False)

    # Freeze all decisions before the final test. The final validation row is
    # purged because its label ends in the first test bucket.
    final_model = make_model(model_name, alpha=ridge_alpha)
    final_model.fit(X.iloc[:valid_end - 1], y[:valid_end - 1])
    test_predictions = final_model.predict(X.iloc[test_start:])
    test_actual = y[test_start:]
    test_metrics = _metrics(test_actual, test_predictions)
    test_metrics["residual_autocorrelation"] = residual_autocorrelation(test_actual - test_predictions)

    pred_frame = pd.DataFrame({
        "timestamp": df.loc[test_start:, "timestamp"].astype(str).to_numpy(),
        "actual_return": test_actual,
        "prediction": test_predictions,
        "spread_bps": df.loc[test_start:, "spread_bps"].to_numpy(dtype=float),
    })
    pred_path = Path(predictions_path)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_frame.to_csv(pred_path, index=False)

    artifact = {
        "model": final_model,
        "model_name": model_name,
        "feature_columns": list(FEATURE_COLUMNS),
        "ridge_alpha": ridge_alpha if model_name == "ridge" else None,
        "trained_rows": int(valid_end - 1),
        "train_end_timestamp": str(df.loc[train_end - 2, "timestamp"]),
        "validation_end_timestamp": str(df.loc[validation_stop - 1, "timestamp"]),
        "test_start_timestamp": str(df.loc[test_start, "timestamp"]),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "warning": "Research artifact only; no guarantee of predictive power or profitability.",
    }
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)

    report = {
        "model_name": model_name,
        "total_rows": n,
        "train_rows": train_end - 1,
        "validation_rows": validation_stop - validation_start,
        "test_rows": n - test_start,
        "purged_boundary_rows": 2,
        "train_end_timestamp": str(df.loc[train_end - 2, "timestamp"]),
        "validation_end_timestamp": str(df.loc[validation_stop - 1, "timestamp"]),
        "test_start_timestamp": str(df.loc[test_start, "timestamp"]),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "artifact_path": str(artifact_path),
        "predictions_path": str(pred_path),
        "feature_diagnostics_path": str(diagnostics_path),
    }
    report_path = artifact_path.with_suffix(".metrics.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
