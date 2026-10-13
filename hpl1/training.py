"""Chronological training, validation-time model comparison, artifact manifest.

Selection rule: models are fit on the train split and compared on the
validation split ONLY. The best-by-validation model is then evaluated once on
the untouched test split. The test split is never used for any tuning input.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from .models import BaseModel, make_model, save_model


def chronological_split(n: int, train_fraction: float = 0.6, validation_fraction: float = 0.2,
                        purge: int = 1, embargo: int = 30) -> dict[str, slice]:
    """Return train/val/test index slices with purge + embargo at boundaries.

    purge: rows removed from the end of each earlier split whose label horizon
    (1 second) extends into the next split. embargo: extra rows skipped after
    each earlier split before the next one starts (serial-correlation buffer).
    """
    purge = max(purge, 1)
    embargo = max(embargo, 0)
    train_end = int(n * train_fraction)
    valid_end = int(n * (train_fraction + validation_fraction))
    splits = {
        "train": slice(0, max(0, train_end - purge)),
        "validation": slice(train_end + embargo, max(0, valid_end - purge)),
        "test": slice(valid_end + embargo, n),
    }
    if splits["train"].stop < 30 or (splits["validation"].stop - splits["validation"].start) < 10 \
            or (splits["test"].stop - splits["test"].start) < 10:
        raise ValueError(
            f"not enough rows for chronological splits (n={n}); "
            "need substantially more data or smaller purge/embargo")
    return splits


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    ok = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[ok], predicted[ok]
    n = len(actual)
    if n == 0:
        return {"rows": 0}
    mse = float(np.mean((actual - predicted) ** 2))
    base_mse = float(np.mean(actual ** 2))
    std_p, std_a = float(np.std(predicted)), float(np.std(actual))
    corr = float(np.corrcoef(actual, predicted)[0, 1]) if n > 2 and std_p > 0 and std_a > 0 else 0.0
    # Directional accuracy is computed over NON-ZERO actual returns: with a
    # 0.1 tick on ~70k prices, ~63% of one-second mid returns are exactly zero,
    # and zero-return seconds would otherwise dominate raw sign agreement.
    nz = actual != 0
    dir_acc = float(np.mean(np.sign(actual[nz]) == np.sign(predicted[nz]))) if nz.any() else None
    zero_frac = float(np.mean(~nz))
    up_mask = predicted > 0
    down_mask = predicted < 0
    cond_up_mean_bps = float(np.mean(actual[up_mask])) * 1e4 if up_mask.sum() > 20 else None
    cond_down_mean_bps = float(np.mean(actual[down_mask])) * 1e4 if down_mask.sum() > 20 else None
    return {
        "rows": int(n),
        "mse": mse,
        "zero_forecast_mse": base_mse,
        "oos_r2_vs_zero": float(1.0 - mse / base_mse) if base_mse > 0 else 0.0,
        "mae": float(np.mean(np.abs(actual - predicted))),
        "directional_accuracy_nonzero": dir_acc,
        "zero_return_fraction": zero_frac,
        "directional_accuracy_note": ("sign agreement over non-zero returns; "
                                      "zero-tick seconds excluded"),
        "conditional_up_mean_bps": cond_up_mean_bps,
        "conditional_down_mean_bps": cond_down_mean_bps,
        "prediction_return_correlation": corr,
    }


def compare_models(feature_frame: pd.DataFrame, feature_columns: list[str],
                   model_names: list[str], seed: int = 13,
                   train_fraction: float = 0.6, validation_fraction: float = 0.2,
                   purge: int = 1, embargo: int = 30,
                   artifact_dir: str | None = None,
                   run_manifest: dict | None = None) -> dict[str, Any]:
    """Train all candidates on train, select on validation, test once for the best."""
    df = feature_frame
    needed = feature_columns + ["target_return", "valid"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"feature frame missing columns: {missing}")
    price_cols = [c for c in ("bid", "ask", "mid", "price", "spread_bps",
                               "funding_rate", "valid") if c in df.columns]
    work = df[list(dict.fromkeys(feature_columns + ["target_return", "ts", "second"] + price_cols))].copy()
    work = work.dropna(subset=feature_columns + ["target_return"])
    work = work[work["valid"].fillna(False)] if "valid" in work.columns else work
    work = work.reset_index(drop=True)
    n = len(work)
    if n < 200:
        raise ValueError(f"only {n} usable labeled rows; >=200 required for comparison")
    splits = chronological_split(n, train_fraction, validation_fraction, purge, embargo)
    X, y = work[feature_columns], work["target_return"].to_numpy(dtype=float)

    results: dict[str, dict] = {}
    fitted: dict[str, BaseModel] = {}
    for name in model_names:
        model = make_model(name, seed=seed)
        t0 = time.time()
        model.fit(X.iloc[splits["train"]], y[splits["train"]])
        val_pred = model.predict(X.iloc[splits["validation"]])
        val_metrics = metrics(y[splits["validation"]], val_pred)
        val_metrics["fit_seconds"] = round(time.time() - t0, 2)
        results[name] = {"validation": val_metrics, "describe": model.describe()}
        fitted[name] = model

    ranked = sorted(results.items(), key=lambda kv: kv[1]["validation"].get("oos_r2_vs_zero", -9e9),
                    reverse=True)
    best_name = ranked[0][0]
    best = fitted[best_name]
    best_val_pred = best.predict(X.iloc[splits["validation"]])
    val_actual = y[splits["validation"]]
    test_pred = best.predict(X.iloc[splits["test"]])
    test_metrics = metrics(y[splits["test"]], test_pred)
    results[best_name]["test"] = test_metrics

    # Test predictions + context needed by the backtester. Price/context
    # columns come from `work` itself: its index was reset after dropna, so
    # looking them up in the ORIGINAL df would silently misalign rows.
    pred_frame = work.iloc[splits["test"]][["ts", "second"]].copy()
    pred_frame["prediction"] = test_pred
    for col in price_cols:
        pred_frame[col] = work.iloc[splits["test"]][col].to_numpy()
    pred_frame["actual_return"] = y[splits["test"]]

    manifest = {
        "selected_model": best_name,
        "selection_rule": "max validation oos_r2_vs_zero; test evaluated once",
        "models_compared": model_names,
        "rows_usable": n,
        "splits": {k: {"start": int(v.start), "stop": int(v.stop),
                       "rows": int(v.stop - v.start)} for k, v in splits.items()},
        "purge_seconds": purge,
        "embargo_seconds": embargo,
        "feature_columns": feature_columns,
        "random_seed": seed,
        "trained_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **(run_manifest or {}),
    }
    saved_paths = {}
    if artifact_dir:
        from pathlib import Path

        ad = Path(artifact_dir)
        ad.mkdir(parents=True, exist_ok=True)
        for name, model in fitted.items():
            p = ad / f"model_{name}.joblib"
            save_model(model, p, meta={"manifest": manifest, "name": name})
            saved_paths[name] = str(p)
        pred_path = ad / "test_predictions.parquet"
        pred_frame.reset_index(drop=True).to_parquet(pred_path, index=False)
        saved_paths["predictions"] = str(pred_path)
    return {"results": results, "best": best_name, "manifest": manifest,
            "predictions": pred_frame, "saved": saved_paths, "splits": splits,
            "validation_predictions": {"prediction": best_val_pred, "actual": val_actual}}


# ----------------------------------------------------------------------------
# Legacy CSV/JSONL path (kept for compatibility with `hpl1 collect` workflows)
# ----------------------------------------------------------------------------

def train_and_evaluate(
    data_path,
    model_name: str = "ridge",
    artifact_path="artifacts/hpl1.joblib",
    predictions_path="artifacts/test_predictions.csv",
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    ridge_alpha: float = 10.0,
) -> dict:
    """Train one model on a feature CSV; chronological split; test once.

    Kept for the legacy JSONL collection path. New research should use
    compare_models(), which selects among candidates on validation only.
    """
    from pathlib import Path

    import joblib

    from .features import FEATURE_COLUMNS

    df = pd.read_csv(data_path)
    needed = set(FEATURE_COLUMNS + ["target_return", "timestamp", "spread_bps"])
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for col in FEATURE_COLUMNS + ["target_return", "spread_bps"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=FEATURE_COLUMNS + ["target_return", "spread_bps", "timestamp"])
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    n = len(df)
    if n < 100:
        raise ValueError(f"Only {n} complete rows; >=100 required for a basic chronological experiment.")

    splits = chronological_split(n, train_fraction, validation_fraction, purge=1, embargo=0)
    X, y = df[FEATURE_COLUMNS], df["target_return"].to_numpy(dtype=float)

    model = make_model(model_name, alpha=ridge_alpha) if model_name == "ridge" else make_model(model_name)
    model.fit(X.iloc[splits["train"]], y[splits["train"]])
    validation_metrics = metrics(y[splits["validation"]], model.predict(X.iloc[splits["validation"]]))

    # Freeze decisions before the final test; refit on train+validation.
    final = make_model(model_name, alpha=ridge_alpha) if model_name == "ridge" else make_model(model_name)
    final.fit(X.iloc[:splits["validation"].stop], y[:splits["validation"].stop])
    test_predictions = final.predict(X.iloc[splits["test"]])
    test_metrics = metrics(y[splits["test"]], test_predictions)

    pred_frame = pd.DataFrame({
        "timestamp": df.loc[splits["test"], "timestamp"].astype(str).to_numpy(),
        "actual_return": y[splits["test"]],
        "prediction": test_predictions,
        "spread_bps": df.loc[splits["test"], "spread_bps"].to_numpy(dtype=float),
    })
    pred_path = Path(predictions_path)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_frame.to_csv(pred_path, index=False)

    artifact = {
        "model": final, "model_name": model_name,
        "feature_columns": list(FEATURE_COLUMNS),
        "trained_rows": int(splits["validation"].stop),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "warning": "Research artifact only; no guarantee of predictive power or profitability.",
    }
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)
    report = {
        "model_name": model_name, "total_rows": n,
        "train_rows": int(splits["train"].stop),
        "validation_rows": int(splits["validation"].stop - splits["validation"].start),
        "test_rows": int(splits["test"].stop - splits["test"].start),
        "validation_metrics": validation_metrics, "test_metrics": test_metrics,
        "artifact_path": str(artifact_path), "predictions_path": str(pred_path),
    }
    import json as _json
    artifact_path.with_suffix(".metrics.json").write_text(_json.dumps(report, indent=2), encoding="utf-8")
    return report
