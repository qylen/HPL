"""Interpretability and residual checks; treat test-set discoveries as exploratory."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def residual_autocorrelation(residuals: np.ndarray, lags: tuple[int, ...] = (1, 2, 5, 10, 30, 60)) -> dict[str, float | None]:
    """Sample residual autocorrelation at selected lags (diagnostic, not a test battery)."""
    x = np.asarray(residuals, dtype=float)
    x = x[np.isfinite(x)]
    result: dict[str, float | None] = {}
    for lag in lags:
        if len(x) <= lag + 2 or np.std(x) == 0:
            result[f"lag_{lag}"] = None
            continue
        corr = np.corrcoef(x[lag:], x[:-lag])[0, 1]
        result[f"lag_{lag}"] = float(corr) if np.isfinite(corr) else None
    return result


def feature_diagnostics(
    model: Any,
    model_name: str,
    feature_columns: list[str],
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
    seed: int = 7,
) -> pd.DataFrame:
    """Create a compact feature diagnostic table without using final test labels.

    Ridge coefficients are in standardized feature units. For the tree model,
    permutation importance is calculated on a bounded validation sample; it can
    be distorted by correlated features and is a diagnostic, not causal proof.
    """
    if model_name == "ridge":
        ridge = model.named_steps["ridge"]
        values = np.asarray(ridge.coef_, dtype=float)
        return pd.DataFrame({
            "feature": feature_columns,
            "diagnostic": "standardized_ridge_coefficient",
            "value": values,
            "absolute_value": np.abs(values),
        }).sort_values("absolute_value", ascending=False).reset_index(drop=True)

    if len(X_validation) > 5000:
        step = int(np.ceil(len(X_validation) / 5000))
        X_sample = X_validation.iloc[::step]
        y_sample = np.asarray(y_validation)[::step]
    else:
        X_sample = X_validation
        y_sample = np.asarray(y_validation)
    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring="neg_mean_squared_error",
        n_repeats=3,
        random_state=seed,
        n_jobs=1,
    )
    return pd.DataFrame({
        "feature": feature_columns,
        "diagnostic": "validation_permutation_mse_increase",
        "value": result.importances_mean,
        "absolute_value": np.abs(result.importances_mean),
        "standard_deviation": result.importances_std,
    }).sort_values("absolute_value", ascending=False).reset_index(drop=True)
