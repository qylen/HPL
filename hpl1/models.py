"""Small, explicit baseline models for predictive-law research."""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(name: str, alpha: float = 10.0):
    name = name.lower()
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    if name in {"hist", "hist_gbr", "gradient_boosting"}:
        # No random early-stopping split: temporal validation is done by the caller.
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=100,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=7,
        )
    raise ValueError("model must be 'ridge' or 'hist'")
