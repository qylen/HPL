"""Candidate models with a consistent interface for identical-condition comparison.

Registry:
- zero      : zero-return benchmark
- ridge     : standardized ridge regression (linear baseline)
- logistic  : directional probabilistic baseline (P(up)) with magnitude calibration
- hist      : histogram gradient boosting (nonlinear)
- law       : hidden-law interaction model  E[r|X] = b1*I + b2*I*L + b3*I*sigma + controls
- regime    : volatility-regime-conditioned ridge (regime quantiles fit on train only)

All models are deterministic given the seed. Scalers/quantiles are fit on the
TRAINING split only by construction (fit() receives train data).
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-12


class BaseModel:
    name = "base"
    predicts_probabilities = False

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> BaseModel:
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def save(self, path) -> None:
        joblib.dump({"name": self.name, "model": self, "feature_names": list(getattr(self, "feature_names_", []))}, path)

    @classmethod
    def load(cls, path) -> BaseModel:
        blob = joblib.load(path)
        model = blob["model"]
        if not isinstance(model, cls):
            raise TypeError(f"artifact holds a {type(model).__name__}, not {cls.__name__}")
        return model

    def describe(self) -> dict:
        return {"name": self.name, "type": type(self).__name__}


class ZeroModel(BaseModel):
    name = "zero"

    def fit(self, X, y):
        self.feature_names_ = list(X.columns)
        self._mean_abs_y_ = float(np.mean(np.abs(y))) if len(y) else 0.0
        return self

    def predict(self, X):
        return np.zeros(len(X))


class RidgeModel(BaseModel):
    name = "ridge"

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.feature_names_ = list(X.columns)
        self.pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=self.alpha))])
        self.pipe.fit(X, y)
        return self

    def predict(self, X):
        return self.pipe.predict(X)

    def describe(self):
        d = super().describe()
        d["alpha"] = self.alpha
        d["standardized_coefficients"] = dict(zip(self.feature_names_,
                                                  np.asarray(self.pipe.named_steps["ridge"].coef_).round(6).tolist()))
        return d


class LogisticModel(BaseModel):
    """Directional baseline: P(r>0); forecast = (2p-1) * train E[|r|]."""
    name = "logistic"
    predicts_probabilities = True

    def __init__(self, C: float = 1.0, max_iter: int = 500):
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y):
        self.feature_names_ = list(X.columns)
        self.scale_ = float(np.mean(np.abs(y))) if len(y) else 0.0
        sign = (np.asarray(y) > 0).astype(int)
        if len(np.unique(sign)) < 2:
            self.constant_ = float(sign[0]) if len(sign) else 0.0
            self.pipe = None
            return self
        self.pipe = Pipeline([("scaler", StandardScaler()),
                              ("logit", LogisticRegression(C=self.C, max_iter=self.max_iter))])
        self.pipe.fit(X, sign)
        return self

    def predict(self, X):
        if getattr(self, "pipe", None) is None:
            return np.full(len(X), (2 * getattr(self, "constant_", 0.0) - 1) * self.scale_)
        p = self.pipe.predict_proba(X)[:, 1]
        return (2 * p - 1) * self.scale_


class HistGBModel(BaseModel):
    name = "hist"

    def __init__(self, seed: int = 13, max_iter: int = 150, learning_rate: float = 0.06,
                 max_leaf_nodes: int = 31, min_samples_leaf: int = 200):
        self.seed = seed
        self.params = dict(loss="squared_error", learning_rate=learning_rate, max_iter=max_iter,
                           max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                           l2_regularization=1.0, early_stopping=False, random_state=seed)

    def fit(self, X, y):
        self.feature_names_ = list(X.columns)
        self.gbr = HistGradientBoostingRegressor(**self.params)
        self.gbr.fit(X, y)
        return self

    def predict(self, X):
        return self.gbr.predict(X)


class HiddenLawModel(BaseModel):
    """E[r(t+1)|X] = b1*I + b2*I*L + b3*I*sigma + controls.

    I = ofi_norm (order-flow imbalance), L = 1/log1p(depth_total) (thinness),
    sigma = volatility_60s (trailing), controls = spread_bps, ret_lag1,
    trade_imbalance. Falls back to nearest equivalents when a column is absent.
    """
    name = "law"

    LAW_TERMS = ("I", "I_L", "I_sigma")

    @staticmethod
    def _design(X: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        if "ofi_norm" in X.columns:
            I = X["ofi_norm"].to_numpy(dtype=float)
        elif "trade_imbalance" in X.columns:
            I = X["trade_imbalance"].to_numpy(dtype=float)
        else:
            I = np.zeros(len(X))
        if "depth_total" in X.columns and X["depth_total"].notna().any():
            depth = X["depth_total"].fillna(0).clip(lower=0).to_numpy(dtype=float)
            L = 1.0 / (np.log1p(depth) + 1.0)
        else:
            L = np.zeros(len(X))
        sig_col = next((c for c in ("volatility_60s", "volatility_30s") if c in X.columns), None)
        sigma = X[sig_col].fillna(0).to_numpy(dtype=float) if sig_col else np.zeros(len(X))
        cols["I"] = I
        cols["I_L"] = I * L
        cols["I_sigma"] = I * sigma
        for c in ("spread_bps", "ret_lag1", "trade_imbalance"):
            if c in X.columns:
                cols[f"ctrl_{c}"] = X[c].fillna(0).to_numpy(dtype=float)
        return pd.DataFrame(cols, index=X.index)

    def fit(self, X, y):
        self.feature_names_ = list(X.columns)
        D = self._design(X)
        self.design_names_ = list(D.columns)
        self.pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        self.pipe.fit(D, np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        D = self._design(X)
        return self.pipe.predict(D)

    def describe(self):
        d = super().describe()
        d["law_terms"] = dict(zip(self.design_names_,
                                  np.asarray(self.pipe.named_steps["ridge"].coef_).round(8).tolist()))
        return d


class RegimeRidge(BaseModel):
    """Volatility-regime-conditioned ridge. Regime cut points (terciles of
    volatility_60s) are estimated on TRAIN only; prediction uses those cuts.
    """
    name = "regime"

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha

    def _sig(self, X: pd.DataFrame) -> np.ndarray:
        col = next((c for c in ("volatility_60s", "volatility_30s") if c in X.columns), None)
        if col is None:
            return np.zeros(len(X))
        return X[col].fillna(0).to_numpy(dtype=float)

    def fit(self, X, y):
        self.feature_names_ = list(X.columns)
        sig = self._sig(X)
        q1, q2 = np.quantile(sig, [1 / 3, 2 / 3])
        self.cuts_ = (float(q1), float(q2))
        regime = np.digitize(sig, [q1, q2])
        self.models_ = {}
        for r in (0, 1, 2):
            m = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=self.alpha))])
            mask = regime == r
            if mask.sum() >= 10:
                m.fit(X.loc[mask] if isinstance(X, pd.DataFrame) else X[mask], np.asarray(y)[mask])
                self.models_[r] = m
        self._any_row_warned_ = False
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # NOTE: reshape single rows explicitly (broadcasting landmine).
        if len(X) == 0:
            return np.zeros(0)
        sig = self._sig(X)
        regime = np.digitize(sig, self.cuts_)
        out = np.zeros(len(X))
        for r in (0, 1, 2):
            mask = regime == r
            if mask.any() and r in self.models_:
                Xr = X.loc[mask] if isinstance(X, pd.DataFrame) else X[mask]
                out[mask] = self.models_[r].predict(Xr)
        return out

    def describe(self):
        d = super().describe()
        d["regime_cuts_volatility"] = self.cuts_
        d["regimes_fitted"] = sorted(self.models_.keys())
        return d


REGISTRY = {
    "zero": ZeroModel,
    "ridge": RidgeModel,
    "logistic": LogisticModel,
    "hist": HistGBModel,
    "law": HiddenLawModel,
    "regime": RegimeRidge,
}


def make_model(name: str, seed: int = 13, **kwargs) -> BaseModel:
    key = (name or "").lower()
    if key in REGISTRY:
        if key == "hist":
            return HistGBModel(seed=seed, **{k: v for k, v in kwargs.items()
                                             if k in HistGBModel.__init__.__code__.co_varnames})
        return REGISTRY[key](**{k: v for k, v in kwargs.items()
                                if k in REGISTRY[key].__init__.__code__.co_varnames})
    raise ValueError(f"unknown model '{name}'; available: {sorted(REGISTRY)}")


def save_model(model: BaseModel, path, meta: dict | None = None) -> None:
    blob = {"name": model.name, "model": model,
            "feature_names": list(getattr(model, "feature_names_", [])),
            "meta": meta or {}}
    joblib.dump(blob, path)


def load_model(path) -> tuple[BaseModel, dict]:
    blob = joblib.load(path)
    return blob["model"], blob.get("meta", {})
