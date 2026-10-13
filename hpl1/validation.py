"""Leakage-detection suite: automated checks for the classic bias families.

Checks:
1. label_alignment   — target[t] == log(price[t+1]/price[t]) exactly
2. feature_truncation— features built on truncated data equal the prefix of
                       features built on full data (any future-dependence breaks it)
3. split_disjoint    — chronological splits do not overlap; purge+embargo honored
4. scaler_domain     — fitted scalers/quantiles come from train statistics only
5. planted-leak detector — verifies check 2 actually FLAGS a leaked feature
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


def check_label_alignment(feature_frame: pd.DataFrame, price_col: str = "price",
                           tol: float = 1e-10) -> dict:
    df = feature_frame
    if price_col not in df.columns or "target_return" not in df.columns:
        return {"name": "label_alignment", "passed": False, "error": "missing price/target columns"}
    price = df[price_col].to_numpy(dtype=float)
    target = df["target_return"].to_numpy(dtype=float)
    n = len(df)
    if n < 3:
        return {"name": "label_alignment", "passed": False, "error": "frame too short"}
    expected = np.full(n, np.nan)
    ok = np.isfinite(price[:-1]) & (price[:-1] > 0)
    expected[:-1][ok] = np.log(price[1:][ok] / price[:-1][ok])
    both = np.isfinite(expected) & np.isfinite(target)
    if both.sum() < 3:
        return {"name": "label_alignment", "passed": False, "error": "not enough finite labels"}
    max_err = float(np.max(np.abs(expected[both] - target[both])))
    tail_nan = bool(np.isnan(target[-1]))
    return {"name": "label_alignment", "passed": bool(max_err <= tol and tail_nan),
            "max_abs_error": max_err, "tolerance": tol, "final_row_unlabeled": tail_nan}


def check_feature_truncation(build_features_fn: Callable[[pd.DataFrame], pd.DataFrame],
                             raw_df: pd.DataFrame, feature_columns: list[str],
                             tail_rows: int = 500, tol: float = 1e-9) -> dict:
    """Features must not depend on data after the row they describe.

    Builds features twice: once on `raw_df`, once on `raw_df` with the last
    `tail_rows` raw rows removed. The first (len-tail) feature rows must match.
    """
    raw_trunc = raw_df.iloc[:-tail_rows] if tail_rows < len(raw_df) else raw_df.iloc[:1]
    f_full = build_features_fn(raw_df)
    f_trunc = build_features_fn(raw_trunc)
    compare_n = min(len(f_trunc), max(0, len(f_full) - tail_rows))
    if compare_n < 5:
        return {"name": "feature_truncation", "passed": False, "error": "frames too short to compare"}
    cols = [c for c in feature_columns if c in f_full.columns and c in f_trunc.columns]
    if not cols:
        return {"name": "feature_truncation", "passed": False, "error": "no common feature columns"}
    a = f_full[cols].iloc[:compare_n].to_numpy(dtype=float)
    b = f_trunc[cols].iloc[:compare_n].to_numpy(dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return {"name": "feature_truncation", "passed": False, "error": "no finite values to compare"}
    max_err = float(np.max(np.abs(a[mask] - b[mask])))
    nan_mismatch = int((np.isfinite(a) != np.isfinite(b)).sum())
    return {"name": "feature_truncation", "passed": bool(max_err <= tol and nan_mismatch == 0),
            "max_abs_error": max_err, "nan_pattern_mismatches": nan_mismatch,
            "rows_compared": int(compare_n), "columns_compared": len(cols)}


def check_split_disjoint(splits: dict[str, slice], n: int, embargo: int = 0) -> dict:
    ranges = {k: (v.start, v.stop) for k, v in splits.items()}
    problems = []
    for name, (s, e) in ranges.items():
        if s < 0 or e > n or e <= s:
            problems.append(f"{name} invalid range [{s},{e}) for n={n}")
    order = ["train", "validation", "test"]
    for a, b in zip(order, order[1:]):
        if a in ranges and b in ranges:
            gap = ranges[b][0] - ranges[a][1]
            if gap < 0:
                problems.append(f"{b} starts before {a} ends (overlap)")
            if gap < embargo:
                problems.append(f"embargo violated between {a} and {b}: gap={gap} < {embargo}")
    return {"name": "split_disjoint", "passed": not problems, "ranges": ranges, "problems": problems}


def check_scaler_domain(model, X_train: pd.DataFrame, X_other: pd.DataFrame,
                        tol: float = 1e-6) -> dict:
    """Fitted StandardScaler means must equal TRAIN means, not pooled/val means."""
    pipe = getattr(model, "pipe", None) or getattr(model, "models_", None)
    scalers = []
    if isinstance(pipe, dict):
        for sub in pipe.values():
            scalers.append(sub.named_steps.get("scaler"))
    elif pipe is not None and hasattr(pipe, "named_steps"):
        scalers.append(pipe.named_steps.get("scaler"))
    if not scalers:
        return {"name": "scaler_domain", "passed": True, "note": "model has no scaler (tree/constant)"}
    train_means = X_train.mean().to_numpy(dtype=float)
    other_means = X_other.mean().to_numpy(dtype=float)
    for sc in scalers:
        if sc is None:
            continue
        sm = np.asarray(sc.mean_)
        if len(sm) == len(train_means):
            if np.max(np.abs(sm - train_means)) > tol * max(1.0, float(np.nanmax(np.abs(train_means)))):
                # fitted means match the OTHER split -> leakage
                if np.max(np.abs(sm - other_means)) < np.max(np.abs(sm - train_means)):
                    return {"name": "scaler_domain", "passed": False,
                            "error": "scaler mean matches non-train split statistics"}
    return {"name": "scaler_domain", "passed": True}


def planted_leak_detection_demo() -> dict:
    """Verify the truncation check catches a deliberately future-leaking feature."""
    rng = np.random.default_rng(7)
    n = 3000
    raw = pd.DataFrame({
        "second": np.arange(n),
        "ts": pd.date_range("2025-01-01", periods=n, freq="s", tz="UTC"),
        "n_updates": np.ones(n),
        "n_trades": rng.integers(0, 5, n),
        "buy_vol": rng.random(n), "sell_vol": rng.random(n),
        "vol": rng.random(n) + 0.1, "notional": rng.random(n) * 100 + 10,
        "bid": 100 + np.cumsum(rng.normal(0, 0.01, n)) - 0.05,
        "ask": 100 + np.cumsum(rng.normal(0, 0.01, n)) + 0.05,
        "bid_qty": rng.random(n) + 1, "ask_qty": rng.random(n) + 1,
        "ofi": rng.normal(0, 1, n),
    })
    raw["ask"] = raw["bid"] + 0.1

    def leaky_builder(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["second"] = df["second"]
        # LEAK: feature uses the NEXT second's bid (future information)
        out["leaked_feature"] = df["bid"].shift(-1).ffill()
        return out

    from .features import build_features
    honest = check_feature_truncation(build_features, raw,
                                      ["ofi_norm", "trade_imbalance", "mid_return_1s"])
    leaky = check_feature_truncation(leaky_builder, raw, ["leaked_feature"])
    return {"name": "planted_leak_detection", "passed": bool(honest["passed"] and not leaky["passed"]),
            "honest_features_pass": honest["passed"], "leaked_feature_flagged": not leaky["passed"],
            "note": "truncation check must pass honest features and fail future-leaking ones"}


def run_leakage_suite(feature_frame: pd.DataFrame, splits: dict[str, slice] | None = None,
                      build_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
                      raw_df: pd.DataFrame | None = None,
                      feature_columns: list[str] | None = None) -> dict[str, Any]:
    checks = [check_label_alignment(feature_frame)]
    if build_fn is not None and raw_df is not None and feature_columns:
        checks.append(check_feature_truncation(build_fn, raw_df, feature_columns))
    if splits is not None:
        checks.append(check_split_disjoint(splits, len(feature_frame)))
    checks.append(planted_leak_detection_demo())
    passed = all(c.get("passed", False) for c in checks)
    return {"passed": passed, "checks": checks}
