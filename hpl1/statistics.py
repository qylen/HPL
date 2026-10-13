"""Statistical validation: HAC inference, block bootstrap, placebo tests,
Ljung-Box, Holm multiple-testing correction, calibration."""
from __future__ import annotations

import math

import numpy as np


def hac_mean_test(values: np.ndarray, max_lags: int = 60) -> dict:
    """Newey-West/HAC z-test for the mean of a dependent series."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return {"hac_lags": 0, "hac_z_score": 0.0, "hac_p_value_two_sided": 1.0}
    mean = float(x.mean())
    centered = x - mean
    lag_count = min(max(0, int(max_lags)), n - 1)
    lrv = float(np.dot(centered, centered) / n)
    for lag in range(1, lag_count + 1):
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        weight = 1.0 - lag / (lag_count + 1.0)
        lrv += 2.0 * weight * gamma
    if not np.isfinite(lrv) or lrv <= 0:
        return {"hac_lags": lag_count, "hac_z_score": 0.0, "hac_p_value_two_sided": 1.0}
    se = math.sqrt(lrv / n)
    z = mean / se if se > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return {"hac_lags": lag_count, "hac_z_score": float(z), "hac_p_value_two_sided": float(p)}


def ljung_box(values: np.ndarray, lags: tuple[int, ...] = (1, 2, 5, 10, 30)) -> dict:
    """Ljung-Box portmanteau test on residuals; returns per-lag autocorrelations
    and a joint Q statistic with chi-square p-value (normal approximation)."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    out: dict = {}
    if n < 20:
        return {"error": "series too short"}
    denom = float(np.dot(x, x))
    if denom <= 0:
        return {"error": "zero variance"}
    Q = 0.0
    L = 0
    for lag in lags:
        if lag >= n:
            continue
        rho = float(np.dot(x[lag:], x[:-lag]) / denom)
        Q += rho * rho / (n - lag)
        L += 1
        out[f"autocorr_lag_{lag}"] = rho
    if L == 0:
        return {"error": "no valid lags"}
    Q *= n * (n + 2)
    # chi-square survival function via regularized upper incomplete gamma (k/2, x/2)
    p = _chi2_sf(Q, L)
    out["ljung_box_q"] = float(Q)
    out["df"] = L
    out["p_value"] = float(p)
    return out


def _chi2_sf(x: float, k: int) -> float:
    """Survival function of chi-square with k dof via series (Wilson-Hilferty backup)."""
    if x <= 0:
        return 1.0
    try:
        from scipy.special import gammaincc

        return float(gammaincc(k / 2.0, x / 2.0))
    except Exception:  # noqa: BLE001
        z = ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
        return math.erfc(z / math.sqrt(2.0))


def moving_block_bootstrap(stat_fn, values: np.ndarray, block: int = 60, b: int = 500,
                           seed: int = 13, alpha: float = 0.05) -> dict:
    """Non-overlapping moving-block bootstrap CI for a statistic of a dependent series."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2 * block or b < 10:
        return {"error": "series too short or too few replicates"}
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    stats = np.empty(b)
    for i in range(b):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        sample = np.concatenate([x[s:s + block] for s in starts])[:n]
        stats[i] = float(stat_fn(sample))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"point_estimate": float(stat_fn(x)), "ci_low": float(lo), "ci_high": float(hi),
            "block": block, "replicates": b, "level": 1 - alpha}


def time_shift_placebo(predictions: np.ndarray, actual: np.ndarray,
                       max_shift: int = 10, seed: int = 13) -> dict:
    """Placebo: correlation of predictions vs time-shifted actual returns.

    If the apparent signal is real (alignment-dependent), shifted correlations
    collapse to ~0; the observed correlation should exceed the shifted 95% band.
    """
    p = np.asarray(predictions, dtype=float)
    a = np.asarray(actual, dtype=float)
    ok = np.isfinite(p) & np.isfinite(a)
    p, a = p[ok], a[ok]
    n = len(p)
    if n < 50:
        return {"error": "series too short"}
    real_corr = float(np.corrcoef(p, a)[0, 1]) if np.std(p) > 0 and np.std(a) > 0 else 0.0
    shifted = []
    for k in range(1, max_shift + 1):
        ps, as_ = p[:-k], a[k:]
        if len(ps) > 10 and np.std(ps) > 0 and np.std(as_) > 0:
            shifted.append(float(np.corrcoef(ps, as_)[0, 1]))
        ps2, as2 = p[k:], a[:-k]
        if len(ps2) > 10 and np.std(ps2) > 0 and np.std(as2) > 0:
            shifted.append(float(np.corrcoef(ps2, as2)[0, 1]))
    if not shifted:
        return {"error": "no valid shifts"}
    shifted = np.asarray(shifted)
    p95 = float(np.quantile(np.abs(shifted), 0.95))
    return {"real_correlation": real_corr,
            "shifted_abs_correlation_mean": float(np.mean(np.abs(shifted))),
            "shifted_abs_correlation_p95": p95,
            "alignment_dependent": bool(abs(real_corr) > p95),
            "n_shifts": len(shifted)}


def holm_correction(pvalues: dict[str, float], alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down adjustment for multiple comparisons."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, dict] = {}
    running_max = 0.0
    for i, (name, p) in enumerate(items):
        adj = (m - i) * p
        running_max = max(running_max, adj)
        adjusted[name] = {"raw_p": float(p), "holm_p": min(1.0, running_max),
                          "significant_at_alpha": bool(min(1.0, running_max) < alpha)}
    return {"alpha": alpha, "n_tests": m, "adjusted": adjusted}


def directional_calibration(predictions: np.ndarray, actual: np.ndarray,
                             bins: int = 5,
                             calibration_fit: tuple | None = None) -> dict:
    """Brier score and reliability bins for the directional forecast.

    Probabilities come from a 1-parameter logistic fit of sign(actual) on the
    standardized prediction. When `calibration_fit` = (validation_predictions,
    validation_actuals) is supplied, the mapping is fit on VALIDATION ONLY and
    applied to the test set (leakage-clean); otherwise an in-sample squash is
    used and flagged as a diagnostic.
    """
    p = np.asarray(predictions, dtype=float)
    a = np.asarray(actual, dtype=float)
    ok = np.isfinite(p) & np.isfinite(a)
    p, a = p[ok], a[ok]
    if len(p) < 50:
        return {"error": "series too short"}
    up = (a > 0).astype(float)
    if calibration_fit is not None:
        vp, va = (np.asarray(v, dtype=float) for v in calibration_fit)
        okv = np.isfinite(vp) & np.isfinite(va)
        vp, va = vp[okv], va[okv]
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=1.0, max_iter=200)
        scale = float(np.std(vp)) + 1e-12
        lr.fit((vp / scale).reshape(-1, 1), (va > 0).astype(int))
        prob = lr.predict_proba((p / scale).reshape(-1, 1))[:, 1]
        mapping = "validation-fit logistic (leakage-clean)"
    else:
        prob = 1.0 / (1.0 + np.exp(-np.clip(p / (np.std(p) + 1e-12) * 2.0, -30, 30)))
        mapping = "in-sample squash (diagnostic only; not leakage-clean)"
    brier = float(np.mean((prob - up) ** 2))
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i in range(bins):
        m = (prob >= edges[i]) & (prob < edges[i + 1] if i < bins - 1 else prob <= edges[i + 1])
        if m.sum() >= 5:
            rows.append({"bin": f"[{edges[i]:.1f},{edges[i + 1]:.1f})",
                         "n": int(m.sum()), "predicted_up_rate": float(prob[m].mean()),
                         "observed_up_rate": float(up[m].mean())})
    return {"brier_score": brier, "brier_baseline_up_rate": float(up.mean()),
            "probability_mapping": mapping, "reliability": rows}
