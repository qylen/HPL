"""Statistics: HAC, Ljung-Box, block bootstrap, placebo, Holm, calibration."""
import numpy as np
import pytest

from hpl1.statistics import (
    directional_calibration,
    hac_mean_test,
    holm_correction,
    ljung_box,
    moving_block_bootstrap,
    time_shift_placebo,
)


def test_hac_rejects_false_mean_on_white_noise():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 0.001, 20_000)
    res = hac_mean_test(x, max_lags=60)
    assert abs(res["hac_z_score"]) < 3


def test_hac_detects_real_mean():
    rng = np.random.default_rng(6)
    x = rng.normal(0.0002, 0.001, 20_000)
    res = hac_mean_test(x, max_lags=60)
    assert res["hac_z_score"] > 10


def test_ljung_box_white_noise_not_rejected():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 5_000)
    res = ljung_box(x)
    assert res["p_value"] > 0.01  # white noise passes


def test_ljung_box_detects_autocorrelation():
    rng = np.random.default_rng(8)
    n = 5_000
    x = np.empty(n)
    x[0] = 0.0
    for i in range(1, n):
        x[i] = 0.8 * x[i - 1] + rng.normal(0, 1)
    res = ljung_box(x)
    assert res["p_value"] < 0.01
    assert res["autocorr_lag_1"] > 0.6


def test_block_bootstrap_ci_contains_point():
    rng = np.random.default_rng(9)
    x = rng.normal(0.5, 1.0, 3_000)
    res = moving_block_bootstrap(np.mean, x, block=60, b=200, seed=13)
    assert res["ci_low"] < res["point_estimate"] < res["ci_high"]


def test_block_bootstrap_wider_than_iid_for_dependent_data():
    rng = np.random.default_rng(10)
    n = 4_000
    x = np.empty(n)
    x[0] = 0
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + rng.normal(0, 1)
    dep = moving_block_bootstrap(np.mean, x, block=60, b=200, seed=13)
    indep = moving_block_bootstrap(np.mean, rng.normal(0, np.std(x), n), block=60, b=200, seed=13)
    width_dep = dep["ci_high"] - dep["ci_low"]
    width_indep = indep["ci_high"] - indep["ci_low"]
    assert width_dep > width_indep  # ignoring dependence would understate uncertainty


def test_placebo_time_shift_flags_real_alignment():
    rng = np.random.default_rng(11)
    n = 5_000
    signal = rng.normal(0, 1, n)
    actual = 0.3 * signal + rng.normal(0, 1, n)  # genuinely correlated
    res = time_shift_placebo(signal, actual, max_shift=10)
    assert res["alignment_dependent"] is True
    assert res["real_correlation"] > 0.25
    assert res["shifted_abs_correlation_p95"] < 0.1


def test_placebo_time_shift_rejects_noise():
    rng = np.random.default_rng(12)
    p = rng.normal(0, 1, 5_000)
    a = rng.normal(0, 1, 5_000)
    res = time_shift_placebo(p, a, max_shift=10)
    assert res["alignment_dependent"] is False


def test_holm_correction_increases_pvalues():
    raw = {"test_a": 0.01, "test_b": 0.04, "test_c": 0.03}
    res = holm_correction(raw, alpha=0.05)
    assert res["n_tests"] == 3
    adj = res["adjusted"]
    assert adj["test_a"]["holm_p"] == pytest.approx(0.03)
    # strongest test stays significant; weaker ones are pushed above alpha
    assert adj["test_a"]["holm_p"] <= adj["test_c"]["holm_p"]


def test_directional_calibration_brier_beats_constant():
    rng = np.random.default_rng(13)
    n = 2_000
    true_up = rng.random(n) < 0.7
    # a forecast that knows the truth 70% of the time
    good = np.where(true_up, 1.0, -1.0) * np.where(rng.random(n) < 0.7, 1.0, -1.0)
    noise = rng.normal(0, 1, n)
    b_good = directional_calibration(good, np.where(true_up, 0.001, -0.001))["brier_score"]
    b_noise = directional_calibration(noise, np.where(true_up, 0.001, -0.001))["brier_score"]
    assert b_good < b_noise
