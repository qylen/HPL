"""Models, chronological splits, selection discipline, save/load."""
import numpy as np
import pandas as pd
import pytest

from hpl1.features import FEATURE_COLUMNS_FULL
from hpl1.models import (
    HiddenLawModel,
    HistGBModel,
    LogisticModel,
    RegimeRidge,
    RidgeModel,
    ZeroModel,
    load_model,
    make_model,
    save_model,
)
from hpl1.training import chronological_split, compare_models, metrics


def _planted_frame(n=3000, seed=11, strength=0.0004):
    """Frame whose NEXT-second return depends on current ofi_norm (predictive)."""
    rng = np.random.default_rng(seed)
    ofi = rng.normal(0, 0.3, n)
    noise = rng.normal(0, 0.0002, n)
    y = np.empty(n)
    y[0] = noise[0]
    y[1:] = strength * ofi[:-1] + noise[1:]
    price = 100.0 * np.exp(np.cumsum(y))
    df = pd.DataFrame({"second": np.arange(1_700_000_000, 1_700_000_000 + n)})
    for c in FEATURE_COLUMNS_FULL:
        df[c] = rng.normal(0, 1, n)
    df["ofi_norm"] = ofi
    df["trade_imbalance"] = np.tanh(ofi + rng.normal(0, 0.5, n))
    df["spread_bps"] = 1.0
    df["depth_total"] = np.abs(rng.normal(5, 1, n))
    df["volatility_60s"] = np.abs(rng.normal(0.0001, 0.00002, n))
    df["ret_lag1"] = np.concatenate([[0.0], y[:-1]])
    df["price"] = price
    df["mid"] = price
    df["bid"] = price - 0.05
    df["ask"] = price + 0.05
    df["ts"] = pd.date_range("2025-01-01", periods=n, freq="s", tz="UTC")
    df["target_return"] = np.concatenate([y[1:], [np.nan]])
    df["valid"] = True
    return df


def test_zero_model_predicts_zero():
    m = ZeroModel().fit(pd.DataFrame({"x": [1.0, 2.0]}), np.array([0.1, -0.1]))
    assert np.allclose(m.predict(pd.DataFrame({"x": [3.0]})), 0.0)


def test_ridge_recovers_planted_linear_relationship():
    df = _planted_frame(4000, strength=0.001)
    X = df[FEATURE_COLUMNS_FULL]
    y = df["target_return"].to_numpy()
    m = RidgeModel(alpha=1.0).fit(X[:-2], y[:-2])
    pred = m.predict(X[-2:])
    assert pred[0] != 0.0


def test_logistic_directional_baseline_beats_coin_flip():
    df = _planted_frame(4000, strength=0.002)
    X = df[["ofi_norm"]]
    y = df["target_return"].to_numpy()
    m = LogisticModel().fit(X.iloc[:3000], y[:3000])
    pred = m.predict(X.iloc[3000:])
    acc = float(np.mean(np.sign(pred) == np.sign(y[3000:])))
    assert acc > 0.55


def test_hist_model_deterministic():
    df = _planted_frame(1500)
    X, y = df[FEATURE_COLUMNS_FULL], df["target_return"].to_numpy()
    m1 = HistGBModel(seed=13).fit(X[:1200], y[:1200])
    m2 = HistGBModel(seed=13).fit(X[:1200], y[:1200])
    assert np.allclose(m1.predict(X[1300:1310]), m2.predict(X[1300:1310]))


def test_hidden_law_model_terms_present_and_fitted():
    df = _planted_frame(3000, strength=0.001)
    X, y = df[FEATURE_COLUMNS_FULL], df["target_return"].to_numpy()
    m = HiddenLawModel().fit(X[:2500], y[:2500])
    d = m.describe()
    assert set(d["law_terms"]) >= {"I", "I_L", "I_sigma"}
    assert all(np.isfinite(v) for v in d["law_terms"].values())


def test_regime_model_single_row_predict_does_not_crash():
    """Regression test for the single-row broadcasting landmine."""
    df = _planted_frame(1200)
    X, y = df[FEATURE_COLUMNS_FULL], df["target_return"].to_numpy()
    m = RegimeRidge().fit(X[:1000], y[:1000])
    single = m.predict(X.iloc[[1100]])  # exactly one row
    assert np.isfinite(single[0])
    many = m.predict(X.iloc[1050:1060])
    assert len(many) == 10 and np.isfinite(many).all()


def test_regime_quantiles_fit_on_train_only():
    df = _planted_frame(1200)
    X, y = df[FEATURE_COLUMNS_FULL], df["target_return"].to_numpy()
    train_sig = X["volatility_60s"].iloc[:1000]
    m = RegimeRidge().fit(X[:1000], y[:1000])
    q1, q2 = m.cuts_
    assert q1 == pytest.approx(np.quantile(train_sig, 1 / 3), rel=1e-6)
    assert q2 == pytest.approx(np.quantile(train_sig, 2 / 3), rel=1e-6)


def test_save_load_roundtrip(tmp_path):
    df = _planted_frame(1500)
    X, y = df[FEATURE_COLUMNS_FULL], df["target_return"].to_numpy()
    m = RidgeModel(alpha=3.0).fit(X[:1200], y[:1200])
    p = tmp_path / "model.joblib"
    save_model(m, p, meta={"note": "test"})
    loaded, meta = load_model(p)
    assert np.allclose(loaded.predict(X[1300:1310]), m.predict(X[1300:1310]))
    assert meta["note"] == "test"


def test_make_model_registry():
    for name in ("zero", "ridge", "logistic", "hist", "law", "regime"):
        m = make_model(name, seed=1)
        assert m.name == name
    with pytest.raises(ValueError):
        make_model("deep_learning_timesfm_9000")


def test_chronological_split_disjoint_purge_embargo():
    s = chronological_split(10_000, 0.6, 0.2, purge=1, embargo=30)
    assert s["train"].stop <= s["validation"].start
    assert s["validation"].start - s["train"].stop >= 30  # embargo honored
    assert s["validation"].stop <= s["test"].start
    assert s["test"].start - s["validation"].stop >= 30
    covered = s["train"].stop + (s["validation"].stop - s["validation"].start) \
        + (s["test"].stop - s["test"].start)
    assert covered <= 10_000 - 60  # purge + embargo rows removed at boundaries


def test_chronological_split_rejects_tiny_data():
    with pytest.raises(ValueError, match="not enough rows"):
        chronological_split(50, 0.6, 0.2)


def test_metrics_shapes():
    m = metrics(np.array([0.001, -0.002, 0.0005]), np.array([0.0009, -0.0019, 0.0004]))
    assert -1 <= m["oos_r2_vs_zero"] <= 1
    assert 0 <= m["directional_accuracy_nonzero"] <= 1
    assert m["rows"] == 3


def test_compare_models_selects_on_validation_not_test():
    df = _planted_frame(4000, strength=0.0015)
    out = compare_models(df, FEATURE_COLUMNS_FULL, ["zero", "ridge", "hist"],
                         seed=13, artifact_dir=None)
    assert out["best"] != "zero"  # planted signal must beat the zero benchmark
    val_r2 = {name: r["validation"]["oos_r2_vs_zero"] for name, r in out["results"].items()}
    best_val = max(val_r2, key=val_r2.get)
    assert out["best"] == best_val  # selection = argmax validation, by construction
    # only the SELECTED model has test metrics (others never touch test)
    with_test = [n for n, r in out["results"].items() if "test" in r]
    assert with_test == [out["best"]]


def test_compare_models_test_predictions_carry_backtest_columns():
    df = _planted_frame(2500)
    out = compare_models(df, FEATURE_COLUMNS_FULL, ["ridge"], seed=13)
    pred = out["predictions"]
    for col in ("ts", "second", "prediction", "actual_return", "bid", "ask", "mid"):
        assert col in pred.columns
