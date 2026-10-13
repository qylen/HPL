"""Leakage-detection suite behaviour (including the planted-leak detector)."""
import numpy as np
import pandas as pd

from hpl1.features import build_features
from hpl1.training import chronological_split
from hpl1.validation import (
    check_feature_truncation,
    check_label_alignment,
    check_scaler_domain,
    check_split_disjoint,
    planted_leak_detection_demo,
    run_leakage_suite,
)


def _raw(seed=7, n=2000, with_book=True):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "second": np.arange(1_700_000_000, 1_700_000_000 + n),
        "n_trades": rng.integers(0, 5, n),
        "buy_vol": rng.random(n), "sell_vol": rng.random(n),
        "vol": rng.random(n) + 0.1, "notional": rng.random(n) * 100 + 10,
        "last_px": 100 + np.cumsum(rng.normal(0, 0.01, n)),
    })
    if with_book:
        mid = 100 + np.cumsum(rng.normal(0, 0.01, n))
        df["bid"] = mid - 0.05
        df["ask"] = mid + 0.05
        df["bid_qty"] = rng.random(n) + 1
        df["ask_qty"] = rng.random(n) + 1
        df["ofi"] = rng.normal(0, 1, n)
        df["n_updates"] = rng.integers(1, 10, n)
    return df


def test_label_alignment_passes_on_honest_features():
    feats = build_features(_raw(), feature_set="full")
    res = check_label_alignment(feats)
    assert res["passed"], res
    assert res["final_row_unlabeled"] is True


def test_label_alignment_fails_on_shifted_labels():
    feats = build_features(_raw(), feature_set="full")
    feats["target_return"] = feats["target_return"].shift(1)  # off-by-one corruption
    res = check_label_alignment(feats)
    assert not res["passed"]


def test_truncation_check_passes_on_honest_features():
    raw = _raw(n=2500)
    res = check_feature_truncation(lambda df: build_features(df, feature_set="full"),
                                   raw, ["ofi_norm", "trade_imbalance", "mid_return_1s"],
                                   tail_rows=300)
    assert res["passed"], res


def test_truncation_check_flags_future_leaking_feature():
    raw = _raw(n=2500)

    def leaky(df):
        out = pd.DataFrame(index=df.index)
        out["second"] = df["second"]
        out["leak"] = df["bid"].shift(-1)  # future bid as today's feature
        return out

    res = check_feature_truncation(leaky, raw, ["leak"], tail_rows=300)
    assert not res["passed"]


def test_planted_leak_detection_demo():
    res = planted_leak_detection_demo()
    assert res["passed"]
    assert res["honest_features_pass"] is True
    assert res["leaked_feature_flagged"] is True


def test_split_disjoint_detects_overlap():
    bad = {"train": slice(0, 6000), "validation": slice(5990, 7990), "test": slice(8000, 10000)}
    res = check_split_disjoint(bad, 10000, embargo=0)
    assert not res["passed"]
    assert any("overlap" in p for p in res["problems"])


def test_split_disjoint_detects_embargo_violation():
    bad = {"train": slice(0, 6000), "validation": slice(6001, 7990), "test": slice(7995, 10000)}
    res = check_split_disjoint(bad, 10000, embargo=30)
    assert not res["passed"]


def test_split_disjoint_passes_on_valid_splits():
    s = chronological_split(10000, 0.6, 0.2, purge=1, embargo=30)
    res = check_split_disjoint(s, 10000, embargo=30)
    assert res["passed"], res["problems"]


def test_scaler_domain_detects_wrong_fit():
    from hpl1.models import RidgeModel

    raw = _raw(n=1200)
    Xa = pd.DataFrame({"a": np.linspace(0, 1, 600), "b": np.linspace(10, 20, 600)})
    Xb = pd.DataFrame({"a": np.linspace(100, 200, 600), "b": np.linspace(-50, -40, 600)})
    y = np.random.default_rng(1).normal(0, 1e-4, 600)
    honest = RidgeModel().fit(Xa, y)
    assert check_scaler_domain(honest, Xa, Xb)["passed"] is True

    leaky = RidgeModel()
    # fit the scaler on the WRONG (future) split statistics
    leaky.feature_names_ = list(Xb.columns)
    leaky.pipe = honest.pipe
    # construct a model whose scaler means match Xb

    from sklearn.linear_model import Ridge as SkRidge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    leaky.pipe = Pipeline([("scaler", StandardScaler()), ("ridge", SkRidge(alpha=1.0))])
    leaky.pipe.fit(Xb, y)  # fitted on the non-train split
    res = check_scaler_domain(leaky, Xa, Xb)
    assert res["passed"] is False


def test_run_leakage_suite_end_to_end():
    raw = _raw(n=2500)
    feats = build_features(raw, feature_set="full")
    s = chronological_split(len(feats), 0.6, 0.2, purge=1, embargo=30)
    suite = run_leakage_suite(feats, splits=s,
                              build_fn=lambda df: build_features(df, feature_set="full"),
                              raw_df=raw,
                              feature_columns=["ofi_norm", "trade_imbalance"])
    assert suite["passed"]
    names = [c["name"] for c in suite["checks"]]
    assert "label_alignment" in names and "planted_leak_detection" in names
