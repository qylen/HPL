"""Feature builder (archive path), labels, staleness, epsilon guards."""
import numpy as np
import pandas as pd
import pytest

from hpl1.features import (
    FEATURE_COLUMNS_TRADE,
    build_features,
    feature_columns_for,
    ofi_between,
)


def _base_frame(seconds: int = 600, seed: int = 5, with_book: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sec = np.arange(1_700_000_000, 1_700_000_000 + seconds)
    vol = np.abs(rng.normal(0.05, 0.02, seconds))
    buy = vol * (0.5 + 0.3 * rng.normal(0, 1, seconds))
    buy = np.clip(buy, 0, None)
    sell = np.clip(vol - buy, 0, None)
    px = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.0001, seconds)))
    df = pd.DataFrame({
        "second": sec,
        "n_trades": rng.integers(0, 8, seconds),
        "buy_vol": buy, "sell_vol": sell, "vol": buy + sell,
        "notional": px * (buy + sell), "last_px": px,
    })
    if with_book:
        mid = px
        df["bid"] = mid - 0.05
        df["ask"] = mid + 0.05
        df["bid_qty"] = 1 + np.abs(rng.normal(0, 0.3, seconds))
        df["ask_qty"] = 1 + np.abs(rng.normal(0, 0.3, seconds))
        df["ofi"] = rng.normal(0, 0.5, seconds)
        df["n_updates"] = rng.integers(1, 10, seconds)
    return df


def test_label_alignment_exact():
    feats = build_features(_base_frame(300), feature_set="full")
    price = feats["price"].to_numpy()
    target = feats["target_return"].to_numpy()
    expected = np.log(price[1:] / price[:-1])
    assert np.allclose(target[:-1], expected, atol=1e-12)
    assert np.isnan(target[-1])  # final row unlabeled


def test_features_use_only_past_information():
    """Truncation invariance: dropping the tail must not change earlier rows."""
    raw = _base_frame(400)
    f_full = build_features(raw, feature_set="full")
    f_trunc = build_features(raw.iloc[:-100], feature_set="full")
    n = len(f_trunc) - 120
    cols = [c for c in feature_columns_for("full") if c in f_full.columns]
    a = f_full[cols].iloc[:n].to_numpy(dtype=float)
    b = f_trunc[cols].iloc[:n].to_numpy(dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    assert np.allclose(a[mask], b[mask], atol=1e-9)


def test_stale_quotes_marked_invalid_not_silently_ffilled():
    raw = _base_frame(300)
    # kill all book events for seconds 100..139 (40s outage)
    raw.loc[100:139, ["bid", "ask", "bid_qty", "ask_qty", "ofi", "n_updates"]] = np.nan
    feats = build_features(raw, feature_set="full")
    # gap flag set during outage, and rows beyond stale_max are invalid
    assert feats.loc[110:138, "data_gap"].all()
    assert (~feats.loc[115:138, "valid"]).all()
    # price was forward-filled but explicitly marked
    assert feats.loc[115:138, "price"].notna().all()
    # recovery: seconds with fresh events become valid again
    assert feats.loc[145, "valid"]


def test_zero_timestamp_is_valid_not_stale():
    """0 is a valid epoch second; falsy-zero checks must not mark it stale."""
    raw = _base_frame(5)
    raw["second"] = [0, 1, 2, 3, 4]
    feats = build_features(raw, feature_set="full")
    assert feats.loc[0, "valid"]
    assert feats.loc[0, "quote_age_seconds"] == 0


def test_zero_denominators_guarded():
    raw = _base_frame(50)
    raw.loc[:, "buy_vol"] = 0.0
    raw.loc[:, "sell_vol"] = 0.0
    raw.loc[:, "vol"] = 0.0
    raw.loc[:, "bid_qty"] = 0.0
    raw.loc[:, "ask_qty"] = 0.0
    feats = build_features(raw, feature_set="full")
    assert np.isfinite(feats["trade_imbalance"].to_numpy()).all()
    assert np.isfinite(feats["depth_imbalance"].to_numpy()).all()
    assert np.isfinite(feats["volume_imb_60s"].to_numpy()).all()


def test_trade_only_feature_set_for_spot():
    raw = _base_frame(300, with_book=False)
    feats = build_features(raw, feature_set="trade_only")
    assert set(FEATURE_COLUMNS_TRADE) - set(feats.columns) == set()
    assert np.isnan(feats["spread_bps"]).all()  # honestly unknown, not fabricated
    assert feats["valid"].iloc[100:].mean() > 0.9


def test_full_set_requires_book_columns():
    with pytest.raises(ValueError, match="no published book history"):
        build_features(_base_frame(10, with_book=False), feature_set="full")


def test_ofi_between_ask_drop_positive():
    # ask 100.1 -> 100.0 (offer lifted): +old ask qty
    assert ofi_between((100.0, 1.0, 100.1, 2.0), (100.0, 1.0, 100.0, 5.0)) == pytest.approx(2.0)
    # ask 100.0 -> 100.2 (new offers): -new ask qty
    assert ofi_between((100.0, 1.0, 100.0, 5.0), (100.0, 1.0, 100.2, 3.0)) == pytest.approx(-3.0)
    # bid 100.0 -> 100.1 (bid lifted): +new bid qty
    assert ofi_between((100.0, 4.0, 100.2, 1.0), (100.1, 6.0, 100.2, 1.0)) == pytest.approx(6.0)
    # bid 100.1 -> 100.0 (bid pulled): -old bid qty
    assert ofi_between((100.1, 6.0, 100.2, 1.0), (100.0, 2.0, 100.2, 1.0)) == pytest.approx(-6.0)


def test_missing_seconds_reindexed_onto_grid():
    raw = _base_frame(100)
    raw_dropped = raw.drop(index=list(range(50, 65)))  # 15-second hole
    feats = build_features(raw_dropped.reset_index(drop=True), feature_set="full")
    assert len(feats) == 100  # full grid restored
    assert feats.loc[62, "data_gap"]  # hole exceeds gap threshold -> detected (age 13s)
    assert (~feats.loc[62, "valid"])
    assert feats.loc[49, "valid"]      # boundary rows unaffected


def test_empty_input_frame():
    empty = pd.DataFrame(columns=["second", "n_trades", "buy_vol", "sell_vol", "vol",
                                  "notional", "last_px"])
    feats = build_features(empty, feature_set="trade_only")
    assert feats.empty


def test_crossed_book_after_ffill_invalid():
    raw = _base_frame(20)
    raw.loc[10, "ask"] = raw.loc[10, "bid"] - 1.0  # crossed quote event
    feats = build_features(raw, feature_set="full")
    assert feats.loc[10, "data_gap"]


def test_funding_column_preserved():
    raw = _base_frame(100)
    raw["funding_rate"] = np.nan
    raw.loc[50, "funding_rate"] = 0.0001
    feats = build_features(raw, feature_set="full")
    assert feats.loc[50, "funding_rate"] == pytest.approx(0.0001)
    assert np.isnan(feats.loc[49, "funding_rate"])
