"""Quality reports and the live paper engine (unit level, no network)."""
import json

import numpy as np
import pandas as pd
import pytest

from hpl1.config import CostConfig, RiskConfig
from hpl1.features import build_features
from hpl1.live import PaperTradingEngine
from hpl1.markets import get_market
from hpl1.models import RidgeModel
from hpl1.quality import quality_markdown, quality_report


def _raw(seed=7, n=3000):
    rng = np.random.default_rng(seed)
    mid = 100 + np.cumsum(rng.normal(0, 0.01, n))
    return pd.DataFrame({
        "second": np.arange(1_700_000_000, 1_700_000_000 + n),
        "n_trades": rng.integers(0, 5, n),
        "buy_vol": rng.random(n), "sell_vol": rng.random(n),
        "vol": rng.random(n) + 0.1, "notional": rng.random(n) * 100 + 10,
        "last_px": mid,
        "bid": mid - 0.05, "ask": mid + 0.05,
        "bid_qty": rng.random(n) + 1, "ask_qty": rng.random(n) + 1,
        "ofi": rng.normal(0, 1, n), "n_updates": rng.integers(1, 10, n),
    })


def test_quality_report_counts():
    raw = _raw(n=2000)
    raw.loc[1000:1050, ["bid", "ask", "ofi", "n_updates"]] = np.nan  # 51s outage
    feats = build_features(raw, feature_set="full")
    rep = quality_report(feats, ingest_stats={"dup_events": 3, "out_of_order": 1,
                                              "crossed_quotes": 2, "applied_events": 9000})
    assert rep["rows_total"] == 2000
    assert rep["data_gap_seconds"] >= 40
    assert rep["usable_rows"] < 2000
    assert rep["integrity"]["duplicate_events"] == 3
    assert rep["gap_runs_ge_5s"] >= 1
    assert "exchange timestamps" in rep["timestamp_semantics"]


def test_quality_markdown_renders():
    raw = _raw(n=500)
    feats = build_features(raw, feature_set="full")
    md = quality_markdown(quality_report(feats), title="T")
    assert "# T" in md and "coverage_pct" in md


def test_quality_report_empty():
    rep = quality_report(pd.DataFrame())
    assert rep["rows"] == 0


# ---------------------------------------------------------------- paper engine

def _engine(tmp_path, seed=13, rolling=200, rules=None):
    profile = get_market("futures")
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x": rng.normal(0, 1, 600)})
    model = RidgeModel(alpha=1.0).fit(X, rng.normal(0, 1e-4, 600))
    model.feature_names_ = None  # engine falls back to available columns
    from hpl1.exchange_meta import SymbolRules

    rules = rules or SymbolRules("futures", "BTCUSDT", "0.10", "0.001", 100.0, "test")
    return PaperTradingEngine(profile, model, CostConfig(taker_fee_bps=5.0),
                              RiskConfig(starting_capital=10_000.0, cooldown_seconds=0,
                                         min_edge_bps=0.0),
                              run_dir=tmp_path / "run", rules=rules,
                              rolling_window=rolling)


def _feed_quote(e, uid, sec_ms, bid=100.0, ask=100.2, bidq=1.0, askq=1.0):
    e._on_quote({"u": uid, "b": str(bid), "B": str(bidq), "a": str(ask), "A": str(askq),
                 "T": sec_ms}, sec_ms)


def _feed_trade(e, tid, sec_ms, px=100.1, qty=0.5, maker=False):
    e._on_trade({"a": tid, "p": str(px), "q": str(qty), "m": maker, "T": sec_ms}, sec_ms)


def _now_ms() -> int:
    import time

    return int(time.time()) * 1000


def test_bucket_rollover_accumulates_per_second(tmp_path):
    e = _engine(tmp_path)
    base = _now_ms() - 10_000
    for i in range(5):
        _feed_quote(e, i + 1, base + i * 200)
    assert e.bucket_second == base // 1000
    assert e.bucket["n_updates"] == 5
    # cross into the next second
    _feed_quote(e, 10, base + 1000 + 5)
    assert e.bucket_second == base // 1000 + 1
    assert e.rolling[-1]["second"] == base // 1000
    assert e.rolling[-1]["n_updates"] == 5


def test_multiple_boundary_crossings_processed_once_each(tmp_path):
    """Regression: the duplicate-bucket race when two boundaries pass at once."""
    e = _engine(tmp_path)
    base = _now_ms() - 10_000
    _feed_quote(e, 1, base)
    _feed_quote(e, 2, base + 100)          # second 0
    # jump 3 seconds ahead in ONE event (event processing catches up)
    _feed_quote(e, 3, base + 3000 + 5)     # second 3
    seconds = [r["second"] for r in e.rolling]
    assert seconds == [base // 1000, base // 1000 + 1, base // 1000 + 2]
    # no duplicates and no skipped bucket in between
    assert len(set(seconds)) == 3


def test_duplicate_update_id_after_reconnect_ignored(tmp_path):
    e = _engine(tmp_path)
    base = _now_ms() - 10_000
    _feed_quote(e, 100, base, bid=100.0, ask=100.2)
    _feed_quote(e, 100, base + 50, bid=99.0, ask=99.2)   # same id: ignored
    assert e.book["bid"] == 100.0
    assert e.events_count == 1
    # reconnect resets state; fresh ids accepted again
    e.book = None
    e.last_update_id = -1
    _feed_quote(e, 5, base + 100, bid=101.0, ask=101.2)
    assert e.book["bid"] == 101.0


def test_crossed_quotes_never_applied(tmp_path):
    e = _engine(tmp_path)
    base = _now_ms() - 10_000
    _feed_quote(e, 1, base, bid=100.0, ask=100.2)
    _feed_quote(e, 2, base + 10, bid=100.5, ask=100.4)  # crossed
    assert e.book["bid"] == 100.0


def test_dual_stream_duplicate_trades_deduplicated(tmp_path):
    e = _engine(tmp_path)
    base = _now_ms() - 10_000
    _feed_trade(e, 77, base, px=100.1, qty=0.5)
    _feed_trade(e, 77, base, px=100.1, qty=0.5)  # same id+ts from the other stream
    assert e.bucket["n_trades"] == 1
    assert e.bucket["buy_vol"] == pytest.approx(0.5)


def test_stale_data_halts_decisions(tmp_path):
    import time as _t

    e = _engine(tmp_path)
    end_sec = int(_t.time()) - 5
    for i in range(120):  # warm the rolling window ending 5s ago
        _feed_quote(e, i + 1, (end_sec - 120 + i) * 1000, bid=100.0, ask=100.2)
    # feed silent from here; push the clock 300s ahead
    now = end_sec + 300
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_t, "time", lambda: now)
        e._boundary_tick()
        e._boundary_tick()
    # decisions on stale rows are skipped, not traded
    assert e.n_skipped >= 1


def test_kill_switch_halts_trading(tmp_path):
    import time as _t

    e = _engine(tmp_path)
    (tmp_path / "run").mkdir(parents=True, exist_ok=True)
    e.kill.path = tmp_path / "run" / "kill_switch"
    e.kill.path.write_text("stop")
    base = (int(_t.time()) - 200) * 1000
    for i in range(150):
        _feed_quote(e, i + 1, base + i * 1000, bid=100.0, ask=100.2)
    assert e.kill.check() is True
    e._boundary_tick()
    # no fill may happen after the switch trips
    assert e.n_fills == 0


def test_paper_equity_accounting_on_fill(tmp_path):
    e = _engine(tmp_path)
    base = (int(__import__("time").time()) - 200) * 1000
    for i in range(150):
        _feed_quote(e, i + 1, base + i * 1000, bid=100.0, ask=100.2, bidq=2.0, askq=2.0)
    # force a paper fill: strong signal
    e._decide_and_trade()  # baseline: likely no trade (hurdle vs zero pred)
    before = e.cash
    # simulate a forced execution
    e._paper_execute(1.0, 100.1, int(base / 1000), {"kind": "decision"})
    assert e.qty == pytest.approx(1.0)
    assert e.cash < before  # paid price + fee
    # close it out
    e._paper_execute(-1.0, 100.0, int(base / 1000) + 1, {"kind": "decision"})
    assert e.qty == pytest.approx(0.0)
    # round trip executes at BOOK prices +/- slippage, plus per-side fees:
    # buy at ask*(1+slip), sell at bid*(1-slip)
    slip = e.cost.slippage_bps / 1e4
    buy_px = 100.2 * (1 + slip)
    sell_px = 100.0 * (1 - slip)
    expected = (sell_px - buy_px) - (buy_px * 5e-4 + sell_px * 5e-4)
    assert (e.cash - before) == pytest.approx(expected, rel=1e-6)
    # logs written
    assert (tmp_path / "run" / "orders.jsonl").exists()
    assert (tmp_path / "run" / "fills.jsonl").exists()
    assert (tmp_path / "run" / "state.json").exists()


def test_state_persistence_across_restart(tmp_path):
    e1 = _engine(tmp_path)
    e1.cash = 9_500.0
    e1.qty = 0.4
    e1._save_state()
    e2 = _engine(tmp_path)
    assert e2.cash == 9_500.0
    assert e2.qty == pytest.approx(0.4)


def test_health_snapshot_written(tmp_path):
    e = _engine(tmp_path)
    e._heartbeat()
    h = json.loads((tmp_path / "run" / "health.json").read_text())
    assert h["events_count"] == 0
    assert "kill_switch" in h
    assert h["model"] == "RidgeModel"
