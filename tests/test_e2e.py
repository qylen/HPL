"""Synthetic generators: predictive (not contemporaneous) planting, noise, e2e."""
import numpy as np

from hpl1.backtest import run_cost_backtest
from hpl1.config import CostConfig, RiskConfig
from hpl1.dataset import build_dataset_days, load_dataset
from hpl1.features import build_features, feature_columns_for
from hpl1.ingest import ingest_archive
from hpl1.markets import get_market
from hpl1.synthetic import generate_synthetic_dataset, write_synthetic_archive
from hpl1.training import compare_models


def test_planted_signal_is_predictive_not_contemporaneous():
    """The planted relation must link features at t to the return t -> t+1."""
    frame = generate_synthetic_dataset("/tmp/syn_check.csv", rows=8000, seed=13)
    ofi = frame["ofi_norm"].to_numpy()
    target = frame["target_return"].to_numpy()
    corr_next = np.corrcoef(ofi[:-1], target[:-1])[0, 1]   # target[t] = ret t->t+1
    corr_now = np.corrcoef(ofi[1:], frame["mid_return_1s"].to_numpy()[1:])[0, 1]
    assert corr_next > 0.15        # predictive link exists
    # features must NOT be contemporaneous with their own second's return:
    # mid_return_1s[t] is the PAST return; ofi[t] must not predict it better
    assert corr_next > abs(corr_now) + 0.05


def test_label_consistency_with_price_path():
    frame = generate_synthetic_dataset("/tmp/syn_check2.csv", rows=5000, seed=7)
    mid = frame["mid"].to_numpy()
    target = frame["target_return"].to_numpy()
    expected = np.log(mid[1:] / mid[:-1])
    assert np.allclose(target[:-1], expected, atol=1e-12)
    assert np.isnan(target[-1])


def test_noise_dataset_has_no_planted_edge():
    frame = generate_synthetic_dataset("/tmp/syn_noise.csv", rows=8000, seed=13, planted=False)
    ofi = frame["ofi_norm"].to_numpy()
    target = frame["target_return"].to_numpy()
    corr = abs(np.corrcoef(ofi[:-1], target[:-1])[0, 1])
    assert corr < 0.05  # no relationship by construction


# ------------------------------------------------------------------ e2e tests

def _run_pipeline(raw_dir, proc_dir, planted, corrupt=False):
    info = write_synthetic_archive(
        raw_dir, market="futures", symbol="SYNUSD", day="2025-06-01", seconds=7200,
        planted=planted, gap_seconds=40 if corrupt else 0,
        duplicate_events=25 if corrupt else 0, out_of_order_events=15 if corrupt else 0,
        crossed_quotes=10 if corrupt else 0)
    stats = {}
    for dtype in ("bookTicker", "aggTrades", "fundingRate"):
        r = ingest_archive("futures", "SYNUSD", dtype, "2025-06-01", raw_dir, proc_dir,
                           rebuild=True)
        stats[dtype] = r["stats"]
    build_dataset_days("futures", "SYNUSD", "2025-06-01", "2025-06-01", proc_dir, rebuild=True)
    raw = load_dataset("futures", "SYNUSD", "2025-06-01", "2025-06-01", proc_dir)
    feats = build_features(raw, feature_set="full")
    fcols = [c for c in feature_columns_for("full") if c in feats.columns]
    out = compare_models(feats, fcols, ["zero", "ridge", "hist", "law"], seed=13)
    cost = CostConfig(taker_fee_bps=0.0, slippage_bps=0.0, fee_scenario="zero")
    risk = RiskConfig(starting_capital=10_000.0, position_fraction=1.0,
                      cooldown_seconds=0, min_edge_bps=0.0)
    bt = run_cost_backtest(out["predictions"], cost, risk, get_market("futures"),
                           tick_size="0.01", step_size="0.001", min_notional=5.0)
    return info, stats, feats, out, bt


def test_e2e_planted_relationship_recovered(tmp_path):
    """Full pipeline on synthetic Binance-format archives: the planted
    one-second-ahead relationship must be recovered end to end."""
    info, stats, feats, out, bt = _run_pipeline(tmp_path / "raw", tmp_path / "proc",
                                                planted=True)
    assert info["planted"] is True
    assert stats["bookTicker"]["applied_events"] > 20_000
    best = out["best"]
    test_metrics = out["results"][best]["test"]
    assert test_metrics["oos_r2_vs_zero"] > 0.05, test_metrics
    assert test_metrics["prediction_return_correlation"] > 0.2
    # labels aligned through the whole pipeline
    from hpl1.validation import check_label_alignment

    assert check_label_alignment(feats)["passed"]
    # zero-fee backtest should trade and CAPTURE the planted edge:
    # a positive net return here proves prediction->price alignment end-to-end
    assert bt["report"]["n_trades"] > 0
    assert bt["report"]["total_net_return_pct"] > 0.05, bt["report"]["total_net_return_pct"]


def test_e2e_noise_not_turned_into_fake_profit(tmp_path):
    """Control: with no planted relationship the pipeline must NOT report a
    profitable edge at realistic costs (multiple-testing discipline)."""
    info, stats, feats, out, bt = _run_pipeline(tmp_path / "raw2", tmp_path / "proc2",
                                                planted=False)
    test_metrics = out["results"][out["best"]]["test"]
    assert abs(test_metrics["oos_r2_vs_zero"]) < 0.02
    cost = CostConfig(taker_fee_bps=5.0, slippage_bps=1.0)
    bt_real = run_cost_backtest(out["predictions"], cost,
                                RiskConfig(starting_capital=10_000.0, cooldown_seconds=0,
                                           min_edge_bps=0.0),
                                get_market("futures"), tick_size="0.01",
                                step_size="0.001", min_notional=5.0)
    assert bt_real["report"]["total_net_return_pct"] <= 0.01  # no fake profit


def test_e2e_corrupted_archive_handled_with_warnings(tmp_path):
    """Duplicates, out-of-order ids, crossed quotes and gaps: counted, skipped,
    flagged — never silently ingested."""
    info, stats, feats, out, bt = _run_pipeline(tmp_path / "raw3", tmp_path / "proc3",
                                                planted=True, corrupt=True)
    book_stats = stats["bookTicker"]
    assert book_stats["dup_events"] >= 20
    assert book_stats["out_of_order"] >= 10
    assert book_stats["crossed_quotes"] >= 8
    # gaps appear as data_gap seconds in the feature frame
    assert int(feats["data_gap"].sum()) > 0
    # despite corruption the planted signal is still recoverable
    assert out["results"][out["best"]]["test"]["oos_r2_vs_zero"] > 0.03


def test_e2e_research_demo_cli(tmp_path):
    """The demo command must run the legacy path end to end on synthetic data."""
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, "-m", "hpl1", "demo", "--rows", "1200",
         "--directory", str(tmp_path / "demo")],
        cwd=".", capture_output=True, text=True, timeout=240)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (tmp_path / "demo" / "model.joblib").exists()
    assert (tmp_path / "demo" / "backtest_report.json").exists()
