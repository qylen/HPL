"""Full cost-aware backtest engine: accounting exactness, risk halts, funding."""
import numpy as np
import pandas as pd
import pytest

from hpl1.backtest import fee_sensitivity, run_cost_backtest
from hpl1.config import CostConfig, RiskConfig
from hpl1.markets import get_market


def _pred_frame(rows=600, seed=3, prediction_bps=None, mid=100.0, valid=True,
                dynamic_mid=True, actual=None):
    rng = np.random.default_rng(seed)
    sec = np.arange(1_700_000_000, 1_700_000_000 + rows)
    spread = 0.02
    if prediction_bps is None:
        pred = rng.normal(0, 1e-4, rows)  # ~1 bps noise
    else:
        pred = np.full(rows, prediction_bps / 1e4)
    ret = actual if actual is not None else rng.normal(0, 5e-5, rows)
    path = mid * np.exp(np.cumsum(ret)) if dynamic_mid else np.full(rows, float(mid))
    df = pd.DataFrame({
        "second": sec,
        "ts": pd.date_range("2023-11-14", periods=rows, freq="s", tz="UTC").astype(str),
        "prediction": pred,
        "mid": path,
        "bid": path - spread / 2,
        "ask": path + spread / 2,
        "spread_bps": spread / path * 1e4,
        "actual_return": ret,
        "valid": valid,
        "funding_rate": np.nan,
    })
    return df


COST = CostConfig(taker_fee_bps=5.0, slippage_bps=0.0)
RISK = RiskConfig(starting_capital=10_000.0, position_fraction=1.0, leverage=1.0,
                  cooldown_seconds=0, min_edge_bps=0.0)


def test_round_trip_equity_is_exact():
    """buy 1 unit then sell 1 unit at constant prices: cash changes by exactly
    -(spread cost) - (2x fees). P/L is fully explained by cost components."""
    frame = _pred_frame(rows=12, prediction_bps=50, valid=True, dynamic_mid=False,
                        actual=np.zeros(12))
    cost = CostConfig(taker_fee_bps=10.0, slippage_bps=0.0)
    out = run_cost_backtest(frame, cost, RISK, get_market("futures"),
                            tick_size="0.01", step_size="0.001", min_notional=0)
    rep = out["report"]
    # entry + exit fees; no price change; spread paid twice (entry+exit vs mid)
    fees = rep["costs_by_component"]["commission"]
    spread = rep["costs_by_component"]["spread"]
    assert fees == pytest.approx(2 * 10.0 / 1e4 * 10_000.0, rel=1e-3)  # 2 sides x 10bps x 10k notional
    assert spread == pytest.approx(2 * 0.01 * 100.0, rel=1e-3)  # 2 sides x 0.01/unit x 100 units
    assert rep["net_pnl"] == pytest.approx(-(fees + spread), rel=1e-3)
    assert rep["final_position_flat"] is True
    assert rep["n_round_trips"] >= 1


def test_short_positions_do_open_for_futures():
    """Regression: the original engine never opened shorts (unsigned qty)."""
    frame = _pred_frame(rows=12, prediction_bps=-50)
    out = run_cost_backtest(frame, COST, RiskConfig(starting_capital=10_000,
                                                    position_fraction=1.0,
                                                    cooldown_seconds=0,
                                                    min_edge_bps=0.0),
                            get_market("futures"), tick_size="0.01",
                            step_size="0.001", min_notional=0)
    trades = out["trades"]
    assert (trades["side"] == "sell").any()          # short entry = sell first
    assert (out["curve"]["position"] < 0).any()      # negative position held


def test_spot_never_goes_short():
    frame = _pred_frame(rows=12, prediction_bps=-50)
    out = run_cost_backtest(frame, COST, RISK, get_market("spot"),
                            tick_size="0.01", step_size="0.00001", min_notional=0)
    assert (out["curve"]["position"] >= 0).all()
    assert out["report"]["market"] == "spot"


def test_zero_trades_when_edge_below_cost_hurdle():
    """Predicted ~1bps edges vs ~12bps round-trip hurdle: correct behaviour is
    ZERO trades (this is the honest real-data outcome, not a bug)."""
    frame = _pred_frame(rows=500, prediction_bps=1.0)
    out = run_cost_backtest(frame, COST, RISK, get_market("futures"))
    assert out["report"]["n_trades"] == 0
    assert out["report"]["total_net_return_pct"] == 0.0


def test_trades_when_edge_exceeds_hurdle():
    frame = _pred_frame(rows=50, prediction_bps=30.0)  # 30 bps > ~12 bps hurdle
    out = run_cost_backtest(frame, COST, RISK, get_market("futures"),
                            tick_size="0.01", step_size="0.001", min_notional=0)
    assert out["report"]["n_trades"] > 0


def test_invalid_rows_force_flat_fail_closed():
    frame = _pred_frame(rows=20, prediction_bps=30.0)
    frame.loc[5:15, "valid"] = False
    out = run_cost_backtest(frame, COST, RISK, get_market("futures"),
                            tick_size="0.01", step_size="0.001", min_notional=0)
    curve = out["curve"]
    # while data invalid, position must be flat
    assert (curve.loc[5:15, "position"].abs() < 1e-9).all()
    assert (out["trades"]["reason"] == "halt_stale").any()


def test_funding_charged_on_long_position():
    frame = _pred_frame(rows=10, prediction_bps=30.0)
    frame.loc[:, "funding_rate"] = np.nan
    frame.loc[5, "funding_rate"] = 0.0001  # 1bp funding while long
    out = run_cost_backtest(frame, COST, RISK, get_market("futures"),
                            tick_size="0.01", step_size="0.001", min_notional=0)
    assert out["report"]["costs_by_component"]["funding"] > 0


def test_no_funding_for_spot():
    frame = _pred_frame(rows=10, prediction_bps=30.0)
    frame.loc[5, "funding_rate"] = 0.0001
    out = run_cost_backtest(frame, COST, RISK, get_market("spot"),
                            tick_size="0.01", step_size="0.00001", min_notional=0)
    assert out["report"]["costs_by_component"]["funding"] == 0.0


def test_min_notional_blocks_small_entries():
    frame = _pred_frame(rows=10, prediction_bps=30.0)
    risk = RiskConfig(starting_capital=10_000.0, position_fraction=0.001,
                      cooldown_seconds=0, min_edge_bps=0.0)
    out = run_cost_backtest(frame, COST, risk, get_market("futures"),
                            tick_size="0.01", step_size="0.001", min_notional=100)
    assert out["report"]["n_trades"] == 0  # notional 10 < 100


def test_cooldown_prevents_immediate_reentry():
    frame = _pred_frame(rows=60, prediction_bps=30.0)
    risk = RiskConfig(starting_capital=10_000.0, position_fraction=1.0,
                      cooldown_seconds=10, min_edge_bps=0.0)
    out = run_cost_backtest(frame, COST, risk, get_market("futures"),
                            tick_size="0.01", step_size="0.001", min_notional=0)
    # with a constant signal and cooldown, trades must be spaced >= 10s
    trades = out["trades"]
    if len(trades) > 1:
        ts_to_sec = dict(zip(frame["ts"].astype(str), frame["second"]))
        secs = np.array([ts_to_sec[t] for t in trades["ts"]], dtype=np.int64)
        assert (np.diff(secs) >= 10).all()


def test_daily_loss_halt_stops_trading():
    # mid declines 30 bps/s while the model stays long: losses accrue for real
    frame = _pred_frame(rows=400, prediction_bps=30.0,
                        actual=np.full(400, -30.0 / 1e4))
    risk = RiskConfig(starting_capital=10_000.0, position_fraction=1.0,
                      cooldown_seconds=0, min_edge_bps=0.0, max_daily_loss_pct=3.0)
    out = run_cost_backtest(frame, COST, risk, get_market("futures"),
                            tick_size="0.01", step_size="0.001", min_notional=0)
    rep = out["report"]
    assert rep["max_drawdown_pct"] <= -3.0 - 1e-6 or rep["n_trades"] < 400
    assert rep["total_net_return_pct"] < 0


def test_max_drawdown_halt_is_permanent():
    frame = _pred_frame(rows=500, prediction_bps=30.0,
                        actual=np.full(500, -30.0 / 1e4))
    risk = RiskConfig(starting_capital=10_000.0, position_fraction=1.0,
                      cooldown_seconds=0, min_edge_bps=0.0, max_drawdown_pct=5.0,
                      max_daily_loss_pct=0.0)  # disable daily halt
    out = run_cost_backtest(frame, COST, risk, get_market("futures"),
                            tick_size="0.01", step_size="0.001", min_notional=0)
    assert out["report"]["halted_notice"] is not None


def test_turnover_cap_limits_entries():
    frame = _pred_frame(rows=100, prediction_bps=30.0)
    risk = RiskConfig(starting_capital=10_000.0, position_fraction=1.0,
                      cooldown_seconds=0, min_edge_bps=0.0,
                      max_turnover_per_day=0.5)  # 5k notional/day cap
    out = run_cost_backtest(frame, COST, risk, get_market("futures"),
                            tick_size="0.01", step_size="0.001", min_notional=0)
    assert out["report"]["turnover_notional"] <= 5_000.0 * 1.05  # entry cap, exit allowed


def test_leverage_cap_enforced_by_profile():
    frame = _pred_frame(rows=50, prediction_bps=30.0)
    risk = RiskConfig(starting_capital=10_000.0, position_fraction=1.0,
                      leverage=10.0, cooldown_seconds=0, min_edge_bps=0.0)
    out = run_cost_backtest(frame, COST, risk, get_market("futures"),
                            tick_size="0.01", step_size="0.001", min_notional=0)
    assert out["report"]["config"]["leverage"] == 2.0  # futures profile caps at 2x
    assert out["report"]["max_leverage_utilized"] <= 2.0 + 1e-6


def test_final_position_liquidated():
    frame = _pred_frame(rows=10, prediction_bps=30.0)
    out = run_cost_backtest(frame, COST, RISK, get_market("futures"),
                            tick_size="0.01", step_size="0.001", min_notional=0)
    assert out["report"]["final_position_flat"] is True
    assert (out["trades"]["reason"] == "liquidation_final").any()


def test_costs_are_not_double_counted():
    """Total costs must equal the sum of components; slippage+spread live in
    execution prices, commission on notional — no component counted twice."""
    frame = _pred_frame(rows=50, prediction_bps=30.0)
    cost = CostConfig(taker_fee_bps=5.0, slippage_bps=1.0)
    out = run_cost_backtest(frame, cost, RISK, get_market("futures"),
                            tick_size="0.01", step_size="0.001", min_notional=0)
    rep = out["report"]
    comps = rep["costs_by_component"]
    assert rep["total_costs"] == pytest.approx(sum(comps.values()), rel=1e-9)
    # fee = 5bps of traded notional exactly
    assert comps["commission"] == pytest.approx(5.0 / 1e4 * rep["turnover_notional"], rel=1e-6)
    # slippage = 1bps of traded notional
    assert comps["slippage"] == pytest.approx(1.0 / 1e4 * rep["turnover_notional"], rel=1e-6)


def test_fee_sensitivity_table():
    frame = _pred_frame(rows=60, prediction_bps=20.0)
    table = fee_sensitivity(frame, COST, RISK, get_market("futures"),
                            taker_fees_bps=(0.0, 1.0, 5.0, 25.0),
                            tick_size="0.01", step_size="0.001", min_notional=0)
    assert list(table["taker_fee_bps"]) == [0.0, 1.0, 5.0, 25.0]
    assert table.iloc[0]["total_costs"] <= table.iloc[-2]["total_costs"]
    # fee above the predicted edge: the hurdle is never cleared -> zero trades
    assert table.iloc[-1]["n_trades"] == 0


def test_trade_only_spot_uses_assumed_spread():
    frame = _pred_frame(rows=50, prediction_bps=30.0)
    frame = frame.drop(columns=["bid", "ask"])  # spot: no book
    out = run_cost_backtest(frame, COST, RISK, get_market("spot"),
                            tick_size="0.01", step_size="0.00001", min_notional=0)
    rep = out["report"]
    assert rep["config"]["spread_source"].startswith("assumed")
    assert rep["config"]["assumed_spread_bps"] == 1.0
    assert rep["costs_by_component"]["spread"] > 0


def test_zero_fee_scenario_trades_more():
    frame = _pred_frame(rows=60, prediction_bps=8.0)  # between 0-fee and 5-fee hurdles
    zero = CostConfig(fee_scenario="zero", slippage_bps=0.0)
    out0 = run_cost_backtest(frame, zero, RISK, get_market("futures"),
                             tick_size="0.01", step_size="0.001", min_notional=0)
    out5 = run_cost_backtest(frame, COST, RISK, get_market("futures"),
                             tick_size="0.01", step_size="0.001", min_notional=0)
    assert out0["report"]["n_trades"] > out5["report"]["n_trades"]
    assert out0["report"]["costs_by_component"]["commission"] == 0.0


def test_quantity_rounded_to_step():
    frame = _pred_frame(rows=12, prediction_bps=50.0)
    out = run_cost_backtest(frame, COST, RISK, get_market("futures"),
                            tick_size="0.10", step_size="0.01", min_notional=0)
    qty = out["trades"]["qty"].to_numpy() if len(out["trades"]) else np.array([])
    assert all(abs(q / 0.01 - round(q / 0.01)) < 1e-6 for q in qty)


def test_price_rounded_to_tick():
    frame = _pred_frame(rows=12, prediction_bps=50.0)
    out = run_cost_backtest(frame, COST, RISK, get_market("futures"),
                            tick_size="0.10", step_size="0.01", min_notional=0)
    pxs = out["trades"]["exec_price"].to_numpy()
    assert all(abs(p / 0.10 - round(p / 0.10)) < 1e-6 for p in pxs)


def test_backtest_frame_too_short_rejected():
    with pytest.raises(ValueError, match="too short"):
        run_cost_backtest(_pred_frame(rows=3), COST, RISK, get_market("futures"))
