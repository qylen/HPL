"""Cost-aware one-second position backtest and HAC mean-return inference."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def hac_mean_test(values: np.ndarray, max_lags: int = 60) -> dict[str, float]:
    """Newey-West/HAC z-test for mean net return; approximate for large samples."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return {"hac_lags": 0, "hac_z_score": 0.0, "hac_p_value_two_sided": 1.0}
    mean = float(x.mean())
    centered = x - mean
    lag_count = min(max(0, int(max_lags)), n - 1)
    long_run_variance = float(np.dot(centered, centered) / n)
    for lag in range(1, lag_count + 1):
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        weight = 1.0 - lag / (lag_count + 1.0)
        long_run_variance += 2.0 * weight * gamma
    if not np.isfinite(long_run_variance) or long_run_variance <= 0:
        return {"hac_lags": lag_count, "hac_z_score": 0.0, "hac_p_value_two_sided": 1.0}
    standard_error = math.sqrt(long_run_variance / n)
    z = mean / standard_error if standard_error > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return {"hac_lags": lag_count, "hac_z_score": float(z), "hac_p_value_two_sided": float(p)}


def run_backtest(
    predictions: pd.DataFrame,
    taker_fee_bps: float = 5.0,
    slippage_bps: float = 1.0,
    min_edge_bps: float = 1.0,
    risk_buffer_bps: float = 0.0,
    hac_lags: int = 60,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Backtest a {-1,0,+1} target-position policy.

    `prediction` and `actual_return` are log returns over the next one-second
    interval. Costs are charged on changes in position and on final liquidation.
    Taker fee and slippage are per side; spread is observed full quoted spread.
    The threshold uses an approximate round-trip cost plus a configurable buffer.
    """
    required = {"timestamp", "actual_return", "prediction", "spread_bps"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Prediction file missing columns: {sorted(required - set(predictions.columns))}")
    if taker_fee_bps < 0 or slippage_bps < 0 or min_edge_bps < 0 or risk_buffer_bps < 0:
        raise ValueError("Fees, slippage, edge threshold, and risk buffer must be non-negative")
    df = predictions.copy()
    for col in ("actual_return", "prediction", "spread_bps"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["actual_return", "prediction", "spread_bps"])
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid prediction rows to backtest")

    pred_bps = df["prediction"].to_numpy(dtype=float) * 10_000.0
    actual_log_return = df["actual_return"].to_numpy(dtype=float)
    spread_bps = df["spread_bps"].clip(lower=0).to_numpy(dtype=float)
    # Approximate cost to enter and exit once: both sides of quoted spread plus fees/slippage.
    round_trip_cost_bps = 2.0 * (taker_fee_bps + slippage_bps) + spread_bps
    decision_threshold = np.maximum(min_edge_bps, round_trip_cost_bps + risk_buffer_bps)
    position = np.where(pred_bps > decision_threshold, 1.0, np.where(pred_bps < -decision_threshold, -1.0, 0.0))
    previous_position = np.r_[0.0, position[:-1]]
    turnover = np.abs(position - previous_position)
    one_way_cost_bps = taker_fee_bps + slippage_bps + spread_bps / 2.0
    trading_cost = turnover * one_way_cost_bps / 10_000.0
    gross_return = np.expm1(position * actual_log_return)
    net_return = gross_return - trading_cost
    # Liquidate any open position at the end of the test instead of giving it away for free.
    liquidation_cost = np.zeros(len(df), dtype=float)
    liquidation_cost[-1] = abs(position[-1]) * one_way_cost_bps[-1] / 10_000.0
    net_return -= liquidation_cost
    equity = np.cumprod(1.0 + net_return)
    peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    drawdown = equity / peak - 1.0
    active = position != 0
    entries = (position != previous_position) & active
    stats = {
        "rows": len(df),
        "total_return_pct": float((equity[-1] - 1.0) * 100.0),
        "gross_return_sum_bps_approx": float(gross_return.sum() * 10_000.0),
        "total_cost_bps_approx": float((trading_cost.sum() + liquidation_cost.sum()) * 10_000.0),
        "mean_net_return_bps_per_second": float(net_return.mean() * 10_000.0),
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "active_seconds_pct": float(active.mean() * 100.0),
        "position_entries_or_reversals": int(entries.sum()),
        "position_changes": int(np.count_nonzero(turnover)),
        "total_position_turnover_units": float(turnover.sum() + abs(position[-1])),
        "net_positive_active_seconds_pct": float((net_return[active] > 0).mean() * 100.0) if active.any() else 0.0,
        "mean_prediction_bps_abs": float(np.mean(np.abs(pred_bps))),
        "mean_decision_threshold_bps": float(np.mean(decision_threshold)),
        "taker_fee_bps_per_side": float(taker_fee_bps),
        "slippage_bps_per_side": float(slippage_bps),
        "risk_buffer_bps": float(risk_buffer_bps),
        "iid_sharpe_per_sqrt_second": float(net_return.mean() / net_return.std(ddof=1)) if len(net_return) > 1 and net_return.std(ddof=1) > 0 else 0.0,
        "warning": "Backtest estimates only. Fill probability, queue position, latency, market impact and funding are not fully simulated.",
    }
    stats.update(hac_mean_test(net_return, max_lags=hac_lags))
    result = df.copy()
    result["prediction_bps"] = pred_bps
    result["decision_threshold_bps"] = decision_threshold
    result["position"] = position
    result["turnover"] = turnover
    result["gross_return"] = gross_return
    result["trading_cost"] = trading_cost
    result["liquidation_cost"] = liquidation_cost
    result["net_return"] = net_return
    result["equity"] = equity
    result["drawdown"] = drawdown
    return result, stats


def backtest_files(
    predictions_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    taker_fee_bps: float = 5.0,
    slippage_bps: float = 1.0,
    min_edge_bps: float = 1.0,
    risk_buffer_bps: float = 0.0,
    hac_lags: int = 60,
) -> dict[str, Any]:
    predictions = pd.read_csv(predictions_path)
    result, report = run_backtest(
        predictions,
        taker_fee_bps=taker_fee_bps,
        slippage_bps=slippage_bps,
        min_edge_bps=min_edge_bps,
        risk_buffer_bps=risk_buffer_bps,
        hac_lags=hac_lags,
    )
    output_path = Path(output_path)
    report_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# =============================================================================
# Full cost-aware backtest engine (Phase E upgrade)
# =============================================================================

import datetime as _dt
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal

from .config import CostConfig, RiskConfig
from .markets import MarketProfile
from .statistics import hac_mean_test as _hac


@dataclass
class TradeRecord:
    ts: str
    side: str          # buy | sell
    qty: float
    exec_price: float
    notional: float
    fee: float
    spread_cost: float
    slippage_cost: float
    reason: str        # entry | exit | reversal | liquidation_final | halt_stale


@dataclass
class RoundTrip:
    entry_ts: str
    exit_ts: str
    direction: int
    qty: float
    entry_price: float
    exit_price: float
    gross_pnl: float
    fees: float
    net_pnl: float


@dataclass
class _State:
    cash: float
    qty: float = 0.0
    peak_equity: float = 0.0
    halted_drawdown: bool = False
    halted_day: str = ""
    last_trade_second: int = -10**9
    day_realized_start: float = 0.0
    turnover_today: float = 0.0
    entry_price: float = 0.0       # average entry for the open position
    entry_ts: str = ""
    costs: dict = field(default_factory=lambda: {
        "commission": 0.0, "spread": 0.0, "slippage": 0.0, "funding": 0.0})


def _round_step(qty: float, step: str) -> float:
    step_d = Decimal(step)
    if step_d <= 0:
        return float(qty)
    q = Decimal(str(qty)) / step_d
    return float((q.to_integral_value(rounding=ROUND_DOWN)) * step_d)


def _round_tick(px: float, tick: str) -> float:
    tick_d = Decimal(tick)
    if tick_d <= 0 or not np.isfinite(px) or px <= 0:
        return float(px)
    p = Decimal(str(px)) / tick_d
    return float((p.to_integral_value(rounding=ROUND_HALF_UP)) * tick_d)


def run_cost_backtest(
    pred_frame: pd.DataFrame,
    cost: CostConfig,
    risk: RiskConfig,
    profile: MarketProfile,
    tick_size: str = "0.10",
    step_size: str = "0.001",
    min_notional: float = 100.0,
    assumed_spread_bps: float | None = None,
    maintenance_margin_rate: float = 0.004,
    hac_lags: int = 120,
) -> dict[str, Any]:
    """Full-mechanics backtest on per-second test predictions.

    Accounting model (no double counting):
    - market buys execute at ask*(1+slippage), sells at bid*(1-slippage);
      the spread cost is the bid/ask distance from mid INSIDE those prices,
      slippage is the extra distance beyond the touch, commission is charged
      on executed notional per side. Funding applies to positions held across
      funding timestamps (long pays positive rate). All P/L is marked at mid.
    - spot profiles cannot go short and have no funding.
    - invalid/stale rows force the position flat (fail closed).
    - the final open position is liquidated at the last valid quote.
    """
    required = {"second", "ts", "prediction", "mid"}
    missing = required - set(pred_frame.columns)
    if missing:
        raise ValueError(f"pred frame missing columns: {sorted(missing)}")
    df = pred_frame.sort_values("second", kind="stable").reset_index(drop=True)
    n = len(df)
    if n < 10:
        raise ValueError("backtest frame too short")

    has_book = {"bid", "ask"}.issubset(df.columns) and df["bid"].notna().any()
    mid = df["mid"].to_numpy(dtype=float)
    bid = df["bid"].to_numpy(dtype=float) if has_book else np.full(n, np.nan)
    ask = df["ask"].to_numpy(dtype=float) if has_book else np.full(n, np.nan)
    spread_bps = np.abs(df["spread_bps"].to_numpy(dtype=float)) if "spread_bps" in df.columns \
        else np.full(n, np.nan)
    if not has_book and assumed_spread_bps is None:
        assumed_spread_bps = 1.0  # documented assumption for trade-only price proxies
    if not has_book:
        spread_bps = np.full(n, float(assumed_spread_bps))
    pred = df["prediction"].to_numpy(dtype=float)
    valid = df["valid"].to_numpy(dtype=bool) if "valid" in df.columns else np.ones(n, dtype=bool)
    funding = df["funding_rate"].to_numpy(dtype=float) if "funding_rate" in df.columns \
        else np.full(n, np.nan)
    seconds = df["second"].to_numpy(dtype=np.int64)

    slip = cost.slippage_bps / 1e4
    fee_rate = cost.taker_fee_bps / 1e4
    leverage = profile.validate_leverage(risk.leverage)
    allow_short = profile.allow_short
    state = _State(cash=risk.starting_capital)
    state.peak_equity = risk.starting_capital
    state.day_realized_start = risk.starting_capital

    equity = np.full(n, np.nan)
    position = np.zeros(n)
    trades: list[TradeRecord] = []
    round_trips: list[RoundTrip] = []
    margin_breach_rows = 0
    mid = pd.Series(mid).ffill().to_numpy(dtype=float)  # guard isolated NaN mids

    def _day_of(sec: int) -> str:
        return _dt.datetime.fromtimestamp(sec, _dt.timezone.utc).date().isoformat()

    def _exec_prices(i: int) -> tuple[float, float]:
        if has_book and np.isfinite(ask[i]) and np.isfinite(bid[i]) and ask[i] > bid[i] > 0:
            buy_px = ask[i] * (1.0 + slip)
            sell_px = bid[i] * (1.0 - slip)
        else:
            half_spread = (spread_bps[i] if np.isfinite(spread_bps[i]) else 1.0) / 2.0 / 1e4
            buy_px = mid[i] * (1.0 + half_spread + slip)
            sell_px = mid[i] * (1.0 - half_spread - slip)
        return _round_tick(buy_px, tick_size), _round_tick(sell_px, tick_size)

    def _close_position(i: int, qty_now: float, reason: str) -> None:
        """Fully reduce the open position at row i prices (risk halt / final liquidation)."""
        nonlocal state
        if state.qty == 0 or qty_now == 0:
            return
        buy_px, sell_px = _exec_prices(i)
        if state.qty > 0:
            px, side = sell_px, "sell"
            spread_cost = max(0.0, (mid[i] - px)) * abs(state.qty) if has_book else \
                (spread_bps[i] / 2.0 / 1e4) * mid[i] * abs(state.qty)
        else:
            px, side = buy_px, "buy"
            spread_cost = max(0.0, (px - mid[i])) * abs(state.qty) if has_book else \
                (spread_bps[i] / 2.0 / 1e4) * mid[i] * abs(state.qty)
        q = abs(state.qty)
        notional = q * px
        fee = notional * fee_rate
        slip_cost = notional * slip
        state.cash += (px * state.qty) - fee
        state.costs["commission"] += fee
        state.costs["spread"] += float(spread_cost)
        state.costs["slippage"] += slip_cost
        gross = (px - state.entry_price) * state.qty
        net = gross - fee
        round_trips.append(RoundTrip(
            entry_ts=state.entry_ts, exit_ts=str(df["ts"].iloc[i]), direction=int(np.sign(state.qty)),
            qty=q, entry_price=state.entry_price, exit_price=px,
            gross_pnl=float(gross), fees=float(fee), net_pnl=float(net)))
        trades.append(TradeRecord(str(df["ts"].iloc[i]), side, q, px, notional, fee,
                                  float(spread_cost), float(slip_cost), reason))
        state.turnover_today += notional
        state.last_trade_second = int(seconds[i])
        state.qty = 0.0
        state.entry_price = 0.0

    halted_notice = None
    for i in range(n):
        day = _day_of(int(seconds[i]))
        eq_now = state.cash + state.qty * mid[i]
        if day != state.halted_day:
            state.halted_day = day
            state.day_realized_start = eq_now
            state.turnover_today = 0.0
        equity[i] = eq_now
        state.peak_equity = max(state.peak_equity, eq_now)

        # --- risk halts -----------------------------------------------------
        if not state.halted_drawdown and risk.max_drawdown_pct > 0:
            dd = eq_now / state.peak_equity - 1.0
            if dd <= -risk.max_drawdown_pct / 100.0:
                state.halted_drawdown = True
                halted_notice = f"drawdown halt at {df['ts'].iloc[i]} (dd={dd:.2%})"
        day_loss = eq_now / state.day_realized_start - 1.0 if state.day_realized_start > 0 else 0.0
        day_halted = (risk.max_daily_loss_pct > 0) and (day_loss <= -risk.max_daily_loss_pct / 100.0)

        # --- funding payment (futures only) ---------------------------------
        if profile.has_funding and np.isfinite(funding[i]) and state.qty != 0:
            payment = state.qty * mid[i] * float(funding[i])
            state.cash -= payment
            state.costs["funding"] += abs(payment)
            equity[i] = state.cash + state.qty * mid[i]

        # --- margin diagnostics (futures) ------------------------------------
        if profile.is_derivative and state.qty != 0:
            notional = abs(state.qty) * mid[i]
            required = notional * (1.0 / leverage + maintenance_margin_rate)
            if equity[i] < required:
                margin_breach_rows += 1

        # --- target position --------------------------------------------------
        target_qty = 0.0
        can_trade = (not state.halted_drawdown) and (not day_halted) and valid[i]
        if not valid[i] and state.qty != 0:
            _close_position(i, state.qty, "halt_stale")
        elif can_trade:
            pred_bps = pred[i] * 1e4
            sp_bps = spread_bps[i] if np.isfinite(spread_bps[i]) else 1.0
            hurdle = max(risk.min_edge_bps, cost.round_trip_cost_bps(sp_bps))
            if abs(pred_bps) >= hurdle:
                conviction = min(1.0, abs(pred_bps) / hurdle)
                max_notional = equity[i] * leverage * risk.position_fraction
                target_notional = conviction * max_notional
                direction = 1 if pred_bps > 0 else (-1 if allow_short else 0)
                raw_qty = direction * target_notional / mid[i]
                target_qty = _round_step(raw_qty, step_size)
                if not allow_short:
                    target_qty = max(0.0, target_qty)
                target_qty = direction * abs(target_qty) if direction != 0 else 0.0
            cooldown_active = (i * 0 + int(seconds[i]) - state.last_trade_second) < risk.cooldown_seconds
            if cooldown_active:
                target_qty = state.qty
            # turnover cap: block increases that exceed the daily cap
            delta_notional = abs(target_qty - state.qty) * mid[i]
            increasing = abs(target_qty) > abs(state.qty)
            if increasing and risk.max_turnover_per_day > 0:
                if state.turnover_today + delta_notional > risk.max_turnover_per_day * risk.starting_capital:
                    target_qty = state.qty

        # --- execute delta ----------------------------------------------------
        delta = target_qty - state.qty
        order_notional = abs(delta) * mid[i]
        is_close = (target_qty == 0.0 and state.qty != 0.0)
        if delta != 0 and (is_close or order_notional >= min_notional):
            buy_px, sell_px = _exec_prices(i)
            if delta > 0:
                px, side, spread_cost = buy_px, "buy", (buy_px - mid[i]) * delta
            else:
                px, side, spread_cost = sell_px, "sell", (mid[i] - sell_px) * (-delta)
            if not has_book:
                spread_cost = (spread_bps[i] / 2.0 / 1e4) * order_notional
            fee = order_notional * fee_rate
            slip_cost = order_notional * slip
            # cash flow for the traded delta
            state.cash -= delta * px
            state.cash -= fee
            state.costs["commission"] += fee
            state.costs["spread"] += float(spread_cost)
            state.costs["slippage"] += slip_cost
            state.turnover_today += order_notional
            state.last_trade_second = int(seconds[i])
            reason = "exit" if is_close else ("reversal" if state.qty * delta < 0 else "entry")
            trades.append(TradeRecord(str(df["ts"].iloc[i]), side, abs(delta), px, order_notional,
                                      fee, float(spread_cost), float(slip_cost), reason))
            # round-trip bookkeeping on reductions
            if state.qty != 0 and state.qty * delta < 0:
                matched = min(abs(state.qty), abs(delta))
                gross = (px - state.entry_price) * np.sign(state.qty) * matched
                rfee = matched * px * fee_rate
                round_trips.append(RoundTrip(
                    entry_ts=state.entry_ts, exit_ts=str(df["ts"].iloc[i]),
                    direction=int(np.sign(state.qty)), qty=matched,
                    entry_price=state.entry_price, exit_price=px,
                    gross_pnl=float(gross), fees=float(rfee), net_pnl=float(gross - rfee)))
            # update average entry for the remaining/new position
            if state.qty * delta > 0:  # increase
                total_q = state.qty + delta
                state.entry_price = (state.entry_price * abs(state.qty) + px * abs(delta)) / abs(total_q)
                state.entry_ts = str(df["ts"].iloc[i])
            elif state.qty * delta < 0 and abs(delta) >= abs(state.qty):
                state.entry_price = px
                state.entry_ts = str(df["ts"].iloc[i])
            elif state.qty * delta < 0:
                pass  # partial reduction: entry stats unchanged
            state.qty += delta
        position[i] = state.qty

    # --- final liquidation ----------------------------------------------------
    if state.qty != 0:
        i = n - 1
        _close_position(i, state.qty, "liquidation_final")
        equity[i] = state.cash + state.qty * mid[i]
        position[i] = state.qty

    # --- metrics ---------------------------------------------------------------
    eq = pd.Series(equity)
    net_ret = eq.pct_change().fillna(0.0)
    total_ret = eq.iloc[-1] / risk.starting_capital - 1.0
    peak = eq.cummax()
    dd = eq / peak - 1.0
    days = pd.Series([_day_of(int(s)) for s in seconds], index=eq.index)
    day_ret = eq.groupby(days).last().pct_change().dropna()
    losing_days = int((day_ret < 0).sum()) if len(day_ret) else 0
    gross_pnl = float(eq.iloc[-1] - risk.starting_capital) + sum(
        state.costs.values())
    # regime split by realized vol terciles of the test window (descriptive)
    regime_stats = {}
    if "actual_return" in df.columns:
        ar = df["actual_return"].to_numpy(dtype=float)
        rv = pd.Series(np.abs(ar)).rolling(300, min_periods=30).mean().to_numpy()
        terc = np.nanquantile(rv, [1 / 3, 2 / 3])
        labels = np.where(rv <= terc[0], "low_vol", np.where(rv <= terc[1], "mid_vol", "high_vol"))
        for lab in ("low_vol", "mid_vol", "high_vol"):
            m = labels == lab
            if m.sum() > 10:
                regime_stats[lab] = {
                    "seconds": int(m.sum()),
                    "mean_net_return_bps": float(net_ret[m].mean() * 1e4),
                }
    # exposure / leverage utilization
    exposure = np.abs(position) * mid
    max_exposure = float(np.nanmax(exposure)) if np.isfinite(exposure).any() else 0.0
    wins = [rt.net_pnl for rt in round_trips if rt.net_pnl > 0]
    losses = [rt.net_pnl for rt in round_trips if rt.net_pnl <= 0]
    hac = _hac(net_ret.to_numpy(dtype=float), max_lags=hac_lags)
    report = {
        "rows": int(n),
        "market": profile.name,
        "symbol": profile.symbol,
        "config": {
            "starting_capital": risk.starting_capital,
            "leverage": leverage,
            "position_fraction": risk.position_fraction,
            "taker_fee_bps": cost.taker_fee_bps,
            "maker_fee_bps": cost.maker_fee_bps,
            "slippage_bps": cost.slippage_bps,
            "fee_scenario": cost.fee_scenario,
            "use_spread": cost.use_spread,
            "assumed_spread_bps": None if has_book else float(assumed_spread_bps),
            "spread_source": "observed_quotes" if has_book else "assumed (no book history for market)",
            "min_edge_bps": risk.min_edge_bps,
        },
        "total_net_return_pct": float(total_ret * 100.0),
        "net_pnl": float(eq.iloc[-1] - risk.starting_capital),
        "gross_pnl_before_costs": float(gross_pnl),
        "costs_by_component": {k: float(v) for k, v in state.costs.items()},
        "total_costs": float(sum(state.costs.values())),
        "max_drawdown_pct": float(dd.min() * 100.0),
        "sharpe_per_second": float(net_ret.mean() / net_ret.std(ddof=1)) if len(net_ret) > 2 and net_ret.std(ddof=1) > 0 else 0.0,
        "sharpe_annualized_caveat": "per-second Sharpe annualized by sqrt(31536000) assumes i.i.d. scaling that one-second returns violate; treat as diagnostic only",
        "downside_deviation_bps": float(net_ret[net_ret < 0].std(ddof=1) * 1e4) if (net_ret < 0).any() else 0.0,
        "n_trades": len(trades),
        "n_round_trips": len(round_trips),
        "turnover_notional": float(sum(t.notional for t in trades)),
        "avg_trade_notional": float(np.mean([t.notional for t in trades])) if trades else 0.0,
        "win_rate": float(len(wins) / len(round_trips)) if round_trips else None,
        "avg_win": float(np.mean(wins)) if wins else 0.0,
        "avg_loss": float(np.mean(losses)) if losses else 0.0,
        "net_expectancy_per_round_trip": float(np.mean([rt.net_pnl for rt in round_trips])) if round_trips else None,
        "losing_days": losing_days,
        "trading_days": len(day_ret),
        "regime_performance": regime_stats,
        "max_exposure_notional": max_exposure,
        "max_leverage_utilized": float(max_exposure / risk.starting_capital),
        "margin_breach_seconds": int(margin_breach_rows),
        "halted_notice": halted_notice,
        "final_position_flat": bool(abs(state.qty) < 1e-12),
        "hac": hac,
        "warning": ("Simulated execution only. Queue position, partial fills, "
                    "market impact beyond slippage bps, exchange outages and "
                    "latency are not modelled; results are not a return forecast."),
    }
    curve = pd.DataFrame({"ts": df["ts"].astype(str), "second": seconds, "equity": equity,
                          "position": position, "net_return": net_ret.to_numpy(),
                          "drawdown": dd.to_numpy()})
    trades_df = pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame(
        columns=["ts", "side", "qty", "exec_price", "notional", "fee", "spread_cost",
                 "slippage_cost", "reason"])
    rt_df = pd.DataFrame([rt.__dict__ for rt in round_trips]) if round_trips else pd.DataFrame(
        columns=["entry_ts", "exit_ts", "direction", "qty", "entry_price", "exit_price",
                 "gross_pnl", "fees", "net_pnl"])
    return {"report": report, "curve": curve, "trades": trades_df, "round_trips": rt_df}


def fee_sensitivity(pred_frame: pd.DataFrame, cost: CostConfig, risk: RiskConfig,
                    profile: MarketProfile, taker_fees_bps=(0.0, 0.5, 1.0, 2.0, 5.0),
                    **engine_kwargs) -> pd.DataFrame:
    """Pure taker-fee sensitivity curve: slippage zeroed, observed spread kept.

    This isolates the fee dimension so the gross edge (0bps row) is visible;
    the realistic combined-cost scenario is the main backtest.
    """
    rows = []
    for fee in taker_fees_bps:
        c = CostConfig(taker_fee_bps=float(fee), maker_fee_bps=float(fee) * 0.4,
                       slippage_bps=0.0, use_spread=cost.use_spread,
                       fee_scenario="user")
        out = run_cost_backtest(pred_frame, c, risk, profile, **engine_kwargs)
        r = out["report"]
        rows.append({
            "taker_fee_bps": float(fee),
            "n_trades": r["n_trades"],
            "n_round_trips": r["n_round_trips"],
            "total_net_return_pct": r["total_net_return_pct"],
            "total_costs": r["total_costs"],
            "max_drawdown_pct": r["max_drawdown_pct"],
        })
    return pd.DataFrame(rows)
