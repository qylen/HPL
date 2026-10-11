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
        "rows": int(len(df)),
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
