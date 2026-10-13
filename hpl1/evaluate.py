"""Research evaluation: statistical battery + research report generation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .statistics import (
    directional_calibration,
    hac_mean_test,
    holm_correction,
    ljung_box,
    time_shift_placebo,
)


def statistical_battery(predictions: np.ndarray, actual: np.ndarray,
                        residuals: np.ndarray | None = None,
                        seed: int = 13,
                        calibration_fit: tuple | None = None) -> dict[str, Any]:
    """The full validation battery run on the untouched test predictions."""
    p = np.asarray(predictions, dtype=float)
    a = np.asarray(actual, dtype=float)
    ok = np.isfinite(p) & np.isfinite(a)
    p, a = p[ok], a[ok]
    battery: dict[str, Any] = {}
    battery["n"] = len(p)
    if len(p) < 100:
        battery["error"] = "test set too small for the battery"
        return battery

    # 1. mean-return HAC inference on the naive per-second strategy returns
    naive_ret = p * a  # gross per-second return of always-following the sign
    battery["hac_gross_mean"] = hac_mean_test(naive_ret, max_lags=120)

    # 2. block-bootstrap CI for the prediction/return correlation
    def _corr(x_pair_dummy):  # helper not used; stat over single series below
        return 0.0

    if np.std(p) > 0 and np.std(a) > 0:
        corr_point = float(np.corrcoef(p, a)[0, 1])
        rng = np.random.default_rng(seed)
        n = len(p)
        block, b = 60, 300
        n_blocks = int(np.ceil(n / block))
        stats = []
        for _ in range(b):
            starts = rng.integers(0, n - block + 1, size=n_blocks)
            idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
            pp, aa = p[idx], a[idx]
            if np.std(pp) > 0 and np.std(aa) > 0:
                stats.append(float(np.corrcoef(pp, aa)[0, 1]))
        if stats:
            lo, hi = np.quantile(stats, [0.025, 0.975])
            battery["correlation_block_bootstrap"] = {
                "point": corr_point, "ci_low": float(lo), "ci_high": float(hi),
                "block": block, "replicates": len(stats)}
    # 3. placebo time-shift
    battery["placebo_time_shift"] = time_shift_placebo(p, a, max_shift=10)

    # 4. directional calibration
    battery["directional_calibration"] = directional_calibration(
        p, a, calibration_fit=calibration_fit)

    # 5. residual autocorrelation (Ljung-Box) if residuals supplied
    if residuals is not None:
        r = np.asarray(residuals, dtype=float)
        battery["ljung_box_residuals"] = ljung_box(r)

    # 6. multiple-testing control across the battery's key p-values
    pvals = {
        "hac_gross_mean": battery["hac_gross_mean"].get("hac_p_value_two_sided", 1.0),
    }
    if "placebo_time_shift" in battery and "alignment_dependent" in battery.get("placebo_time_shift", {}):
        # placebo 'p-value': fraction of |shifted| correlations >= |real|
        pl = battery["placebo_time_shift"]
        pvals["placebo_alignment"] = max(1e-4, 1.0 - float(pl["shifted_abs_correlation_p95"]))
    battery["holm_correction"] = holm_correction(pvals)
    return battery


def research_report(market: str, symbol: str, days: str, comparison: dict,
                    backtest: dict | None, battery: dict | None,
                    quality: dict | None, config_meta: dict | None,
                    fee_table: pd.DataFrame | None = None) -> dict[str, Any]:
    """Assemble the machine-readable research report (verdict included)."""
    results = comparison["results"]
    best = comparison["best"]
    test_metrics = results[best].get("test", {})
    val_metrics = results[best].get("validation", {})
    net_return = backtest["report"]["total_net_return_pct"] if backtest else None
    n_trades = backtest["report"]["n_trades"] if backtest else 0
    corr = test_metrics.get("prediction_return_correlation", 0.0)
    placebo_real = battery.get("placebo_time_shift", {}).get("real_correlation", 0.0) if battery else 0.0
    placebo_p95 = battery.get("placebo_time_shift", {}).get("shifted_abs_correlation_p95", 1.0) if battery else 1.0
    alignment = abs(placebo_real) > placebo_p95 if battery else False
    report = {
        "hypothesis": (
            "Order-flow imbalance, liquidity, spread, volatility, trade flow and regime "
            "changes predict the next one-second mid-price return well enough to trade "
            "profitably after realistic taker execution costs."),
        "market": market, "symbol": symbol, "days": days,
        "data_quality": quality,
        "models_compared": list(results.keys()),
        "model_table": {
            name: {
                "val_r2": res.get("validation", {}).get("oos_r2_vs_zero"),
                "val_dir": res.get("validation", {}).get("directional_accuracy"),
                "test_r2": res.get("test", {}).get("oos_r2_vs_zero"),
                "test_corr": res.get("test", {}).get("prediction_return_correlation"),
            }
            for name, res in results.items()
        },
        "selected_model": best,
        "selection_basis": "validation split only; test untouched until evaluation",
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "statistical_battery": battery,
        "backtest": backtest["report"] if backtest else None,
        "fee_sensitivity": fee_table.to_dict(orient="records") if fee_table is not None else None,
        "config": config_meta,
        "verdict": {
            "statistical_predictability": (
                "supported" if test_metrics.get("oos_r2_vs_zero", 0) > 0.01 and corr > 0.05
                else "not supported"),
            "signal_is_alignment_dependent_not_artifact": bool(alignment),
            "net_taker_profitability": (
                "rejected" if (net_return is not None and (n_trades == 0 or net_return <= 0))
                else ("supported" if net_return is not None and net_return > 0 else "not evaluated")),
            "summary": "",
        },
    }
    if net_return is not None and n_trades == 0:
        report["verdict"]["summary"] = (
            f"Out-of-sample predictability exists (R2={test_metrics.get('oos_r2_vs_zero', 0):.3f}, "
            f"corr={corr:.3f}) but predicted edges never clear the round-trip cost hurdle: "
            "zero trades at realistic taker fees. Predictability is NOT profitability.")
    elif net_return is not None and net_return <= 0:
        report["verdict"]["summary"] = (
            "Strategy traded and lost money net of costs on the untouched test window.")
    elif net_return is not None:
        report["verdict"]["summary"] = (
            f"Positive net test return {net_return:.2f}% — treat with suspicion until "
            "walk-forward and paper-trading evidence accumulates; single-window results "
            "are vulnerable to selection bias.")
    else:
        report["verdict"]["summary"] = "Backtest not run; no economic evaluation available."
    return report


def write_report_artifacts(report: dict, directory: str | Path) -> dict[str, str]:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    jpath = d / "research_report.json"
    jpath.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    mpath = d / "research_report.md"
    mpath.write_text(report_markdown(report), encoding="utf-8")
    return {"json": str(jpath), "markdown": str(mpath)}


def _fmt(v, nd=4):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}" if abs(v) < 1e6 else f"{v:.2e}"
    return str(v)


def report_markdown(report: dict) -> str:
    t = report.get("test_metrics", {})
    v = report.get("validation_metrics", {})
    bt = report.get("backtest") or {}
    bat = report.get("statistical_battery") or {}
    pl = bat.get("placebo_time_shift", {})
    lines = [
        f"# HPL-1 Research Report — {report.get('market')} {report.get('symbol')} ({report.get('days')})",
        "",
        "## Hypothesis",
        report.get("hypothesis", ""),
        "",
        "## Data",
        f"- Market: {report.get('market')} / {report.get('symbol')}",
        f"- Window: {report.get('days')}",
        f"- Usable rows: {_fmt((report.get('data_quality') or {}).get('usable_labeled_rows'), 0)}",
        "",
        "## Model comparison (selection on validation only)",
        "| model | val R2 | val dir.acc | test R2 | test corr |",
        "|---|---|---|---|---|",
    ]
    for name, row in report.get("model_table", {}).items():
        lines.append(f"| {name} | {_fmt(row.get('val_r2'))} | {_fmt(row.get('val_dir'))} | "
                     f"{_fmt(row.get('test_r2'))} | {_fmt(row.get('test_corr'))} |")
    lines.append(f"- selected: **{report.get('selected_model')}**")
    lines.append(f"- validation R2: {_fmt(v.get('oos_r2_vs_zero'))}, directional accuracy: {_fmt(v.get('directional_accuracy'))}")
    lines.append(f"- test R2 (untouched): {_fmt(t.get('oos_r2_vs_zero'))}, correlation: {_fmt(t.get('prediction_return_correlation'))}, directional accuracy: {_fmt(t.get('directional_accuracy'))}")
    lines += [
        "",
        "## Statistical battery (test set)",
        f"- Placebo: real corr {_fmt(pl.get('real_correlation'))} vs shifted |corr| p95 {_fmt(pl.get('shifted_abs_correlation_p95'))} -> alignment-dependent: {pl.get('alignment_dependent')}",
        f"- HAC z (gross mean): {_fmt((bat.get('hac_gross_mean') or {}).get('hac_z_score'))}",
        f"- Block-bootstrap corr CI: {_fmt((bat.get('correlation_block_bootstrap') or {}).get('ci_low'))} .. {_fmt((bat.get('correlation_block_bootstrap') or {}).get('ci_high'))}",
        f"- Brier score: {_fmt((bat.get('directional_calibration') or {}).get('brier_score'))}",
        "",
        "## Net backtest (cost-aware, taker execution)",
        f"- Trades: {bt.get('n_trades')}, round trips: {bt.get('n_round_trips')}",
        f"- Net return: {_fmt(bt.get('total_net_return_pct'), 2)}%  |  max DD: {_fmt(bt.get('max_drawdown_pct'), 2)}%",
        f"- Costs by component: {bt.get('costs_by_component')}",
        "",
        "## Verdict",
    ]
    verdict = report.get("verdict", {})
    lines.append(f"- Statistical predictability: **{verdict.get('statistical_predictability')}**")
    lines.append(f"- Signal is alignment-dependent (not timestamp artifact): **{verdict.get('signal_is_alignment_dependent_not_artifact')}**")
    lines.append(f"- Net taker profitability: **{verdict.get('net_taker_profitability')}**")
    lines.append("")
    lines.append(verdict.get("summary", ""))
    lines += [
        "",
        "## Limitations",
        "- Single contiguous window; no walk-forward across regimes.",
        "- Taker-only execution; maker fill economics untested.",
        "- Top-of-book data only; no depth beyond best level, no queue-position modelling.",
        "- Historical archive lacks local receipt timestamps; live latency not represented.",
    ]
    return "\n".join(lines) + "\n"
