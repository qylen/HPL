"""Per-dataset data-quality reports: coverage, gaps, duplicates, staleness."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .features import GAP_THRESHOLD_SECONDS, STALE_MAX_SECONDS


def _gap_runs(seconds_with_events: np.ndarray, grid: np.ndarray, min_run: int = 5):
    """Find consecutive missing-second runs of length >= min_run on the day grid."""
    have = np.isin(grid, seconds_with_events)
    runs = []
    start = None
    for i, h in enumerate(have):
        if not h and start is None:
            start = grid[i]
        elif h and start is not None:
            end = grid[i - 1]
            if i - int(np.searchsorted(grid, start)) >= min_run:
                runs.append({"start_second": int(start), "end_second": int(end),
                             "length_seconds": int(end - start + 1)})
            start = None
    if start is not None:
        length = int(len(grid) - np.searchsorted(grid, start))
        if length >= min_run:
            runs.append({"start_second": int(start), "end_second": int(grid[-1]),
                         "length_seconds": length})
    return runs


def quality_report(feature_frame: pd.DataFrame, ingest_stats: dict[str, int] | None = None,
                   gap_threshold: float = GAP_THRESHOLD_SECONDS,
                   stale_max: float = STALE_MAX_SECONDS) -> dict[str, Any]:
    """Compute the data-quality report from a built feature frame.

    Semantics: coverage counts seconds with an actual event (book or trade);
    the feature frame covers the FULL second grid, so 'usable' excludes
    data_gap/stale seconds explicitly rather than by row absence.
    """
    df = feature_frame
    n = len(df)
    if n == 0:
        return {"rows": 0, "error": "empty dataset"}
    has_book = "n_updates" in df.columns and df["n_updates"].notna().any()
    book_seconds = int((df["n_updates"].fillna(0) > 0).sum()) if has_book else 0
    trade_seconds = int((df["n_trades"].fillna(0) > 0).sum()) if "n_trades" in df.columns else 0
    event_masks = [df[c].fillna(0) > 0 for c in ("n_updates", "n_trades") if c in df.columns]
    event_seconds = int(pd.concat(event_masks, axis=1).any(axis=1).sum()) if event_masks else 0
    gap_seconds = int(df["data_gap"].fillna(True).sum()) if "data_gap" in df.columns else n
    valid_rows = int(df["valid"].fillna(False).sum()) if "valid" in df.columns else 0
    labeled_valid = int((df["valid"].fillna(False) & df["target_return"].notna()).sum()) \
        if "valid" in df.columns else 0
    invalid_spread = 0
    if has_book and "spread_bps" in df.columns:
        sp = df["spread_bps"]
        invalid_spread = int(((sp <= 0) | sp.isna()).sum())
    report: dict[str, Any] = {
        "rows_total": n,
        "book_seconds": book_seconds,
        "trade_seconds": trade_seconds,
        "event_seconds": event_seconds,
        "coverage_pct": round(100.0 * event_seconds / n, 3),
        "data_gap_seconds": gap_seconds,
        "stale_or_invalid_seconds": n - valid_rows,
        "usable_rows": valid_rows,
        "usable_labeled_rows": labeled_valid,
        "usable_pct": round(100.0 * valid_rows / n, 3),
        "invalid_spread_rows": invalid_spread,
        "timestamp_semantics": (
            "exchange timestamps (archive transaction/transact time); no local "
            "receipt time exists in historical archives"),
    }
    if ingest_stats:
        report["integrity"] = {
            "duplicate_events": int(ingest_stats.get("dup_events", 0)),
            "out_of_order_events": int(ingest_stats.get("out_of_order", 0)),
            "crossed_quotes": int(ingest_stats.get("crossed_quotes", 0)),
            "applied_events": int(ingest_stats.get("applied_events", 0)),
        }
    # gap runs on the second grid. When book columns exist, gaps are defined
    # by missing BOOK events (a book outage with trades still flowing is a gap
    # for any book-based feature set); otherwise by missing trades.
    grid = df["second"].to_numpy(dtype=np.int64)
    if "n_updates" in df.columns:
        gap_source_mask = (df["n_updates"].fillna(0) > 0).to_numpy()
    elif "n_trades" in df.columns:
        gap_source_mask = (df["n_trades"].fillna(0) > 0).to_numpy()
    elif event_masks:
        gap_source_mask = pd.concat(event_masks, axis=1).any(axis=1).to_numpy()
    else:
        gap_source_mask = np.zeros(n, dtype=bool)
    seconds_with_events = grid[gap_source_mask]
    runs = _gap_runs(seconds_with_events, grid, min_run=5)
    report["gap_runs_ge_5s"] = len(runs)
    report["gap_run_examples"] = runs[:10]
    if "quote_age_seconds" in df.columns:
        ages = df["quote_age_seconds"].replace([np.inf], np.nan).dropna()
        if len(ages):
            report["quote_age_seconds_p50"] = float(ages.quantile(0.5))
            report["quote_age_seconds_p99"] = float(ages.quantile(0.99))
    return report


def quality_markdown(report: dict[str, Any], title: str = "Data quality report") -> str:
    lines = [f"# {title}", ""]
    for k, v in report.items():
        if isinstance(v, dict):
            lines.append(f"- **{k}**: " + ", ".join(f"{a}={b}" for a, b in v.items()))
        elif isinstance(v, list):
            lines.append(f"- **{k}**: {len(v)} item(s)" if v else f"- **{k}**: 0")
        else:
            lines.append(f"- **{k}**: {v}")
    return "\n".join(lines) + "\n"
