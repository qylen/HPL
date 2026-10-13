"""Deterministic synthetic market-data generators.

Two levels:
1. Frame level (`generate_synthetic_dataset`): fast feature CSVs for the legacy
   smoke path. The planted signal is PREDICTIVE: features at second t drive
   the return over t -> t+1 (the original generator planted a contemporaneous
   relation, which validated the wrong alignment; fixed here).
2. Event level (`write_synthetic_archive`): Binance-format daily ZIPs
   (bookTicker / aggTrades / fundingRate) with a planted or noise-only
   relationship, plus corruption scenarios (duplicates, out-of-order ids,
   crossed quotes, missing seconds). Used by the end-to-end tests.

None of this is market evidence; it exists to verify pipeline mechanics.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS


def generate_synthetic_dataset(output_path, rows: int = 20_000, seed: int = 13,
                               planted: bool = True, start: str = "2025-01-01T00:00:00Z") -> pd.DataFrame:
    """Feature table with a planted PREDICTIVE relationship (or pure noise)."""
    if rows < 500:
        raise ValueError("rows must be at least 500")
    rng = np.random.default_rng(seed)
    timestamp = pd.date_range(pd.Timestamp(start), periods=rows, freq="s")
    ofi_norm = rng.normal(0, 0.25, rows)
    trade_imbalance = np.tanh(ofi_norm * 1.8 + rng.normal(0, 0.7, rows))
    spread = np.clip(rng.lognormal(mean=0.0, sigma=0.3, size=rows), 0.2, 5.0)
    depth = rng.lognormal(mean=1.0, sigma=0.5, size=rows)
    volatility = np.abs(rng.normal(0.00015, 0.00007, rows))
    noise = rng.normal(0, 0.00035, rows)
    alpha = 0.00020 if planted else 0.0
    beta = 0.00008 if planted else 0.0
    # y[i] = return over second (i-1 -> i), driven by flow during second i-1.
    y = np.empty(rows)
    y[0] = noise[0]
    y[1:] = alpha * ofi_norm[:-1] + beta * trade_imbalance[:-1] + noise[1:]
    mid = 100_000.0 * np.exp(np.cumsum(y))
    # Label at row i = return over (i -> i+1) = y[i+1]; last row unlabeled.
    target = np.concatenate([y[1:], [np.nan]])
    frame = pd.DataFrame({
        "timestamp": timestamp,
        "target_return": target,
        "ofi": ofi_norm * depth,
        "ofi_norm": ofi_norm,
        "ofi_norm_5s": pd.Series(ofi_norm).rolling(5, min_periods=1).mean().to_numpy(),
        "trade_imbalance": trade_imbalance,
        "trade_imbalance_5s": pd.Series(trade_imbalance).rolling(5, min_periods=1).mean().to_numpy(),
        "trade_volume_log": np.log1p(rng.lognormal(0, 0.6, rows)),
        "trade_volume_5s_log": np.log1p(rng.lognormal(1, 0.5, rows)),
        "buy_volume_log": np.log1p(rng.lognormal(-0.2, 0.7, rows)),
        "sell_volume_log": np.log1p(rng.lognormal(-0.2, 0.7, rows)),
        "trade_count_log": np.log1p(rng.poisson(3, rows)),
        "spread_bps": spread,
        "depth_total": depth,
        "depth_imbalance": np.tanh(ofi_norm + rng.normal(0, 0.5, rows)),
        "mid_return_1s": np.concatenate([[0.0], y[:-1]]),
        "volatility_5s": pd.Series(volatility).rolling(5, min_periods=3).std().fillna(0).to_numpy(),
        "volatility_30s": pd.Series(volatility).rolling(30, min_periods=10).std().fillna(0).to_numpy(),
        "quote_update_count_log": np.log1p(rng.poisson(4, rows)),
        "quote_age_ms": rng.uniform(0, 1000, rows),
        "time_sin": np.sin(2 * np.pi * (np.arange(rows) % 86400) / 86400),
        "time_cos": np.cos(2 * np.pi * (np.arange(rows) % 86400) / 86400),
        "mid": mid,
    })
    wanted = list(dict.fromkeys(["timestamp", "mid", "target_return", "spread_bps",
                                 "quote_age_ms", *FEATURE_COLUMNS]))
    frame = frame[wanted]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


# ----------------------------------------------------------------------------
# Event-level generator: Binance-format daily archives
# ----------------------------------------------------------------------------

BOOK_HEADER = "update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time\n"
TRADE_HEADER = "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
FUNDING_HEADER = "calc_time,funding_interval_hours,last_funding_rate\n"


def _book_rows(second_start: int, seconds: int, ofi: np.ndarray, mids: np.ndarray,
               spread: np.ndarray, rng, events_per_second: int = 8) -> list[str]:
    """Book event rows; the LAST event of each second sets quantity imbalance
    so that per-second OFI tracks `ofi[s]` (same-price quantity changes)."""
    rows = []
    uid = 1000
    base_qty = 5.0
    prev_bidq, prev_askq = base_qty, base_qty
    for s in range(seconds):
        t0 = (second_start + s) * 1000
        bid = mids[s] - spread[s] / 2.0
        ask = mids[s] + spread[s] / 2.0
        # quantity imbalance target for this second (scaled OFI)
        k = ofi[s] * 2.0
        bidq_t = base_qty + k / 2.0
        askq_t = base_qty - k / 2.0
        bidq_t = max(0.1, bidq_t)
        askq_t = max(0.1, askq_t)
        for e in range(events_per_second):
            ts = t0 + int(1000 * (e + 1) / (events_per_second + 1))
            if e == events_per_second - 1:
                bq, aq = bidq_t, askq_t
            else:
                bq = max(0.1, prev_bidq + rng.normal(0, 0.2))
                aq = max(0.1, prev_askq + rng.normal(0, 0.2))
            rows.append(f"{uid},{bid:.2f},{bq:.4f},{ask:.2f},{aq:.4f},{ts},{ts}")
            uid += 1
            prev_bidq, prev_askq = bq, aq
    return rows


def _trade_rows(second_start: int, seconds: int, ofi: np.ndarray, mids: np.ndarray, rng,
                trades_per_second: int = 5) -> list[str]:
    rows = []
    tid = 1
    for s in range(seconds):
        t0 = (second_start + s) * 1000
        n = max(1, int(trades_per_second + rng.integers(-2, 3)))
        for e in range(n):
            ts = t0 + int(1000 * (e + 1) / (n + 1))
            px = mids[s] * float(rng.normal(1.0, 0.0002))
            qty = float(abs(rng.normal(0.05, 0.02)))
            buy = rng.random() < (0.5 + 0.35 * float(np.clip(ofi[s], -1, 1)))
            rows.append(f"{tid},{px:.2f},{qty:.6f},{tid},{tid},{ts},{'true' if not buy else 'false'}")
            tid += 1
    return rows


def write_synthetic_archive(directory, market: str = "futures", symbol: str = "SYNUSD",
                            day: str = "2025-06-01", seconds: int = 3600,
                            planted: bool = True, seed: int = 13,
                            gap_seconds: int = 0, duplicate_events: int = 0,
                            out_of_order_events: int = 0, crossed_quotes: int = 0,
                            with_funding: bool = True) -> dict:
    """Write Binance-format daily ZIP archives for the synthetic market."""
    import datetime as dt

    rng = np.random.default_rng(seed)
    d = dt.date.fromisoformat(day)
    second_start = int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp())
    directory = Path(directory)
    sym = symbol.upper()
    dtype_dir = directory / market / sym
    for dtype in ("bookTicker", "aggTrades", "fundingRate"):
        (dtype_dir / dtype).mkdir(parents=True, exist_ok=True)

    # Mid path: random walk; if planted, next return driven by current OFI.
    ofi = np.tanh(rng.normal(0, 0.5, seconds))
    noise = rng.normal(0, 0.00012, seconds)
    alpha = 0.00025 if planted else 0.0
    r = np.empty(seconds)
    r[0] = noise[0]
    r[1:] = alpha * ofi[:-1] + noise[1:]
    mids = 50_000.0 * np.exp(np.cumsum(r))
    spread = np.abs(rng.normal(1.0, 0.2, seconds))

    keep = np.ones(seconds, dtype=bool)
    if gap_seconds > 0:
        drop = rng.choice(np.arange(10, seconds - 10), size=gap_seconds, replace=False)
        keep[drop] = False

    book_rows = _book_rows(second_start, seconds, ofi, mids, spread, rng)
    trade_rows = _trade_rows(second_start, seconds, ofi, mids, rng)

    # Corruptions (applied to the book event stream)
    if duplicate_events > 0 and book_rows:
        idxs = rng.choice(len(book_rows), size=min(duplicate_events, len(book_rows)), replace=False)
        for i in sorted(idxs, reverse=True):
            book_rows.insert(int(i), book_rows[int(i)])
    if out_of_order_events > 0 and len(book_rows) > 2:
        idxs = rng.choice(len(book_rows) - 1, size=min(out_of_order_events, len(book_rows) - 1), replace=False)
        for i in sorted(idxs, reverse=True):
            book_rows[int(i)], book_rows[int(i) + 1] = book_rows[int(i) + 1], book_rows[int(i)]
    if crossed_quotes > 0 and book_rows:
        idxs = rng.choice(len(book_rows), size=min(crossed_quotes, len(book_rows)), replace=False)
        for i in idxs:
            parts = book_rows[int(i)].split(",")
            parts[1], parts[3] = parts[3], parts[1]  # bid px >= ask px -> crossed book
            book_rows[int(i)] = ",".join(parts)

    # Drop seconds with no events (gaps): remove rows whose second is dropped
    def _drop_seconds(rows: list[str], ts_col: int) -> list[str]:
        out = []
        for row in rows:
            ts = int(row.split(",")[ts_col])
            s = ts // 1000 - second_start
            if 0 <= s < seconds and keep[s]:
                out.append(row)
        return out

    book_rows = _drop_seconds(book_rows, 5)
    trade_rows = _drop_seconds(trade_rows, 5)

    book_csv = BOOK_HEADER + "\n".join(book_rows) + ("\n" if book_rows else "")
    trade_csv = TRADE_HEADER + "\n".join(trade_rows) + ("\n" if trade_rows else "")
    funding_csv = ""
    if with_funding and market == "futures":
        fts = (second_start + 3600) * 1000  # one event at 00:15 for a short test day
        funding_csv = FUNDING_HEADER + f"{fts},8,0.00010\n"

    paths = {}
    for dtype, content in (("bookTicker", book_csv), ("aggTrades", trade_csv), ("fundingRate", funding_csv)):
        if not content:
            continue
        zpath = dtype_dir / dtype / f"{sym}-{dtype}-{day}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{sym}-{dtype}-{day}.csv", content)
        paths[dtype] = str(zpath)
    return {"paths": paths, "seconds": seconds, "planted": planted, "seed": seed,
            "gap_seconds": gap_seconds, "mids_first": float(mids[0]),
            "alpha": alpha}
