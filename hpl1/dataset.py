"""Dataset assembly: per-second buckets across days -> one continuous frame.

The assembled frame is the single input to `features.build_features` for BOTH
historical research and the live paper engine (same code path). Raw archives
are never modified; processed Parquet buckets and dataset frames are derived
data kept separate under data/processed/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .binance_archive import iter_dates
from .ingest import processed_path

BOOK_DTYPES = ["bookTicker"]
TRADE_DTYPES = ["aggTrades", "trades"]
FUNDING_DTYPES = ["fundingRate"]


def day_grid(day: str) -> np.ndarray:
    """Full UTC second grid for one calendar day (86400 seconds)."""
    import datetime as dt

    d = dt.date.fromisoformat(day)
    start = int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp())
    return np.arange(start, start + 86400, dtype=np.int64)


def assemble_day(market: str, symbol: str, day: str, processed_dir: str | Path) -> pd.DataFrame:
    """Merge book/trade/funding buckets for one day onto the full second grid."""
    grid = day_grid(day)
    frame = pd.DataFrame({"second": grid})
    frame["ts"] = pd.to_datetime(frame["second"], unit="s", utc=True)

    for dtype in BOOK_DTYPES:
        p = processed_path(processed_dir, market, symbol, dtype, day)
        if p.exists():
            book = pd.read_parquet(p)
            book = book[book["second"].isin(set(grid.tolist()))]
            frame = frame.merge(book, on="second", how="left")
    for dtype in TRADE_DTYPES:
        p = processed_path(processed_dir, market, symbol, dtype, day)
        if p.exists():
            trades = pd.read_parquet(p)
            trades = trades[trades["second"].isin(set(grid.tolist()))]
            frame = frame.merge(trades, on="second", how="left")
    for dtype in FUNDING_DTYPES:
        p = processed_path(processed_dir, market, symbol, dtype, day)
        if p.exists():
            fund = pd.read_parquet(p)
            fund = fund[fund["second"].isin(set(grid.tolist()))]
            frame = frame.merge(fund, on="second", how="left")
    return frame


def dataset_path(processed_dir: str | Path, market: str, symbol: str, day: str) -> Path:
    return Path(processed_dir) / market / symbol.upper() / "dataset" / f"{day}.parquet"


def build_dataset_days(market: str, symbol: str, start: str, end: str,
                       processed_dir: str | Path, rebuild: bool = False) -> list[dict]:
    """Write one assembled dataset parquet per day. Returns per-day summaries."""
    summaries = []
    for day in iter_dates(start, end):
        d = day.isoformat()
        out = dataset_path(processed_dir, market, symbol, d)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and not rebuild:
            frame = pd.read_parquet(out)
            summaries.append({"day": d, "status": "exists", "rows": len(frame),
                              "book_seconds": int(frame["n_updates"].fillna(0).gt(0).sum())
                              if "n_updates" in frame.columns else 0})
            continue
        frame = assemble_day(market, symbol, d, processed_dir)
        frame.to_parquet(out, index=False)
        book_secs = int(frame["n_updates"].fillna(0).gt(0).sum()) if "n_updates" in frame.columns else 0
        trade_secs = int(frame["n_trades"].fillna(0).gt(0).sum()) if "n_trades" in frame.columns else 0
        summaries.append({"day": d, "status": "ok", "rows": len(frame),
                          "book_seconds": book_secs, "trade_seconds": trade_secs})
    return summaries


def load_dataset(market: str, symbol: str, start: str, end: str,
                 processed_dir: str | Path) -> pd.DataFrame:
    """Load assembled days into one continuous frame (book state ffilled across
    day boundaries via the feature builder; here we only concatenate raw)."""
    frames = []
    for day in iter_dates(start, end):
        p = dataset_path(processed_dir, market, symbol, day.isoformat())
        if not p.exists():
            raise FileNotFoundError(
                f"dataset day missing: {p} — run `hpl1 build --market {market}` first")
        day_frame = pd.read_parquet(p)
        if day_frame.empty:
            continue
        frames.append(day_frame)
    if not frames:
        raise FileNotFoundError(f"no dataset days found for {market} {symbol} {start}..{end}")
    full = pd.concat(frames, ignore_index=True)
    return full.sort_values("second", kind="stable").reset_index(drop=True)


def load_feature_frame(market: str, symbol: str, start: str, end: str,
                       processed_dir: str | Path) -> pd.DataFrame:
    """Convenience: dataset -> features for the market's supported feature set."""
    from .features import build_features
    from .markets import get_market

    profile = get_market(market)
    raw = load_dataset(market, symbol, start, end, processed_dir)
    feature_set = "full" if any(d in raw.columns for d in ("bid", "ask")) else "trade_only"
    if profile.market_type == "spot":
        feature_set = "trade_only"
    return build_features(raw, feature_set=feature_set)
