"""Streaming archive ingestion: Binance daily ZIPs -> per-second Parquet buckets.

Memory-bounded: CSVs are read in chunks inside the zip; nothing holds a full
day of raw events. Timestamp semantics: HISTORICAL research uses EXCHANGE
timestamps (futures bookTicker `transaction_time`; trades `transact_time`);
archives contain no local receipt time (documented limitation). Live
collection separately records receipt time for latency monitoring.

Integrity per day:
- book events applied in exchange order; update_id strictly monotonic
- duplicate ids counted and skipped; out-of-order ids counted and skipped
- crossed/invalid quotes counted, never applied to book state
- OFI uses the standard Cont-Kukanov-Stoikov best-level convention:
  bid-up -> +bq_new, bid-down -> -bq_old, ask-down -> +aq_old, ask-up -> -aq_new
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK_ROWS = 2_000_000

BOOK_COLS = ["update_id", "best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty",
             "transaction_time", "event_time"]
TRADE_COLS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
              "transact_time", "is_buyer_maker"]
SPOT_TRADE_COLS = ["id", "price", "qty", "first", "last", "time", "is_buyer_maker", "ignore"]
FUNDING_COLS = ["calc_time", "funding_interval_hours", "last_funding_rate"]

BOOK_BUCKET_COLS = ["second", "bid", "bid_qty", "ask", "ask_qty", "ofi", "n_updates",
                    "last_update_id", "crossed_count", "dup_count", "out_of_order_count"]
TRADE_BUCKET_COLS = ["second", "n_trades", "buy_vol", "sell_vol", "vol", "notional",
                     "last_px", "high_px", "low_px"]


class _PrependedReader:
    """File-like splicing a peeked first line back in front of the stream."""

    def __init__(self, first_line: str, text):
        self._first = first_line
        self._text = text

    def read(self, size=-1):
        if size is None or size < 0:
            head, self._first = self._first, ""
            return head + self._text.read()
        if not self._first:
            return self._text.read(size)
        take = self._first[:size]
        self._first = self._first[size:]
        if len(take) < size:
            return take + self._text.read(size - len(take))
        return take


def _first_field_is_numeric(first_line: str) -> bool:
    token = first_line.split(",", 1)[0].strip()
    try:
        float(token)
        return True
    except ValueError:
        return False


def _zip_csv_chunks(path: Path, chunk_rows: int = CHUNK_ROWS):
    """Yield DataFrame chunks from the single CSV inside a Binance daily zip.

    Handles BOTH schemas transparently: futures archives carry a header line;
    spot aggTrades archives are headerless (first line is data). The first
    line is peeked and spliced back, so no data row is ever consumed as a
    header.
    """
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError(f"no CSV inside archive {path.name}")
        with zf.open(names[0]) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8", newline="")
            first_line = text.readline()
            headerless = _first_field_is_numeric(first_line)
            reader = _PrependedReader(first_line, text)
            yield from pd.read_csv(reader, chunksize=chunk_rows,
                                   header=None if headerless else 0,
                                   low_memory=False)


def _is_headerless(chunk: pd.DataFrame) -> bool:
    """Spot aggTrades CSVs have no header: first column is an integer trade id."""
    try:
        return str(chunk.columns[0]).replace(".0", "").isdigit()
    except Exception:  # noqa: BLE001
        return False


def ingest_trades_zip(path: Path, market: str) -> pd.DataFrame:
    """Aggregate trades -> one row per second (buy/sell volume, count, vwap parts).

    `is_buyer_maker=true` means the aggressor sold (aggressive sell volume).
    """
    parts: list[pd.DataFrame] = []
    for chunk in _zip_csv_chunks(path):
        if _is_headerless(chunk):
            if len(chunk.columns) != 8:
                raise ValueError(f"unexpected headerless trade schema in {path.name}: {chunk.columns}")
            chunk = chunk.iloc[:, :7]
            chunk.columns = TRADE_COLS
        needed = ["transact_time", "price", "quantity", "is_buyer_maker"]
        if not set(needed).issubset(chunk.columns):
            raise ValueError(f"unexpected trade schema in {path.name}: {list(chunk.columns)}")
        df = chunk[needed].copy()
        df["transact_time"] = pd.to_numeric(df["transact_time"], errors="coerce")
        for c in ("price", "quantity"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["transact_time", "price", "quantity"])
        df = df[(df["transact_time"] > 0) & (df["price"] > 0) & (df["quantity"] >= 0)]
        if df.empty:
            continue
        df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).str.lower().isin(["true", "1", "yes"])
        sec = (df["transact_time"].astype("int64") // 1000).to_numpy()
        qty = df["quantity"].to_numpy(dtype=float)
        buy = np.where(~df["is_buyer_maker"].to_numpy(), qty, 0.0)
        sell = np.where(df["is_buyer_maker"].to_numpy(), qty, 0.0)
        px = df["price"].to_numpy(dtype=float)
        g = pd.DataFrame({
            "second": sec, "buy_vol": buy, "sell_vol": sell, "vol": qty,
            "notional": px * qty, "last_px": px,
        }).groupby("second", sort=True).agg(
            n_trades=("vol", "size"), buy_vol=("buy_vol", "sum"), sell_vol=("sell_vol", "sum"),
            vol=("vol", "sum"), notional=("notional", "sum"), last_px=("last_px", "last"),
        ).reset_index()
        parts.append(g)
    empty = pd.DataFrame(columns=TRADE_BUCKET_COLS)
    if not parts:
        return empty
    full = pd.concat(parts, ignore_index=True)
    agg = full.groupby("second", sort=True).agg(
        n_trades=("n_trades", "sum"), buy_vol=("buy_vol", "sum"), sell_vol=("sell_vol", "sum"),
        vol=("vol", "sum"), notional=("notional", "sum"), last_px=("last_px", "last"),
    ).reset_index()
    return agg


def ingest_bookticker_zip(path: Path) -> tuple[pd.DataFrame, dict]:
    """bookTicker -> per-second last quote, summed OFI, integrity counters.

    Vectorized event processing with exact monotonic-update_id semantics:
    an event is applied only if update_id > the last APPLIED update_id.
    """
    parts: list[pd.DataFrame] = []
    counter_parts: list[pd.DataFrame] = []
    carry = None  # (bid, bidq, ask, askq) of last applied quote across chunks
    last_applied_id = -1
    stats = {"dup_events": 0, "out_of_order": 0, "crossed_quotes": 0, "applied_events": 0}

    for chunk in _zip_csv_chunks(path):
        if list(chunk.columns) != BOOK_COLS:
            if len(chunk.columns) == 7:
                chunk.columns = BOOK_COLS
            else:
                raise ValueError(f"unexpected bookTicker schema in {path.name}: {list(chunk.columns)}")
        for col in BOOK_COLS:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk = chunk.dropna(subset=BOOK_COLS)
        if chunk.empty:
            continue
        sec = (chunk["transaction_time"].astype("int64") // 1000).to_numpy()
        uid = chunk["update_id"].to_numpy(dtype=np.int64)
        bid = chunk["best_bid_price"].to_numpy(dtype=float)
        bidq = chunk["best_bid_qty"].to_numpy(dtype=float)
        ask = chunk["best_ask_price"].to_numpy(dtype=float)
        askq = chunk["best_ask_qty"].to_numpy(dtype=float)

        prev_uid = np.concatenate(([last_applied_id], uid[:-1]))
        dup = uid == prev_uid
        ooo = uid < prev_uid
        crossed = (bid <= 0) | (ask <= 0) | (bid >= ask)
        applied = ~(dup | ooo | crossed)

        # Enforce strict monotonicity vs the running max of APPLIED ids.
        idx = np.flatnonzero(applied)
        if idx.size:
            cand = uid[idx]
            prev_max = np.maximum.accumulate(np.concatenate(([last_applied_id], cand[:-1])))
            viol = cand <= prev_max
            if viol.any():
                demoted = idx[viol]
                applied[demoted] = False
                dup[idx[viol & (cand == prev_max)]] = True
                ooo[idx[viol & (cand < prev_max)]] = True

        n_applied = int(applied.sum())
        stats["dup_events"] += int(dup.sum())
        stats["out_of_order"] += int(ooo.sum())
        stats["crossed_quotes"] += int(crossed.sum())
        stats["applied_events"] += n_applied

        if n_applied:
            b, bq, a, aq = bid[applied], bidq[applied], ask[applied], askq[applied]
            if carry is None:
                ofi = np.zeros(n_applied)
                ofi[1:] = _ofi_vector(b, bq, a, aq)[1:]
            else:
                ofi = _ofi_vector_with_carry(b, bq, a, aq, carry)
            parts.append(pd.DataFrame({
                "second": sec[applied], "update_id": uid[applied],
                "bid": b, "bid_qty": bq, "ask": a, "ask_qty": aq, "ofi": ofi,
            }))
            carry = (b[-1], bq[-1], a[-1], aq[-1])
            last_applied_id = int(uid[applied][-1])

        counters = {"dup": dup.astype(int), "ooo": ooo.astype(int), "crossed": crossed.astype(int)}
        if any(int(c.sum()) > 0 for c in counters.values()):
            cdf = pd.DataFrame({"second": sec, **counters})
            counter_parts.append(cdf.groupby("second", sort=True).sum().reset_index())

    if not parts:
        empty = pd.DataFrame(columns=BOOK_BUCKET_COLS)
        return empty, stats
    books = pd.concat(parts, ignore_index=True)
    agg = books.groupby("second", sort=True).agg(
        bid=("bid", "last"), bid_qty=("bid_qty", "last"),
        ask=("ask", "last"), ask_qty=("ask_qty", "last"),
        ofi=("ofi", "sum"), n_updates=("bid", "size"),
        last_update_id=("update_id", "last"),
    ).reset_index()
    if counter_parts:
        csum = pd.concat(counter_parts, ignore_index=True).groupby("second", sort=True).sum().reset_index()
        agg = agg.merge(csum, on="second", how="left").fillna(
            {"dup": 0, "ooo": 0, "crossed": 0})
        agg = agg.rename(columns={"dup": "dup_count", "ooo": "out_of_order_count",
                                  "crossed": "crossed_count"})
    else:
        agg["dup_count"] = 0; agg["out_of_order_count"] = 0; agg["crossed_count"] = 0
    return agg, stats


def _ofi_vector(b, bq, a, aq):
    """OFI between consecutive applied quotes within one array (first = 0).

    Standard best-level convention (positive = bid-side pressure):
      bid-up -> +bq_new; bid-same -> bq_new - bq_old; bid-down -> -bq_old
      ask-down (offer lifted) -> +aq_old; ask-same -> -(aq_new - aq_old); ask-up -> -aq_new
    """
    n = len(b)
    if n < 2:
        return np.zeros(n)
    pb, pbq, pa, paq = b[:-1], bq[:-1], a[:-1], aq[:-1]
    cb, cbq, ca, caq = b[1:], bq[1:], a[1:], aq[1:]
    bid_c = np.where(cb > pb, cbq, np.where(cb == pb, cbq - pbq, -pbq))
    ask_down, ask_up, ask_same = ca < pa, ca > pa, ca == pa
    ask_c = np.where(ask_down, paq, np.where(ask_up, -caq, -(caq - paq)))
    out = np.zeros(n)
    out[1:] = bid_c + ask_c
    return out


def _ofi_vector_with_carry(b, bq, a, aq, carry):
    """OFI including the carried quote from the previous chunk."""
    inner = _ofi_vector(b, bq, a, aq)
    cb, cbq, ca, caq = carry
    first_bid_c = b[0] - 0.0  # placeholder
    if b[0] > cb:
        first = bq[0]
    elif b[0] == cb:
        first = bq[0] - cbq
    else:
        first = -cbq
    if a[0] < ca:
        first += caq
    elif a[0] > ca:
        first -= aq[0]
    else:
        first -= aq[0] - caq
    inner[0] = first
    return inner


def ingest_funding_zip(path: Path) -> pd.DataFrame:
    """Funding events -> sparse frame (second, funding_rate, interval_hours)."""
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError(f"no CSV inside archive {path.name}")
        with zf.open(names[0]) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8")
            df = pd.read_csv(text, header=None, names=FUNDING_COLS, skip_blank_lines=True)
    df = df[pd.to_numeric(df["calc_time"], errors="coerce").notna()]
    df["calc_time"] = pd.to_numeric(df["calc_time"]).astype("int64")
    df["last_funding_rate"] = pd.to_numeric(df["last_funding_rate"], errors="coerce")
    df["funding_interval_hours"] = pd.to_numeric(df["funding_interval_hours"], errors="coerce")
    df = df[df["calc_time"] > 0]
    if df.empty:
        return pd.DataFrame(columns=["second", "funding_rate", "funding_interval_hours"])
    out = pd.DataFrame({
        "second": df["calc_time"].to_numpy() // 1000,
        "funding_rate": df["last_funding_rate"].to_numpy(dtype=float),
        "funding_interval_hours": df["funding_interval_hours"].to_numpy(dtype=float),
    })
    return out.groupby("second", sort=True).last().reset_index()


def processed_path(processed_dir: str | Path, market: str, symbol: str, dtype: str, day: str) -> Path:
    return Path(processed_dir) / market / symbol.upper() / dtype / f"{day}.parquet"


def ingest_archive(market: str, symbol: str, dtype: str, day: str,
                   raw_dir: str | Path, processed_dir: str | Path, rebuild: bool = False) -> dict:
    """Ingest one daily archive into a per-second Parquet bucket (idempotent)."""
    from .downloader import raw_path_for

    raw_file = raw_path_for(raw_dir, market, symbol, dtype) / f"{symbol.upper()}-{dtype}-{day}.zip"
    if not raw_file.exists():
        raise FileNotFoundError(f"archive not downloaded: {raw_file}")
    out = processed_path(processed_dir, market, symbol, dtype, day)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not rebuild:
        return {"day": day, "status": "exists", "path": str(out), "rows": None, "stats": None}

    if dtype == "bookTicker":
        frame, stats = ingest_bookticker_zip(raw_file)
        frame.to_parquet(out, index=False)
        return {"day": day, "status": "ok", "path": str(out), "rows": len(frame), "stats": stats}
    if dtype in ("aggTrades", "trades"):
        frame = ingest_trades_zip(raw_file, market)
        frame.to_parquet(out, index=False)
        return {"day": day, "status": "ok", "path": str(out), "rows": len(frame),
                "stats": {"seconds_with_trades": len(frame)}}
    if dtype == "fundingRate":
        frame = ingest_funding_zip(raw_file)
        frame.to_parquet(out, index=False)
        return {"day": day, "status": "ok", "path": str(out), "rows": len(frame),
                "stats": {"events": len(frame)}}
    raise ValueError(f"ingest for dtype {dtype} is not implemented")


def ingest_funding_monthly(market: str, symbol: str, month: str, days: list[str],
                           raw_dir: str | Path, processed_dir: str | Path,
                           rebuild: bool = False) -> dict:
    """Ingest a MONTHLY fundingRate archive into per-day buckets.

    The public archive publishes fundingRate monthly-only (verified: daily
    files return 404). This splits the month's events into the requested days.
    """
    from .downloader import raw_path_for

    raw_file = raw_path_for(raw_dir, market, symbol, "fundingRate") / \
        f"{symbol.upper()}-fundingRate-{month}.zip"
    if not raw_file.exists():
        raise FileNotFoundError(f"monthly funding archive not downloaded: {raw_file}")
    frame = ingest_funding_zip(raw_file)
    total = 0
    written = []
    for day in days:
        d0 = int(pd.Timestamp(day, tz="UTC").timestamp())
        d1 = d0 + 86400
        day_frame = frame[(frame["second"] >= d0) & (frame["second"] < d1)]
        out = processed_path(processed_dir, market, symbol, "fundingRate", day)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and not rebuild:
            continue
        day_frame.to_parquet(out, index=False)
        total += len(day_frame)
        written.append(day)
    return {"month": month, "days_written": written, "events": total}
