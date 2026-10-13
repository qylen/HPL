"""Binance public historical archive: URL patterns, S3 discovery, size estimation.

All knowledge about https://data.binance.vision/ lives here. Verified facts
(2026-10 probe, see docs/DOWNLOAD_RECORD.md):
- Spot publishes NO bookTicker/depth history (trade-flow only).
- Futures USD-M bookTicker daily archives END at 2024-03-30 (dataset discontinued).
- .CHECKSUM files hold sha256 digests.
"""
from __future__ import annotations

import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, timedelta

import requests

BASE_URL = "https://data.binance.vision/"
S3_ENDPOINT = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"

FUTURES_ONLY = {"fundingRate", "bookDepth"}
BOOK_TYPES = {"bookTicker", "bookDepth"}


@dataclass(frozen=True)
class ArchiveKey:
    key: str            # S3 object key, e.g. data/futures/um/daily/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2024-03-25.zip
    size: int
    market: str         # spot | futures
    dtype: str
    symbol: str
    period: str         # "daily" | "monthly"
    date_label: str     # YYYY-MM-DD or YYYY-MM

    @property
    def url(self) -> str:
        return f"{BASE_URL}{self.key}"

    @property
    def checksum_url(self) -> str:
        return f"{self.url}.CHECKSUM"


def archive_key(market: str, symbol: str, dtype: str, day: str | date, period: str = "daily") -> ArchiveKey:
    """Build the canonical archive key for a market/symbol/dtype/date."""
    from .markets import get_market

    profile = get_market(market)
    symbol = symbol.upper()
    if profile.market_type == "spot" and dtype in FUTURES_ONLY:
        raise ValueError(f"{dtype} is a futures-only dataset; spot publishes none of it")
    day_str = day if isinstance(day, str) else day.isoformat()
    if period == "monthly":
        date_label = day_str[:7]
        fname = f"{symbol}-{dtype}-{date_label}.zip"
    elif period == "daily":
        date_label = day_str
        fname = f"{symbol}-{dtype}-{day_str}.zip"
    else:
        raise ValueError("period must be 'daily' or 'monthly'")
    key = f"{profile.base_data_path}/{period}/{dtype}/{symbol}/{fname}"
    return ArchiveKey(key=key, size=-1, market=market, dtype=dtype, symbol=symbol, period=period, date_label=date_label)


def parse_s3_listing(xml_text: str) -> tuple[list[ArchiveKey], str | None]:
    """Parse S3 ListObjects XML into archive keys; returns (keys, next_marker).

    Keys look like data/<market-path>/<period>/<dtype>/<SYMBOL>/<SYMBOL>-<dtype>-<date>.zip
    """
    root = ET.fromstring(xml_text)
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    keys: list[ArchiveKey] = []
    for content in root.findall(f"{ns}Contents"):
        key_el = content.find(f"{ns}Key")
        size_el = content.find(f"{ns}Size")
        if key_el is None or not key_el.text:
            continue
        parsed = _parse_key(key_el.text)
        if parsed is None:
            continue
        parsed = _replace_size(parsed, int(size_el.text) if size_el is not None and size_el.text else -1)
        keys.append(parsed)
    truncated = root.find(f"{ns}IsTruncated")
    marker = None
    if truncated is not None and truncated.text == "true":
        next_marker = root.find(f"{ns}NextMarker")
        marker = next_marker.text if next_marker is not None and next_marker.text else (keys[-1].key if keys else "")
    return keys, marker


def _replace_size(k: ArchiveKey, size: int) -> ArchiveKey:
    return ArchiveKey(k.key, size, k.market, k.dtype, k.symbol, k.period, k.date_label)


def _parse_key(key: str) -> ArchiveKey | None:
    parts = key.strip("/").split("/")
    # data / spot | futures[/um] / daily|monthly / dtype / SYMBOL / file.zip
    # spot paths have 6 segments; futures USD-M paths have 7 (extra 'um').
    if parts[0] != "data" or len(parts) not in (6, 7):
        return None
    market_seg = parts[1]
    if market_seg == "futures":
        if len(parts) != 7 or parts[2] != "um":
            return None  # coin-m or options: out of scope
        market, idx = "futures", 3
    elif market_seg == "spot":
        if len(parts) != 6:
            return None
        market, idx = "spot", 2
    else:
        return None
    period = parts[idx]
    if period not in ("daily", "monthly"):
        return None
    dtype, symbol, fname = parts[idx + 1], parts[idx + 2], parts[idx + 3]
    stem = fname[:-4] if fname.endswith(".zip") else None
    if stem is None or not stem.startswith(f"{symbol}-"):
        return None
    date_label = stem[len(symbol) + len(dtype) + 2:]
    if not date_label:
        return None
    return ArchiveKey(key=key, size=-1, market=market, dtype=dtype, symbol=symbol, period=period, date_label=date_label)


def list_archive_files(
    market: str,
    symbol: str,
    dtype: str,
    period: str = "daily",
    session: requests.Session | None = None,
    timeout: float = 30.0,
) -> list[ArchiveKey]:
    """List available archive files for a symbol/dtype via the public S3 API."""
    from .markets import get_market

    profile = get_market(market)
    prefix = f"{profile.base_data_path}/{period}/{dtype}/{symbol.upper()}/"
    sess = session or requests.Session()
    keys: list[ArchiveKey] = []
    marker: str | None = None
    for _ in range(50):  # hard cap on pagination
        params = {"delimiter": "/", "prefix": prefix, "max-keys": 1000}
        if marker:
            params["marker"] = marker
        url = f"{S3_ENDPOINT}?{urllib.parse.urlencode(params)}"
        resp = sess.get(url, timeout=timeout)
        resp.raise_for_status()
        page, next_marker = parse_s3_listing(resp.text)
        keys.extend(k for k in page if k.key.endswith(".zip"))
        if not next_marker:
            break
        marker = next_marker
    return sorted(keys, key=lambda k: k.date_label)


def iter_dates(start: str, end: str):
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    if d1 < d0:
        raise ValueError("end date before start date")
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def estimate_download(
    market: str,
    symbol: str,
    dtype: str,
    start: str,
    end: str,
    period: str = "daily",
    session: requests.Session | None = None,
) -> dict:
    """Estimate file count and total size from the S3 listing (no file downloads)."""
    available = list_archive_files(market, symbol, dtype, period=period, session=session)
    by_date = {k.date_label: k for k in available}
    wanted = [d.isoformat() for d in iter_dates(start, end)]
    found = [by_date[d] for d in wanted if d in by_date]
    missing = [d for d in wanted if d not in by_date]
    total_size = sum(k.size for k in found if k.size > 0)
    unknown = [k.date_label for k in found if k.size <= 0]
    return {
        "market": market,
        "symbol": symbol.upper(),
        "dtype": dtype,
        "period": period,
        "start": start,
        "end": end,
        "files_found": len(found),
        "files_missing": len(missing),
        "missing_dates": missing[:20],
        "approx_total_bytes": total_size,
        "approx_total_mb": round(total_size / 1e6, 1),
        "sizes_unknown_for": len(unknown),
        "note": (
            "Estimates come from the public S3 listing; no files were downloaded."
        ),
    }
