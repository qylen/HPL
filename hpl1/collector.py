"""Record public Binance USD-M Futures best-book and aggregate-trade events.

This module records data only. It never submits orders and needs no API key for
public market streams. Exchange stream schemas/endpoints can change; check the
exchange's current docs before relying on it for research.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import websockets

LOG = logging.getLogger(__name__)
BASE_WS_URL = "wss://fstream.binance.com/stream?streams={streams}"


def _normalize_message(message: dict[str, Any], seq: int, received_ms: int) -> dict[str, Any] | None:
    """Normalize a Binance combined-stream message to the HPL-1 event schema."""
    data = message.get("data", message)
    if not isinstance(data, dict):
        return None
    event_type = data.get("e")

    # USD-M Futures bookTicker best-bid/best-ask update.
    if event_type == "bookTicker" or all(k in data for k in ("b", "B", "a", "A")):
        try:
            return {
                "type": "quote",
                "event_time_ms": int(data.get("T", data.get("E", received_ms))),
                "received_time_ms": received_ms,
                "seq": seq,
                "bid": float(data["b"]),
                "bid_qty": float(data["B"]),
                "ask": float(data["a"]),
                "ask_qty": float(data["A"]),
            }
        except (KeyError, TypeError, ValueError):
            return None

    # aggTrade's m=true means the buyer was maker, i.e. sell-aggressor trade.
    if event_type == "aggTrade" or all(k in data for k in ("p", "q", "m")):
        try:
            return {
                "type": "trade",
                "event_time_ms": int(data.get("T", data.get("E", received_ms))),
                "received_time_ms": received_ms,
                "seq": seq,
                "price": float(data["p"]),
                "qty": float(data["q"]),
                "buyer_is_maker": bool(data["m"]),
            }
        except (KeyError, TypeError, ValueError):
            return None
    return None


async def collect_events(
    symbol: str,
    output: str | Path,
    duration_seconds: float = 0,
    reconnect_max_seconds: float = 30,
) -> None:
    """Append quote/trade events to JSONL, reconnecting after transient errors.

    duration_seconds=0 runs until interrupted. A single monotonically increasing
    `seq` preserves local receive order across the two multiplexed streams.
    """
    symbol = symbol.lower().strip()
    if not symbol.isalnum():
        raise ValueError("symbol must contain only letters and digits, e.g. btcusdt")
    if duration_seconds < 0:
        raise ValueError("duration_seconds cannot be negative")

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    streams = f"{symbol}@bookTicker/{symbol}@aggTrade"
    url = BASE_WS_URL.format(streams=streams)
    deadline = time.monotonic() + duration_seconds if duration_seconds > 0 else None
    seq = 0
    # Continue the sequence number when appending to an existing event log.
    if out.exists() and out.stat().st_size > 0:
        try:
            with out.open("rb") as existing:
                existing.seek(0, 2)
                pos = existing.tell() - 1
                if pos >= 0:
                    existing.seek(pos)
                    if existing.read(1) == b"\n":
                        pos -= 1
                while pos >= 0:
                    existing.seek(pos)
                    if existing.read(1) == b"\n":
                        break
                    pos -= 1
                existing.seek(pos + 1)
                last_line = existing.readline().decode("utf-8", errors="ignore").strip()
                if last_line:
                    seq = int(json.loads(last_line).get("seq", 0))
        except (OSError, ValueError, json.JSONDecodeError):
            LOG.warning("Could not read last event sequence number; restarting seq at zero")
            seq = 0
    saved = 0
    backoff = 1.0

    LOG.info("Connecting to public market stream for %s", symbol.upper())
    with out.open("a", encoding="utf-8", buffering=1) as fh:
        while deadline is None or time.monotonic() < deadline:
            remaining = None if deadline is None else max(0.1, deadline - time.monotonic())
            try:
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=5,
                    open_timeout=10,
                    max_size=None,
                ) as ws:
                    LOG.info("Connected. Appending events to %s", out)
                    backoff = 1.0
                    while deadline is None or time.monotonic() < deadline:
                        try:
                            timeout = None if deadline is None else max(0.1, deadline - time.monotonic())
                            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                        except asyncio.TimeoutError:
                            break
                        received_ms = time.time_ns() // 1_000_000
                        try:
                            message = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        seq += 1
                        record = _normalize_message(message, seq, received_ms)
                        if record is None:
                            continue
                        fh.write(json.dumps(record, separators=(",", ":")) + "\n")
                        saved += 1
                        if saved % 100 == 0:
                            fh.flush()
                        if saved % 10_000 == 0:
                            LOG.info("Recorded %s events", f"{saved:,}")
                if deadline is not None and time.monotonic() >= deadline:
                    break
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # network disconnects and transient stream errors
                LOG.warning("Stream error: %s; reconnecting in %.1fs", exc, backoff)
                if deadline is not None and time.monotonic() >= deadline:
                    break
                await asyncio.sleep(min(backoff, max(0.0, deadline - time.monotonic())) if deadline else backoff)
                backoff = min(reconnect_max_seconds, backoff * 2)
        fh.flush()
    LOG.info("Stopped. Wrote %s events to %s", f"{saved:,}", out)
