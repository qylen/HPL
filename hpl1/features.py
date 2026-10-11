"""Transform raw event JSONL into timestamp-safe one-second features/labels."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-12
FEATURE_COLUMNS = [
    "ofi",
    "ofi_norm",
    "ofi_norm_5s",
    "trade_imbalance",
    "trade_imbalance_5s",
    "trade_volume_log",
    "trade_volume_5s_log",
    "buy_volume_log",
    "sell_volume_log",
    "trade_count_log",
    "spread_bps",
    "depth_total",
    "depth_imbalance",
    "mid_return_1s",
    "volatility_5s",
    "volatility_30s",
    "quote_update_count_log",
    "quote_age_ms",
    "time_sin",
    "time_cos",
]


def load_events(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read normalized quote/trade records from JSON Lines."""
    quotes: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    malformed = 0
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if item.get("type") == "quote":
                    quotes.append(item)
                elif item.get("type") == "trade":
                    trades.append(item)
            except (json.JSONDecodeError, AttributeError):
                malformed += 1
    if not quotes:
        raise ValueError("No quote events found. Run `hpl1 collect` or check the input file.")
    if malformed:
        # No logger dependency in this module; malformed lines are safely skipped.
        pass
    return pd.DataFrame(quotes), pd.DataFrame(trades)


def _build_quote_buckets(quotes: pd.DataFrame) -> pd.DataFrame:
    required = {"event_time_ms", "seq", "bid", "bid_qty", "ask", "ask_qty"}
    if not required.issubset(quotes.columns):
        raise ValueError(f"Quote events missing columns: {sorted(required - set(quotes.columns))}")

    q = quotes.copy()
    for col in ("event_time_ms", "seq"):
        q[col] = pd.to_numeric(q[col], errors="coerce")
    for col in ("bid", "bid_qty", "ask", "ask_qty"):
        q[col] = pd.to_numeric(q[col], errors="coerce")
    q = q.dropna(subset=["event_time_ms", "bid", "bid_qty", "ask", "ask_qty"])
    q = q[(q["bid"] > 0) & (q["ask"] > 0) & (q["ask"] >= q["bid"])]
    q = q.sort_values(["event_time_ms", "seq"], kind="stable").reset_index(drop=True)
    if q.empty:
        raise ValueError("No valid quote events after basic validation.")

    bid = q["bid"].to_numpy(dtype=float)
    ask = q["ask"].to_numpy(dtype=float)
    bq = q["bid_qty"].to_numpy(dtype=float)
    aq = q["ask_qty"].to_numpy(dtype=float)
    ofi = np.zeros(len(q), dtype=float)
    if len(q) > 1:
        # Best-level order-flow imbalance. Positive values mean net bid-side pressure.
        ofi[1:] = (
            (bid[1:] >= bid[:-1]) * bq[1:]
            - (bid[1:] <= bid[:-1]) * bq[:-1]
            - (ask[1:] <= ask[:-1]) * aq[1:]
            + (ask[1:] >= ask[:-1]) * aq[:-1]
        )
    q["ofi_event"] = ofi
    q["mid"] = (q["bid"] + q["ask"]) / 2.0
    q["spread_bps"] = (q["ask"] - q["bid"]) / q["mid"] * 10_000.0
    q["depth_bid"] = q["bid_qty"].clip(lower=0)
    q["depth_ask"] = q["ask_qty"].clip(lower=0)
    q["depth_total"] = q["depth_bid"] + q["depth_ask"]
    q["depth_imbalance"] = (q["depth_bid"] - q["depth_ask"]) / (q["depth_total"] + EPS)
    q["bucket_ms"] = (q["event_time_ms"].astype("int64") // 1000) * 1000

    # `last` is valid because quotes were sorted chronologically first.
    grouped = q.groupby("bucket_ms", sort=True).agg(
        bid=("bid", "last"),
        ask=("ask", "last"),
        bid_qty=("bid_qty", "last"),
        ask_qty=("ask_qty", "last"),
        mid=("mid", "last"),
        spread_bps=("spread_bps", "last"),
        depth_bid=("depth_bid", "last"),
        depth_ask=("depth_ask", "last"),
        depth_total=("depth_total", "last"),
        depth_imbalance=("depth_imbalance", "last"),
        ofi=("ofi_event", "sum"),
        quote_update_count=("ofi_event", "size"),
        last_quote_event_ms=("event_time_ms", "last"),
    )
    return grouped


def _build_trade_buckets(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["bucket_ms", "buy_volume", "sell_volume", "trade_count"])
    required = {"event_time_ms", "qty", "buyer_is_maker"}
    if not required.issubset(trades.columns):
        raise ValueError(f"Trade events missing columns: {sorted(required - set(trades.columns))}")
    t = trades.copy()
    t["event_time_ms"] = pd.to_numeric(t["event_time_ms"], errors="coerce")
    t["qty"] = pd.to_numeric(t["qty"], errors="coerce")
    t = t.dropna(subset=["event_time_ms", "qty"])
    t = t[t["qty"] >= 0]
    if t.empty:
        return pd.DataFrame(columns=["bucket_ms", "buy_volume", "sell_volume", "trade_count"])
    t["buyer_is_maker"] = t["buyer_is_maker"].astype(str).str.lower().isin(["true", "1", "yes"])
    t["buy_volume"] = np.where(~t["buyer_is_maker"], t["qty"], 0.0)
    t["sell_volume"] = np.where(t["buyer_is_maker"], t["qty"], 0.0)
    t["bucket_ms"] = (t["event_time_ms"].astype("int64") // 1000) * 1000
    return t.groupby("bucket_ms", sort=True).agg(
        buy_volume=("buy_volume", "sum"),
        sell_volume=("sell_volume", "sum"),
        trade_count=("qty", "size"),
    )


def build_feature_frame(quotes: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Build one row per second; features at t predict mid return t -> t+1.

    The feature row includes only events timestamped in its bucket or earlier.
    `target_return` uses the next bucket's mid-price and is a label, never a feature.
    """
    qb = _build_quote_buckets(quotes)
    tb = _build_trade_buckets(trades)
    start_ms = int(qb.index.min())
    end_ms = int(qb.index.max())
    if len(tb):
        start_ms = min(start_ms, int(tb.index.min()))
        end_ms = max(end_ms, int(tb.index.max()))
    # Do not create pre-quote rows; use quote availability as the timeline anchor.
    start_ms = int(qb.index.min())
    second_index = pd.Index(np.arange(start_ms, end_ms + 1000, 1000), name="bucket_ms")
    qcols_ffill = [
        "bid", "ask", "bid_qty", "ask_qty", "mid", "spread_bps",
        "depth_bid", "depth_ask", "depth_total", "depth_imbalance", "last_quote_event_ms",
    ]
    qfull = qb.reindex(second_index)
    for col in qcols_ffill:
        qfull[col] = qfull[col].ffill()
    for col in ("ofi", "quote_update_count"):
        qfull[col] = qfull[col].fillna(0.0)

    tfull = tb.reindex(second_index)
    for col in ("buy_volume", "sell_volume", "trade_count"):
        if col not in tfull:
            tfull[col] = 0.0
        tfull[col] = pd.to_numeric(tfull[col], errors="coerce").astype(float).fillna(0.0)
    df = qfull.join(tfull, how="left")
    df["buy_volume"] = df["buy_volume"].fillna(0.0)
    df["sell_volume"] = df["sell_volume"].fillna(0.0)
    df["trade_count"] = df["trade_count"].fillna(0.0)

    # Drop rows before the first available quote (normally impossible, but explicit).
    df = df.dropna(subset=["mid", "spread_bps"])
    df["timestamp"] = pd.to_datetime(df.index, unit="ms", utc=True)
    df["quote_age_ms"] = ((df.index.to_numpy(dtype=np.int64) + 1000) - df["last_quote_event_ms"].to_numpy(dtype=float)).clip(min=0)
    df["ofi_norm"] = df["ofi"] / (df["depth_total"] + EPS)
    df["trade_volume"] = df["buy_volume"] + df["sell_volume"]
    df["trade_imbalance"] = (df["buy_volume"] - df["sell_volume"]) / (df["trade_volume"] + EPS)
    rolling_buy = df["buy_volume"].rolling(5, min_periods=1).sum()
    rolling_sell = df["sell_volume"].rolling(5, min_periods=1).sum()
    rolling_trade_volume = rolling_buy + rolling_sell
    df["trade_imbalance_5s"] = (rolling_buy - rolling_sell) / (rolling_trade_volume + EPS)
    df["trade_volume_log"] = np.log1p(df["trade_volume"].clip(lower=0))
    df["trade_volume_5s_log"] = np.log1p(rolling_trade_volume.clip(lower=0))
    df["buy_volume_log"] = np.log1p(df["buy_volume"].clip(lower=0))
    df["sell_volume_log"] = np.log1p(df["sell_volume"].clip(lower=0))
    df["trade_count_log"] = np.log1p(df["trade_count"].clip(lower=0))
    df["ofi_norm_5s"] = df["ofi"].rolling(5, min_periods=1).sum() / (df["depth_total"].rolling(5, min_periods=1).mean() + EPS)
    df["mid_return_1s"] = np.log(df["mid"] / df["mid"].shift(1))
    df["volatility_5s"] = df["mid_return_1s"].rolling(5, min_periods=3).std()
    df["volatility_30s"] = df["mid_return_1s"].rolling(30, min_periods=10).std()
    df["quote_update_count_log"] = np.log1p(df["quote_update_count"].clip(lower=0))
    seconds_of_day = (df["timestamp"].dt.hour * 3600 + df["timestamp"].dt.minute * 60 + df["timestamp"].dt.second).astype(float)
    df["time_sin"] = np.sin(2 * np.pi * seconds_of_day / 86400.0)
    df["time_cos"] = np.cos(2 * np.pi * seconds_of_day / 86400.0)

    # Predict next bucket's observed mid-price change; last row has no future label.
    next_mid = df["mid"].shift(-1)
    df["target_return"] = np.log(next_mid / df["mid"])
    # spread_bps and quote_age_ms are also model features; keep each CSV column once.
    keep = list(dict.fromkeys(["timestamp", "mid", "target_return", "spread_bps", "quote_age_ms", *FEATURE_COLUMNS]))
    result = df[keep].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    return result


def build_from_jsonl(input_path: str | Path) -> pd.DataFrame:
    quotes, trades = load_events(input_path)
    return build_feature_frame(quotes, trades)


def save_feature_csv(frame: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
