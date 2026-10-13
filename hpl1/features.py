"""One-second feature and label generation.

Two entry points share the same math:
- `build_features(dataset_frame, ...)` for archive-based research/paper trading
- legacy `build_feature_frame(quotes, trades)` for JSONL collected events

Guarantees:
- features at second t use ONLY information available up to the end of second t
- target_return[t] = log(price[t+1] / price[t]); the final row has no label
- OFI uses the standard Cont-Kukanov-Stoikov best-level convention (the
  original repository's ask-side sign was inverted; fixed here)
- stale quotes are forward-filled but explicitly marked invalid via
  quote_age/data_gap flags; they are never silently treated as fresh
- zero denominators are guarded by EPS
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-12
GAP_THRESHOLD_SECONDS = 10.0     # no book/trade event for this long -> data_gap flag
STALE_MAX_SECONDS = 30.0         # quote older than this -> invalid for trading

# Legacy 20-feature schema (kept for JSONL compatibility and old artifacts).
FEATURE_COLUMNS = [
    "ofi", "ofi_norm", "ofi_norm_5s", "trade_imbalance", "trade_imbalance_5s",
    "trade_volume_log", "trade_volume_5s_log", "buy_volume_log", "sell_volume_log",
    "trade_count_log", "spread_bps", "depth_total", "depth_imbalance", "mid_return_1s",
    "volatility_5s", "volatility_30s", "quote_update_count_log", "quote_age_ms",
    "time_sin", "time_cos",
]

# Full archive feature set (book + trades; futures bookTicker path).
FEATURE_COLUMNS_FULL = FEATURE_COLUMNS + [
    "volatility_60s", "volume_imb_60s", "ofi_lag1", "ret_lag1", "liq_pressure",
    "price_impact_5s", "activity_z", "vol_regime",
]

# Trade-only feature set (spot has NO published book history).
FEATURE_COLUMNS_TRADE = [
    "trade_imbalance", "trade_imbalance_5s", "trade_volume_log", "trade_volume_5s_log",
    "buy_volume_log", "sell_volume_log", "trade_count_log", "trade_count_5s_log",
    "mid_return_1s", "volatility_5s", "volatility_30s", "volatility_60s",
    "time_sin", "time_cos", "ret_lag1", "volume_imb_60s", "price_impact_5s",
]

META_COLUMNS = ["ts", "second", "price", "bid", "ask", "mid", "spread_bps",
                "quote_age_ms", "quote_age_seconds", "valid", "data_gap", "funding_rate",
                "n_updates", "n_trades"]  # event counts kept for quality reports


def feature_columns_for(feature_set: str) -> list[str]:
    if feature_set == "full":
        return list(FEATURE_COLUMNS_FULL)
    if feature_set == "trade_only":
        return list(FEATURE_COLUMNS_TRADE)
    raise ValueError("feature_set must be 'full' or 'trade_only'")


# ----------------------------------------------------------------------------
# OFI (Cont-Kukanov-Stoikov best level; positive = bid-side pressure)
# ----------------------------------------------------------------------------

def ofi_between(prev_quote, quote) -> float:
    """OFI of one quote update vs the previous applied quote.

    prev_quote/quote: (bid, bid_qty, ask, ask_qty) tuples.
    """
    pb, pbq, pa, paq = prev_quote
    b, bq, a, aq = quote
    if b > pb:
        bid_c = bq
    elif b == pb:
        bid_c = bq - pbq
    else:
        bid_c = -pbq
    if a < pa:            # offer lifted by buyers
        ask_c = paq
    elif a == pa:
        ask_c = -(aq - paq)
    else:                 # new offers posted above
        ask_c = -aq
    return bid_c + ask_c


# ----------------------------------------------------------------------------
# Archive-based feature construction (research + live paper trading share this)
# ----------------------------------------------------------------------------

def build_features(df: pd.DataFrame, feature_set: str = "full",
                   gap_threshold_seconds: float = GAP_THRESHOLD_SECONDS,
                   stale_max_seconds: float = STALE_MAX_SECONDS) -> pd.DataFrame:
    """Build the per-second feature frame from an assembled dataset frame.

    Input frame (from dataset.py or the live engine): one row per second with
    columns: second, ts, n_trades, buy_vol, sell_vol, vol, notional and,
    when book data exists, bid, bid_qty, ask, ask_qty, ofi, n_updates,
    last_quote_second, plus optional funding_rate.
    """
    if feature_set == "full" and not {"bid", "ask"}.issubset(df.columns):
        raise ValueError("feature_set='full' requires book columns (bid/ask); "
                         "spot has no published book history — use 'trade_only'")
    df = df.sort_values("second", kind="stable").reset_index(drop=True)
    # Restore missing seconds so rolling windows stay time-true; inserted rows
    # carry no events and are flagged as gaps by the staleness logic below.
    sec_values = df["second"].to_numpy(dtype=np.int64)
    if len(sec_values) > 1:
        full = np.arange(sec_values[0], sec_values[-1] + 1, dtype=np.int64)
        if len(full) != len(sec_values) and len(full) <= max(len(sec_values) * 4, 100_000):
            df = (df.set_index("second").reindex(full).rename_axis("second").reset_index())
    out = pd.DataFrame(index=df.index)
    out["second"] = df["second"]
    out["ts"] = pd.to_datetime(df["second"], unit="s", utc=True)

    for c in ("n_trades", "buy_vol", "sell_vol", "vol", "notional"):
        if c not in df.columns:
            out[c] = 0.0
        else:
            out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
    has_book = {"bid", "ask"}.issubset(df.columns)

    # ---- price definition -------------------------------------------------
    if has_book:
        book_bid = pd.to_numeric(df["bid"], errors="coerce")
        book_ask = pd.to_numeric(df["ask"], errors="coerce")
        book_bidq = pd.to_numeric(df.get("bid_qty"), errors="coerce")
        book_askq = pd.to_numeric(df.get("ask_qty"), errors="coerce")
        # last_quote_second: NaN where no book event happened this second
        lqs = df["second"].where(df["n_updates"].fillna(0) > 0) if "n_updates" in df.columns \
            else df["second"].where(book_bid.notna())
        lqs = lqs.ffill()
        quote_age = (df["second"] - lqs).astype(float)  # 0 when event this second
        out["bid"] = book_bid.ffill()
        out["ask"] = book_ask.ffill()
        out["bid_qty"] = book_bidq.ffill()
        out["ask_qty"] = book_askq.ffill()
        out["mid"] = (out["bid"] + out["ask"]) / 2.0
        out["price"] = out["mid"]
        out["spread_bps"] = (out["ask"] - out["bid"]) / (out["mid"] + EPS) * 1e4
        out["depth_total"] = (out["bid_qty"].fillna(0) + out["ask_qty"].fillna(0))
        out["depth_imbalance"] = ((out["bid_qty"].fillna(0) - out["ask_qty"].fillna(0))
                                  / (out["depth_total"] + EPS))
        ofi = pd.to_numeric(df.get("ofi"), errors="coerce").fillna(0.0) if "ofi" in df.columns else 0.0
        out["ofi"] = ofi
        out["n_updates"] = pd.to_numeric(df.get("n_updates"), errors="coerce").fillna(0.0) \
            if "n_updates" in df.columns else 0.0
        # invalid quote conditions: never applied, crossed after ffill, nonpositive
        invalid_quote = out["bid"].isna() | out["ask"].isna() | (out["bid"] <= 0) | \
            (out["ask"] <= out["bid"])
        out["quote_age_seconds"] = quote_age.fillna(np.inf)
        out["quote_age_ms"] = quote_age.fillna(np.inf) * 1000.0
        out["data_gap"] = (quote_age > gap_threshold_seconds) | invalid_quote
        out["valid"] = (~out["data_gap"]) & (out["quote_age_seconds"] <= stale_max_seconds)
    else:
        # Trade-only price proxy: per-second VWAP, forward-filled with ageing.
        px = pd.to_numeric(df.get("last_px"), errors="coerce")
        vol = out["vol"]
        vwap = pd.to_numeric(df.get("notional"), errors="coerce") / vol.where(vol > 0)
        price = vwap.where(vwap.notna() & (vol > 0), np.nan)
        price = price.ffill()
        if price.isna().all():
            price = px.ffill()
        lts = df["second"].where(vol > 0).ffill()
        trade_age = (df["second"] - lts).astype(float)
        out["price"] = price
        out["bid"] = np.nan
        out["ask"] = np.nan
        out["mid"] = price
        out["spread_bps"] = np.nan          # unknown without book data
        out["depth_total"] = np.nan
        out["depth_imbalance"] = np.nan
        out["ofi"] = 0.0
        out["n_updates"] = 0.0
        out["quote_age_seconds"] = trade_age.fillna(np.inf)
        out["quote_age_ms"] = out["quote_age_seconds"] * 1000.0
        out["data_gap"] = (trade_age > gap_threshold_seconds) | price.isna()
        out["valid"] = (~out["data_gap"]) & (out["quote_age_seconds"] <= stale_max_seconds)

    # ---- trade-flow features ----------------------------------------------
    buy, sell, vol, ntr = out["buy_vol"], out["sell_vol"], out["vol"], out["n_trades"]
    out["trade_volume"] = vol
    out["trade_imbalance"] = (buy - sell) / (vol + EPS)
    out["trade_volume_log"] = np.log1p(vol.clip(lower=0))
    out["trade_volume_5s_log"] = np.log1p(vol.rolling(5, min_periods=1).sum().clip(lower=0))
    out["buy_volume_log"] = np.log1p(buy.clip(lower=0))
    out["sell_volume_log"] = np.log1p(sell.clip(lower=0))
    out["trade_count_log"] = np.log1p(ntr.clip(lower=0))
    out["trade_count_5s_log"] = np.log1p(ntr.rolling(5, min_periods=1).sum().clip(lower=0))
    rb = buy.rolling(5, min_periods=1).sum()
    rs = sell.rolling(5, min_periods=1).sum()
    out["trade_imbalance_5s"] = (rb - rs) / (rb + rs + EPS)
    rb60 = buy.rolling(60, min_periods=1).sum()
    rs60 = sell.rolling(60, min_periods=1).sum()
    out["volume_imb_60s"] = (rb60 - rs60) / (rb60 + rs60 + EPS)

    # ---- book features ------------------------------------------------------
    if has_book:
        out["ofi_norm"] = out["ofi"] / (out["depth_total"] + EPS)
        out["ofi_norm_5s"] = out["ofi"].rolling(5, min_periods=1).sum() / (
            out["depth_total"].rolling(5, min_periods=1).mean() + EPS)
        out["ofi_lag1"] = out["ofi"].shift(1)
        out["quote_update_count_log"] = np.log1p(out["n_updates"].clip(lower=0))
        out["liq_pressure"] = (buy - sell) / (out["depth_total"] + EPS)
    else:
        out["ofi_norm"] = 0.0
        out["ofi_norm_5s"] = 0.0
        out["ofi_lag1"] = 0.0
        out["quote_update_count_log"] = np.log1p(ntr.clip(lower=0))
        out["liq_pressure"] = out["trade_imbalance"]

    # ---- returns, volatility, regimes, calendar ------------------------------
    ret = np.log(out["price"] / out["price"].shift(1))
    out["mid_return_1s"] = ret
    out["ret_lag1"] = ret.shift(1)
    out["volatility_5s"] = ret.rolling(5, min_periods=3).std()
    out["volatility_30s"] = ret.rolling(30, min_periods=10).std()
    out["volatility_60s"] = ret.rolling(60, min_periods=20).std()
    out["volatility_300s"] = ret.rolling(300, min_periods=60).std()
    out["vol_regime"] = out["volatility_60s"] / (out["volatility_300s"].abs() + EPS)
    ret5 = np.log(out["price"] / out["price"].shift(5))
    vol5 = vol.rolling(5, min_periods=1).sum()
    out["price_impact_5s"] = ret5 / (vol5 + EPS)  # Kyle-lambda proxy: bps per unit volume
    activity_mean = ntr.rolling(300, min_periods=30).mean()
    activity_std = ntr.rolling(300, min_periods=30).std()
    out["activity_z"] = (ntr - activity_mean) / (activity_std + EPS)
    sec_of_day = out["ts"].dt.hour * 3600 + out["ts"].dt.minute * 60 + out["ts"].dt.second
    out["time_sin"] = np.sin(2 * np.pi * sec_of_day / 86400.0)
    out["time_cos"] = np.cos(2 * np.pi * sec_of_day / 86400.0)

    if "funding_rate" in df.columns:
        out["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    else:
        out["funding_rate"] = np.nan

    # ---- labels --------------------------------------------------------------
    out["target_return"] = np.log(out["price"].shift(-1) / out["price"])

    cols = (META_COLUMNS + feature_columns_for(feature_set) + ["target_return"])
    cols = list(dict.fromkeys(c for c in cols if c in out.columns))
    result = out[cols].replace([np.inf, -np.inf], np.nan)
    return result


# ----------------------------------------------------------------------------
# Legacy JSONL path (kept for compatibility with `hpl1 collect`)
# ----------------------------------------------------------------------------

def load_events(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return pd.DataFrame(quotes), pd.DataFrame(trades)


def _build_quote_buckets(quotes: pd.DataFrame) -> pd.DataFrame:
    required = {"event_time_ms", "seq", "bid", "bid_qty", "ask", "ask_qty"}
    if not required.issubset(quotes.columns):
        raise ValueError(f"Quote events missing columns: {sorted(required - set(quotes.columns))}")
    q = quotes.copy()
    for col in q.columns:
        if col != "type" and col != "buyer_is_maker":
            q[col] = pd.to_numeric(q[col], errors="coerce")
    q = q.dropna(subset=["event_time_ms", "bid", "bid_qty", "ask", "ask_qty"])
    q = q[(q["bid"] > 0) & (q["ask"] > 0) & (q["ask"] >= q["bid"])]
    q = q.sort_values(["event_time_ms", "seq"], kind="stable").reset_index(drop=True)
    if q.empty:
        raise ValueError("No valid quote events after basic validation.")
    bid = q["bid"].to_numpy(dtype=float)
    ask = q["ask"].to_numpy(dtype=float)
    bidq = q["bid_qty"].to_numpy(dtype=float)
    askq = q["ask_qty"].to_numpy(dtype=float)
    # Corrected OFI (see ofi_between): ask-side sign fixed vs the original code.
    ofi = np.zeros(len(q))
    for i in range(1, len(q)):
        ofi[i] = ofi_between((bid[i - 1], bidq[i - 1], ask[i - 1], askq[i - 1]),
                             (bid[i], bidq[i], ask[i], askq[i]))
    q["ofi_event"] = ofi
    q["mid"] = (q["bid"] + q["ask"]) / 2.0
    q["spread_bps"] = (q["ask"] - q["bid"]) / q["mid"] * 10_000.0
    q["depth_bid"] = q["bid_qty"].clip(lower=0)
    q["depth_ask"] = q["ask_qty"].clip(lower=0)
    q["depth_total"] = q["depth_bid"] + q["depth_ask"]
    q["depth_imbalance"] = (q["depth_bid"] - q["depth_ask"]) / (q["depth_total"] + EPS)
    q["bucket_ms"] = (q["event_time_ms"].astype("int64") // 1000) * 1000
    return q.groupby("bucket_ms", sort=True).agg(
        bid=("bid", "last"), ask=("ask", "last"), bid_qty=("bid_qty", "last"),
        ask_qty=("ask_qty", "last"), mid=("mid", "last"), spread_bps=("spread_bps", "last"),
        depth_bid=("depth_bid", "last"), depth_ask=("depth_ask", "last"),
        depth_total=("depth_total", "last"), depth_imbalance=("depth_imbalance", "last"),
        ofi=("ofi_event", "sum"), quote_update_count=("ofi_event", "size"),
        last_quote_event_ms=("event_time_ms", "last"),
    )


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
        buy_volume=("buy_volume", "sum"), sell_volume=("sell_volume", "sum"),
        trade_count=("qty", "size"),
    )


def build_feature_frame(quotes: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Legacy one-row-per-second frame from collected JSONL events."""
    qb = _build_quote_buckets(quotes)
    tb = _build_trade_buckets(trades)
    start_ms = int(qb.index.min())
    end_ms = int(qb.index.max())
    second_index = pd.Index(np.arange(start_ms, end_ms + 1000, 1000), name="bucket_ms")
    qcols = ["bid", "ask", "bid_qty", "ask_qty", "mid", "spread_bps", "depth_bid",
             "depth_ask", "depth_total", "depth_imbalance", "last_quote_event_ms"]
    qfull = qb.reindex(second_index)
    for col in qcols:
        qfull[col] = qfull[col].ffill()
    for col in ("ofi", "quote_update_count"):
        qfull[col] = qfull[col].fillna(0.0)
    tfull = tb.reindex(second_index)
    for col in ("buy_volume", "sell_volume", "trade_count"):
        if col not in tfull:
            tfull[col] = 0.0
        tfull[col] = pd.to_numeric(tfull[col], errors="coerce").astype(float).fillna(0.0)
    df = qfull.join(tfull, how="left")
    for col in ("buy_volume", "sell_volume", "trade_count"):
        df[col] = df[col].fillna(0.0)
    df = df.dropna(subset=["mid", "spread_bps"])
    df["timestamp"] = pd.to_datetime(df.index, unit="ms", utc=True)
    df["quote_age_ms"] = ((df.index.to_numpy(dtype=np.int64) + 1000)
                          - df["last_quote_event_ms"].to_numpy(dtype=float)).clip(min=0)
    df["ofi_norm"] = df["ofi"] / (df["depth_total"] + EPS)
    df["trade_volume"] = df["buy_volume"] + df["sell_volume"]
    df["trade_imbalance"] = (df["buy_volume"] - df["sell_volume"]) / (df["trade_volume"] + EPS)
    rolling_buy = df["buy_volume"].rolling(5, min_periods=1).sum()
    rolling_sell = df["sell_volume"].rolling(5, min_periods=1).sum()
    rolling_vol = rolling_buy + rolling_sell
    df["trade_imbalance_5s"] = (rolling_buy - rolling_sell) / (rolling_vol + EPS)
    df["trade_volume_log"] = np.log1p(df["trade_volume"].clip(lower=0))
    df["trade_volume_5s_log"] = np.log1p(rolling_vol.clip(lower=0))
    df["buy_volume_log"] = np.log1p(df["buy_volume"].clip(lower=0))
    df["sell_volume_log"] = np.log1p(df["sell_volume"].clip(lower=0))
    df["trade_count_log"] = np.log1p(df["trade_count"].clip(lower=0))
    df["ofi_norm_5s"] = df["ofi"].rolling(5, min_periods=1).sum() / (
        df["depth_total"].rolling(5, min_periods=1).mean() + EPS)
    df["mid_return_1s"] = np.log(df["mid"] / df["mid"].shift(1))
    df["volatility_5s"] = df["mid_return_1s"].rolling(5, min_periods=3).std()
    df["volatility_30s"] = df["mid_return_1s"].rolling(30, min_periods=10).std()
    df["quote_update_count_log"] = np.log1p(df["quote_update_count"].clip(lower=0))
    sod = (df["timestamp"].dt.hour * 3600 + df["timestamp"].dt.minute * 60
           + df["timestamp"].dt.second).astype(float)
    df["time_sin"] = np.sin(2 * np.pi * sod / 86400.0)
    df["time_cos"] = np.cos(2 * np.pi * sod / 86400.0)
    df["target_return"] = np.log(df["mid"].shift(-1) / df["mid"])
    keep = list(dict.fromkeys(["timestamp", "mid", "target_return", "spread_bps",
                               "quote_age_ms", *FEATURE_COLUMNS]))
    return df[keep].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


def build_from_jsonl(input_path: str | Path) -> pd.DataFrame:
    quotes, trades = load_events(input_path)
    return build_feature_frame(quotes, trades)


def save_feature_csv(frame: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
