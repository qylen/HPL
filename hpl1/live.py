"""Live-data paper-trading engine.

Subscribes to public market streams (no API key, no orders — paper only),
maintains a synchronized top-of-book, aggregates per-second buckets, and runs
THE SAME feature builder and cost accounting as historical research. Includes
reconnect with book-state reset, staleness detection, dual trade-stream
subscription (futures aggTrade streams can be silently dead on some networks;
raw @trade is the fallback), duplicate-bucket-safe boundary processing, kill
switch, health snapshots and graceful shutdown.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import websockets

from .config import CostConfig, RiskConfig
from .exchange_meta import SymbolRules, fetch_symbol_rules
from .features import build_features, feature_columns_for, ofi_between
from .logging_setup import setup_logging
from .markets import MarketProfile
from .models import BaseModel
from .risk import KillSwitch

LOG_NAME = "hpl1.live"


class PaperTradingEngine:
    """One instance runs one symbol/market paper session."""

    def __init__(self, profile: MarketProfile, model: BaseModel, cost: CostConfig,
                 risk: RiskConfig, run_dir: str | Path,
                 duration_seconds: float = 0, heartbeat_seconds: int = 60,
                 rules: SymbolRules | None = None, log_events_every: int = 200,
                 rolling_window: int = 900):
        self.profile = profile
        self.symbol = profile.symbol
        self.model = model
        self.cost = cost
        self.risk = risk
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.duration = float(duration_seconds or 0)
        self.heartbeat = heartbeat_seconds
        self.log = setup_logging(log_dir=self.run_dir / "logs", name=LOG_NAME)
        self.rules = rules or SymbolRules(profile.name, profile.symbol, "0.10", "0.001",
                                          100.0, source="fallback")
        self.log_events_every = log_events_every
        self.rolling_window = rolling_window
        self.kill = KillSwitch(path=self.run_dir / "kill_switch")

        # market state
        self.book: dict[str, float] | None = None       # bid,bidq,ask,askq
        self.last_update_id = -1
        self.book_event_second = -1
        self.last_event_monotonic = time.monotonic()
        self.stream_dead_warning_ts = 0.0
        self.funding_rate: float | None = None
        self._seen_trade_ids: dict[str, bool] = {}

        # per-second bucket being accumulated
        self.bucket_second = -1
        self.bucket = self._empty_bucket()

        # rolling dataset frame (completed seconds)
        self.rolling: list[dict] = []

        # paper account
        self.cash = self.risk.starting_capital
        self.qty = 0.0
        self.entry_price = 0.0
        self.realized_pnl = 0.0
        self.n_fills = 0
        self.n_decisions = 0
        self.n_skipped = 0
        self.last_trade_second = -10**9
        self.last_decided_second = -1
        self.day_realized_start = self.risk.starting_capital
        self.current_day = ""
        self.turnover_today = 0.0
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self.events_count = 0
        self.last_prediction = None
        self._stop = asyncio.Event()
        self._load_state()

        # files
        self.obs_fh = (self.run_dir / "observations.jsonl").open("a", encoding="utf-8")
        self.dec_fh = (self.run_dir / "decisions.jsonl").open("a", encoding="utf-8")
        self.ord_fh = (self.run_dir / "orders.jsonl").open("a", encoding="utf-8")
        self.fil_fh = (self.run_dir / "fills.jsonl").open("a", encoding="utf-8")
        self.eq_fh = (self.run_dir / "equity.jsonl").open("a", encoding="utf-8")

    # ------------------------------------------------------------------ state
    def _empty_bucket(self) -> dict:
        return {"n_updates": 0, "ofi": 0.0, "bid": np.nan, "bid_qty": np.nan,
                "ask": np.nan, "ask_qty": np.nan, "last_update_id": -1,
                "n_trades": 0, "buy_vol": 0.0, "sell_vol": 0.0, "vol": 0.0,
                "notional": 0.0, "last_px": np.nan, "funding_rate": np.nan}

    def _state_path(self) -> Path:
        return self.run_dir / "state.json"

    def _load_state(self) -> None:
        p = self._state_path()
        if not p.exists():
            return
        try:
            s = json.loads(p.read_text())
            self.cash = float(s.get("cash", self.cash))
            self.qty = float(s.get("qty", 0.0))
            self.entry_price = float(s.get("entry_price", 0.0))
            self.realized_pnl = float(s.get("realized_pnl", 0.0))
            self.n_fills = int(s.get("n_fills", 0))
            self.started_utc = s.get("started_utc", self.started_utc)
            self.log.warning("resumed paper account state: qty=%.6f cash=%.2f", self.qty, self.cash)
        except (json.JSONDecodeError, ValueError, OSError):
            self.log.warning("state.json unreadable; starting flat accounting")

    def _save_state(self) -> None:
        self._state_path().write_text(json.dumps({
            "cash": self.cash, "qty": self.qty, "entry_price": self.entry_price,
            "realized_pnl": self.realized_pnl, "n_fills": self.n_fills,
            "started_utc": self.started_utc,
            "saved_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2))

    # ---------------------------------------------------------------- streams
    def _streams(self) -> list[str]:
        s = self.symbol.lower()
        streams = [f"{s}@bookTicker", f"{s}@aggTrade", f"{s}@trade"]
        if self.profile.has_funding:
            streams.append(f"{s}@markPrice@1s")
        return streams

    def _ws_url(self) -> str:
        streams = "/".join(self._streams())
        return f"{self.profile.ws_base}/stream?streams={streams}"

    def _handle_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        data = msg.get("data", msg)
        if not isinstance(data, dict):
            return
        etype = data.get("e")
        now_ms = int(time.time() * 1000)
        if etype == "bookTicker" or ("b" in data and "B" in data and "a" in data and "A" in data):
            self._on_quote(data, now_ms)
        elif etype in ("aggTrade", "trade") or ("p" in data and "q" in data and "m" in data):
            self._on_trade(data, now_ms)
        elif etype == "markPriceUpdate":
            with contextlib.suppress(ValueError, TypeError):
                self.funding_rate = float(data.get("r", self.funding_rate or 0.0))

    def _on_quote(self, data: dict, received_ms: int) -> None:
        try:
            u = int(data.get("u", data.get("U", -1)))
            bid = float(data["b"]); bidq = float(data["B"])
            ask = float(data["a"]); askq = float(data["A"])
            ts_ms = int(data.get("T", data.get("E", data.get("transactTime", received_ms))))
        except (KeyError, TypeError, ValueError):
            return
        if bid <= 0 or ask <= 0 or bid >= ask:
            return  # invalid/crossed: never apply
        sec = ts_ms // 1000
        prev = self.book
        if u <= self.last_update_id and prev is not None:
            return  # duplicate/regression after reconnect
        self.last_event_monotonic = time.monotonic()
        self.events_count += 1
        if sec > self.bucket_second:
            self._rollover(sec)
        if prev is not None:
            # OFI of this event (vs the previous applied quote) belongs to the
            # second the EVENT is timestamped in = the current bucket.
            self.bucket["ofi"] += ofi_between(
                (prev["bid"], prev["bidq"], prev["ask"], prev["askq"]), (bid, bidq, ask, askq))
        self.book = {"bid": bid, "bidq": bidq, "ask": ask, "askq": askq}
        self.last_update_id = u
        self.book_event_second = sec
        b = self.bucket
        b["n_updates"] += 1
        b["bid"], b["bid_qty"], b["ask"], b["ask_qty"] = bid, bidq, ask, askq
        b["last_update_id"] = u
        if self.events_count % self.log_events_every == 0:
            self._log_json(self.obs_fh, {"kind": "quote_sample", "t": received_ms,
                                         "u": u, "bid": bid, "ask": ask, "seq": self.events_count})

    def _on_trade(self, data: dict, received_ms: int) -> None:
        try:
            px = float(data["p"]); qty = float(data["q"])
            maker = bool(data.get("m", False))
            ts_ms = int(data.get("T", data.get("E", data.get("tradeTime", received_ms))))
        except (KeyError, TypeError, ValueError):
            return
        if px <= 0 or qty < 0:
            return
        # dual-stream dedup: futures @aggTrade and @trade can both deliver the
        # same fill; drop a trade with an id we already logged this second.
        tid = data.get("a", data.get("t", data.get("tradeId")))
        key = f"{ts_ms}:{tid}"
        if key in self._seen_trade_ids:
            return
        self._seen_trade_ids[key] = True
        if len(self._seen_trade_ids) > 20000:
            self._seen_trade_ids.clear()
        sec = ts_ms // 1000
        self.last_event_monotonic = time.monotonic()
        self.events_count += 1
        if sec > self.bucket_second:
            self._rollover(sec)
        b = self.bucket
        b["n_trades"] += 1
        b["vol"] += qty
        b["notional"] += px * qty
        b["last_px"] = px
        if maker:
            b["sell_vol"] += qty
        else:
            b["buy_vol"] += qty

    # ------------------------------------------------------------- boundaries
    def _rollover(self, new_second: int) -> None:
        """Close the current bucket (possibly multiple boundaries late) and
        start a new one. Duplicate-bucket safe: process every missed boundary.
        Monotonic: backward timestamps (exchange/local clock skew) are ignored."""
        old = self.bucket_second
        if new_second <= old:
            return  # out-of-order event timestamp; keep the current bucket
        if old >= 0:
            row = {"second": old, **self.bucket}
            self.rolling.append(row)
            if len(self.rolling) > self.rolling_window:
                self.rolling = self.rolling[-self.rolling_window:]
        # fill any fully-missed seconds as empty buckets (gaps marked later)
        if old >= 0:
            missed = new_second - old - 1
            if missed > 3600:
                self.log.warning("event timestamp jumped %ds ahead; resyncing", missed)
                missed = 0
            for k in range(1, min(missed, 60) + 1):
                self.rolling.append({"second": old + k, **self._empty_bucket()})
                if len(self.rolling) > self.rolling_window:
                    self.rolling = self.rolling[-self.rolling_window:]
        self.bucket_second = new_second
        self.bucket = self._empty_bucket()
        if new_second % self.heartbeat == 0:
            self._heartbeat()
        if self.profile.has_funding and new_second % (3600 * self.profile.funding_interval_hours) == 0:
            self._apply_funding(new_second)

    def _apply_funding(self, second: int) -> None:
        if self.qty == 0:
            return
        rate = self.funding_rate if self.funding_rate is not None else 0.0
        px = (self.book or {}).get("bid", np.nan)
        if not np.isfinite(px):
            return
        payment = self.qty * px * rate
        self.cash -= payment
        self._log_json(self.fil_fh, {"kind": "funding", "second": second, "rate": rate,
                                     "payment": payment, "qty": self.qty})

    # ------------------------------------------------------------- decisions
    def _infer_feature_set(self) -> str:
        """The feature set THE MODEL was trained on wins over data availability:
        a trade-only model must never be fed (or rejected for lacking) book
        features just because a book stream happens to be connected."""
        names = getattr(self.model, "feature_names_", None) or []
        book_markers = {"ofi", "spread_bps", "depth_imbalance", "depth_total",
                        "quote_update_count_log"}
        if names and not (set(names) & book_markers):
            return "trade_only"
        return "full"

    def _feature_frame(self) -> pd.DataFrame | None:
        if len(self.rolling) < 60:
            return None
        df = pd.DataFrame(self.rolling)
        if df["second"].duplicated().any():
            df = df.drop_duplicates(subset="second", keep="last")
        df = df.sort_values("second", kind="stable")
        feature_set = self._infer_feature_set()
        if feature_set == "full" and not df["bid"].notna().any():
            feature_set = "trade_only"  # no book data at all
        try:
            feats = build_features(df, feature_set=feature_set)
        except ValueError:
            feats = build_features(df, feature_set="trade_only")
        return feats

    def _decide_and_trade(self) -> None:
        feats = self._feature_frame()
        if feats is None or len(feats) == 0:
            return
        row = feats.iloc[-1]
        fcols = feature_columns_for(self._infer_feature_set())
        # fall back to the trade-only columns that actually exist
        fcols = [c for c in fcols if c in feats.columns]
        raw_names = getattr(self.model, "feature_names_", None)
        model_cols = list(raw_names) if raw_names else []
        if model_cols and set(model_cols) - set(feats.columns):
            self.log.error("model/feature mismatch: missing %s",
                           sorted(set(model_cols) - set(feats.columns))[:5])
            self.n_skipped += 1
            return
        available = [c for c in (model_cols or fcols) if c in feats.columns]
        X = feats[available].iloc[[-1]]
        if X.isna().to_numpy().any():
            self.n_skipped += 1
            return  # insufficient rolling history for a clean feature row
        try:
            pred = float(self.model.predict(X)[0])
        except Exception as exc:  # noqa: BLE001
            self.log.error("predict failed: %s", exc)
            self.n_skipped += 1
            return
        self.n_decisions += 1
        self.last_prediction = pred
        now_sec = int(row["second"])
        valid = bool(row.get("valid", False))
        mid = float(row.get("mid", np.nan)) if np.isfinite(row.get("mid", np.nan)) else \
            float(row.get("price", np.nan))
        spread_bps = float(row.get("spread_bps", 1.0)) if np.isfinite(row.get("spread_bps", np.nan)) else 1.0
        record = {"kind": "decision", "second": now_sec, "prediction_bps": pred * 1e4,
                  "mid": mid, "valid": valid, "qty": self.qty, "cash": self.cash}
        if not valid or not np.isfinite(mid):
            self.n_skipped += 1
            record["action"] = "skip_invalid_data"
            self._log_json(self.dec_fh, record)
            return
        if self.kill.check():
            record["action"] = "skip_kill_switch"
            record["kill"] = self.kill.status()
            self._log_json(self.dec_fh, record)
            return
        equity = self.cash + self.qty * mid
        day = datetime.fromtimestamp(now_sec, timezone.utc).date().isoformat()
        if day != self.current_day:
            self.current_day = day
            self.day_realized_start = equity
            self.turnover_today = 0.0
        day_loss = equity / self.day_realized_start - 1.0 if self.day_realized_start > 0 else 0.0
        if self.risk.max_daily_loss_pct > 0 and day_loss <= -self.risk.max_daily_loss_pct / 100.0:
            self.kill.trip(f"daily loss {day_loss:.2%} at {now_sec}")
            record["action"] = "skip_daily_loss_halt"
            self._log_json(self.dec_fh, record)
            return
        if now_sec - self.last_trade_second < self.risk.cooldown_seconds:
            record["action"] = "skip_cooldown"
            self._log_json(self.dec_fh, record)
            return
        pred_bps = pred * 1e4
        hurdle = max(self.risk.min_edge_bps, self.cost.round_trip_cost_bps(spread_bps))
        record["hurdle_bps"] = hurdle
        if abs(pred_bps) < hurdle:
            record["action"] = "flat_below_hurdle"
            self._log_json(self.dec_fh, record)
            return
        conviction = min(1.0, abs(pred_bps) / hurdle)
        direction = 1 if pred_bps > 0 else (-1 if self.profile.allow_short else 0)
        if direction == 0:
            record["action"] = "skip_no_short_spot"
            self._log_json(self.dec_fh, record)
            return
        leverage = self.profile.validate_leverage(self.risk.leverage)
        target_notional = conviction * equity * leverage * self.risk.position_fraction
        target_qty = direction * target_notional / mid
        if self.profile.allow_short is False:
            target_qty = max(0.0, target_qty)
        delta = target_qty - self.qty
        order_notional = abs(delta) * mid
        is_close = target_qty == 0.0 and self.qty != 0.0
        if delta == 0 or (not is_close and order_notional < self.rules.min_notional):
            record["action"] = "skip_min_notional_or_zero"
            self._log_json(self.dec_fh, record)
            return
        if (abs(target_qty) > abs(self.qty)) and self.risk.max_turnover_per_day > 0:
            if self.turnover_today + order_notional > self.risk.max_turnover_per_day * self.risk.starting_capital:
                record["action"] = "skip_turnover_cap"
                self._log_json(self.dec_fh, record)
                return
        self._paper_execute(delta, mid, now_sec, record)

    def _paper_execute(self, delta: float, mid: float, now_sec: int, record: dict) -> None:
        slip = self.cost.slippage_bps / 1e4
        book = self.book or {}
        bid = book.get("bid", mid * (1 - 0.00005))
        ask = book.get("ask", mid * (1 + 0.00005))
        if delta > 0:
            px = ask * (1 + slip)
            side = "buy"
        else:
            px = bid * (1 - slip)
            side = "sell"
        qty = abs(delta)
        notional = qty * px
        fee = notional * self.cost.taker_fee_bps / 1e4
        self.cash -= delta * px
        self.cash -= fee
        self.turnover_today += notional
        self.last_trade_second = now_sec
        self.n_fills += 1
        if self.qty * delta < 0 and self.qty != 0:  # reduction: realize P/L
            matched = min(abs(self.qty), qty)
            pnl = (px - self.entry_price) * np.sign(self.qty) * matched
            self.realized_pnl += pnl - fee
        if self.qty * delta > 0:
            total = self.qty + delta
            self.entry_price = (self.entry_price * abs(self.qty) + px * qty) / abs(total) if total else px
        elif self.qty != 0 and abs(delta) >= abs(self.qty):
            self.entry_price = px
        self.qty += delta
        record["action"] = "paper_fill"
        record.update({"side": side, "qty": qty, "exec_price": px, "notional": notional,
                       "fee": fee, "position_after": self.qty, "cash_after": self.cash})
        self._log_json(self.ord_fh, {"kind": "order", "second": now_sec, "side": side,
                                     "qty": qty, "type": "market", "reason": "model_signal"})
        self._log_json(self.fil_fh, {"kind": "fill", "second": now_sec, "side": side,
                                     "qty": qty, "price": px, "fee": fee,
                                     "position_after": self.qty, "cash_after": self.cash})
        self._log_json(self.dec_fh, record)
        self._save_state()

    # ---------------------------------------------------------------- health
    def _heartbeat(self) -> None:
        book = self.book or {}
        mid = (book.get("bid", np.nan) + book.get("ask", np.nan)) / 2 if book else np.nan
        equity = self.cash + self.qty * (mid if np.isfinite(mid) else 0.0)
        health = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "events_count": self.events_count,
            "seconds_buffered": len(self.rolling),
            "book_age_seconds": time.monotonic() - self.last_event_monotonic,
            "book_valid": self.book is not None and (time.monotonic() - self.last_event_monotonic) < self.risk.stale_quote_max_seconds,
            "decisions": self.n_decisions, "skipped": self.n_skipped, "fills": self.n_fills,
            "position_qty": self.qty, "cash": self.cash, "equity": equity,
            "realized_pnl": self.realized_pnl,
            "last_prediction_bps": None if self.last_prediction is None else self.last_prediction * 1e4,
            "kill_switch": self.kill.status(),
            "model": type(self.model).__name__,
            "started_utc": self.started_utc,
        }
        (self.run_dir / "health.json").write_text(json.dumps(health, indent=2, default=str))
        self._log_json(self.eq_fh, {"kind": "equity", **health})

    def _log_json(self, fh, obj: dict) -> None:
        try:
            fh.write(json.dumps(obj, separators=(",", ":"), default=str) + "\n")
            fh.flush()
        except (OSError, ValueError):
            pass

    # ------------------------------------------------------------------- run
    async def run(self) -> dict:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, self._stop.set)
        url = self._ws_url()
        self.log.info("paper trading %s %s -> %s (NO real orders; paper only)",
                      self.profile.name, self.symbol, self.run_dir)
        deadline = time.monotonic() + self.duration if self.duration > 0 else None
        backoff = 1.0
        ticker = asyncio.create_task(self._ticker_loop(deadline))
        try:
            while not self._stop.is_set() and (deadline is None or time.monotonic() < deadline):
                try:
                    async with websockets.connect(url, ping_interval=20, ping_timeout=20,
                                                  close_timeout=5, open_timeout=10,
                                                  max_size=None) as ws:
                        self.log.info("connected: %s", url)
                        backoff = 1.0
                        self.book = None          # book state invalid until fresh quote
                        self.last_update_id = -1  # prevent duplicate actions post-reconnect
                        watchdog_deadline = time.monotonic() + 30.0
                        while not self._stop.is_set():
                            timeout = 1.0
                            if deadline is not None:
                                timeout = max(0.05, min(1.0, deadline - time.monotonic()))
                                if time.monotonic() >= deadline:
                                    self._stop.set()
                                    break
                            try:
                                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                                self._handle_message(raw)
                                watchdog_deadline = time.monotonic() + 30.0
                            except asyncio.TimeoutError:
                                pass  # boundaries handled by the ticker task
                            if time.monotonic() > watchdog_deadline:
                                self.log.warning("stream silent >30s; forcing reconnect")
                                break
                        if self._stop.is_set():
                            break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    self.log.warning("stream error: %s; reconnect in %.1fs", exc, backoff)
                    await asyncio.sleep(backoff)
                    backoff = min(30.0, backoff * 2)
        finally:
            ticker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ticker
            await self.shutdown()
        return self._summary()

    async def _ticker_loop(self, deadline: float | None) -> None:
        """Second-resolution boundary processing independent of event flow:
        with ~200 events/s the recv loop never times out, so decisions must be
        driven by this timer, not by recv idleness."""
        while not self._stop.is_set():
            await asyncio.sleep(1.0)
            if deadline is not None and time.monotonic() >= deadline:
                self._stop.set()
                return
            try:
                self._boundary_tick()
            except Exception as exc:  # noqa: BLE001 - never kill the ticker
                self.log.error("boundary tick failed: %s", exc)

    def _boundary_tick(self) -> None:
        """Called once per second by the ticker, independent of event flow.

        Decisions fire whenever a second has been COMPLETED (appended to the
        rolling frame), no matter whether events or the ticker drove the
        rollover. Missed boundaries are filled as empty gap buckets first.
        """
        now_sec = int(time.time())
        if self.bucket_second < 0:
            self.bucket_second = now_sec
            return
        missed = now_sec - self.bucket_second
        if missed > 3600:
            # clock jump / long stall: resync instead of looping for hours
            self.log.warning("boundary gap of %ds detected; resyncing bucket clock", missed)
            self.bucket_second = now_sec - 1
        while self.bucket_second < now_sec:
            self._rollover(self.bucket_second + 1)
        # Safety-critical: poll the kill switch EVERY second, not only when a
        # trade decision is due (warmup periods must still honour the switch).
        if self.kill.check() and not getattr(self, "_kill_logged", False):
            self._kill_logged = True
            self.log.warning("KILL SWITCH TRIPPED: %s — trading halted; restart required",
                             self.kill.reason)
        if self.rolling and self.rolling[-1]["second"] > self.last_decided_second:
            self.last_decided_second = self.rolling[-1]["second"]
            self._decide_and_trade()

    async def shutdown(self) -> None:
        self._heartbeat()
        self._save_state()
        for fh in (self.obs_fh, self.dec_fh, self.ord_fh, self.fil_fh, self.eq_fh):
            with contextlib.suppress(OSError, ValueError):
                fh.flush()
                fh.close()
        self.log.info("paper session closed: decisions=%d fills=%d realized_pnl=%.2f",
                      self.n_decisions, self.n_fills, self.realized_pnl)

    def _summary(self) -> dict:
        book = self.book or {}
        mid = (book.get("bid", 0.0) + book.get("ask", 0.0)) / 2
        return {"events": self.events_count, "decisions": self.n_decisions,
                "skipped": self.n_skipped, "fills": self.n_fills,
                "final_position": self.qty, "cash": self.cash,
                "equity": self.cash + self.qty * mid,
                "realized_pnl": self.realized_pnl,
                "kill_switch": self.kill.status(),
                "run_dir": str(self.run_dir)}


async def run_paper_session(profile: MarketProfile, model: BaseModel, cost: CostConfig,
                            risk: RiskConfig, run_dir: str | Path,
                            duration_seconds: float = 0, heartbeat_seconds: int = 60,
                            rules: SymbolRules | None = None) -> dict:
    if rules is None:
        rules = fetch_symbol_rules(profile.name, profile.symbol)
    engine = PaperTradingEngine(profile, model, cost, risk, run_dir,
                                duration_seconds=duration_seconds,
                                heartbeat_seconds=heartbeat_seconds, rules=rules)
    return await engine.run()
