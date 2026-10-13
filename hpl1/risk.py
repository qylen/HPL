"""Risk limits and kill switches shared by the backtester and paper engine.

Kill switch semantics:
- MANUAL: an operator creates the kill-switch file (default `run/kill_switch`).
  Every decision loop checks it; on detection trading halts immediately and
  the halt is logged. Removing the file does NOT resume trading in the current
  process: a restart is required (documented in the runbook).
- AUTOMATIC: stale data beyond the configured threshold, daily-loss limit, or
  maximum drawdown trigger the same halted state.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class KillSwitch:
    path: str | Path = "run/kill_switch"
    tripped: bool = False
    tripped_at: str | None = None
    reason: str | None = None
    manual_file_found: bool = False

    def check(self) -> bool:
        """Return True if trading must halt (idempotent)."""
        if self.tripped:
            return True
        if Path(self.path).exists():
            self.tripped = True
            self.manual_file_found = True
            self.tripped_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.reason = f"manual kill-switch file present: {self.path}"
            return True
        return False

    def trip(self, reason: str) -> None:
        if not self.tripped:
            self.tripped = True
            self.tripped_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.reason = reason

    def status(self) -> dict:
        return {"tripped": self.tripped, "reason": self.reason,
                "tripped_at": self.tripped_at, "manual_file_found": self.manual_file_found}


@dataclass
class TradingLimits:
    """Hard limits enforced before any order (paper or future live adapter)."""
    max_position_qty: float = 0.1       # absolute cap in base units
    max_notional: float = 25_000.0      # absolute notional cap
    max_leverage: float = 2.0
    min_notional: float = 100.0
    max_daily_loss_pct: float = 3.0
    max_drawdown_pct: float = 10.0
    cooldown_seconds: int = 10
    max_turnover_per_day: float = 10.0
    stale_data_max_seconds: float = 30.0

    def validate_order(self, qty: float, price: float, equity: float) -> tuple[bool, str]:
        notional = abs(qty) * price
        if notional < self.min_notional:
            return False, f"order notional {notional:.2f} < min {self.min_notional}"
        if abs(qty) > self.max_position_qty:
            return False, f"qty {abs(qty)} > max {self.max_position_qty}"
        if notional > self.max_notional:
            return False, f"notional {notional:.2f} > max {self.max_notional}"
        if equity > 0 and notional > equity * self.max_leverage:
            return False, f"notional {notional:.2f} exceeds leverage cap {self.max_leverage}x equity"
        if not (price > 0):
            return False, "invalid price"
        return True, "ok"

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
