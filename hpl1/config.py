"""Validated configuration for research and paper trading.

Exchange-specific settings are kept separate from strategy parameters. All
configs validate loudly: a malformed configuration must never silently fall
back to a default that could activate trading-like behaviour.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .markets import get_market

DEFAULT_SEED = 13


@dataclass
class CostConfig:
    # Per-side taker commission in basis points. Zero fees must be an EXPLICIT
    # scenario (fee_scenario="zero"), never a silent default.
    taker_fee_bps: float = 5.0
    maker_fee_bps: float = 2.0
    slippage_bps: float = 1.0          # additional adverse price movement per side
    use_spread: bool = True             # cross the observed quoted spread on market orders
    fee_scenario: str = "user"          # "user" | "zero" | "promo" (label only, must be explicit)

    def __post_init__(self) -> None:
        for name in ("taker_fee_bps", "maker_fee_bps", "slippage_bps"):
            v = float(getattr(self, name))
            if v < 0:
                raise ValueError(f"{name} must be >= 0, got {v}")
        if self.fee_scenario == "zero":
            self.taker_fee_bps = 0.0
            self.maker_fee_bps = 0.0
        if self.fee_scenario not in ("user", "zero", "promo"):
            raise ValueError("fee_scenario must be 'user', 'zero' or 'promo'")

    def round_trip_cost_bps(self, spread_bps: float) -> float:
        """Approximate round-trip cost: both sides of fees+slippage (+full spread).

        Spread is charged through execution prices (buy at ask, sell at bid);
        this helper is used for the entry threshold only, NOT as a separate
        cost line, so spread+slippage are never double-counted.
        """
        spread = float(spread_bps) if self.use_spread else 0.0
        return 2.0 * (self.taker_fee_bps + self.slippage_bps) + spread


@dataclass
class RiskConfig:
    starting_capital: float = 10_000.0
    leverage: float = 1.0               # capped by market profile (spot: 1.0)
    position_fraction: float = 0.5      # fraction of max notional actually deployed
    max_daily_loss_pct: float = 3.0     # halt for the rest of the UTC day
    max_drawdown_pct: float = 10.0      # permanent halt
    cooldown_seconds: int = 10          # no trade re-entry right after a trade
    max_turnover_per_day: float = 10.0  # in multiples of starting equity (notional traded)
    min_edge_bps: float = 0.5           # never trade below this predicted edge
    stale_quote_max_seconds: float = 30.0
    kill_switch_file: str = "run/kill_switch"

    def __post_init__(self) -> None:
        if self.starting_capital <= 0:
            raise ValueError("starting_capital must be > 0")
        if not 0 < self.position_fraction <= 1.0:
            raise ValueError("position_fraction must be in (0, 1]")
        if self.leverage <= 0:
            raise ValueError("leverage must be > 0")


@dataclass
class ResearchConfig:
    market: str = "futures"
    symbol: str | None = None            # default: profile symbol
    start_date: str = "2024-03-25"
    end_date: str = "2024-03-29"
    data_types: list[str] = field(default_factory=lambda: ["bookTicker", "aggTrades", "fundingRate"])
    train_fraction: float = 0.60
    validation_fraction: float = 0.20    # remainder is the untouched final test
    purge_seconds: int = 1               # labels span 1 second ahead of features
    embargo_seconds: int = 30            # extra gap after validation before test
    models: list[str] = field(default_factory=lambda: ["zero", "ridge", "logistic", "hist", "law", "regime"])
    random_seed: int = DEFAULT_SEED
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    artifacts_dir: str = "artifacts/research"
    large_download_allow_mb: float = 500.0   # require --yes above this
    max_download_concurrency: int = 4

    def __post_init__(self) -> None:
        profile = get_market(self.market)
        if self.symbol is None:
            self.symbol = profile.symbol
        if not (0 < self.train_fraction < 1 and 0 < self.validation_fraction < 1
                and self.train_fraction + self.validation_fraction < 1):
            raise ValueError("train/validation fractions must be positive and sum < 1")
        for d in (self.start_date, self.end_date):
            _parse_date(d)
        if self.start_date > self.end_date:
            raise ValueError("start_date must be <= end_date")
        allowed = {"bookTicker", "aggTrades", "trades", "klines", "fundingRate", "bookDepth"}
        bad = set(self.data_types) - allowed
        if bad:
            raise ValueError(f"unknown data types {sorted(bad)}; allowed {sorted(allowed)}")
        if profile.market_type == "spot":
            # fundingRate/bookDepth are futures-only datasets; spot publishes NO
            # book history at all (verified against the archive) — trade-only.
            self.data_types = [d for d in self.data_types
                               if d not in ("fundingRate", "bookDepth", "bookTicker")] or ["aggTrades"]
        if self.max_download_concurrency < 1:
            raise ValueError("max_download_concurrency must be >= 1")

    @property
    def profile(self):
        return get_market(self.market)


def _parse_date(s: str) -> str:
    import datetime as _dt
    try:
        _dt.date.fromisoformat(s)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"date must be YYYY-MM-DD, got {s!r}") from exc
    return s


@dataclass
class PaperConfig:
    market: str = "futures"
    symbol: str | None = None
    model_path: str = "artifacts/research/best_model.joblib"
    run_dir: str = "data/paper"
    duration_seconds: float = 0          # 0 = until Ctrl+C
    heartbeat_seconds: int = 60
    health_file: str = "run/health.json"
    dry_run: bool = True                 # paper only; no real orders exist in this codebase at all


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    text = p.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping, got {type(data).__name__}")
    if data.get("live_trading_enabled"):
        raise ValueError(
            "live_trading_enabled=true is rejected: this codebase has NO live-order "
            "execution client; only research and paper trading are supported."
        )
    return data


def build_research_config(data: dict[str, Any]) -> tuple[ResearchConfig, CostConfig, RiskConfig]:
    research = ResearchConfig(**{
        k: v for k, v in (data.get("research") or {}).items() if k in ResearchConfig.__dataclass_fields__
    })
    cost = CostConfig(**{
        k: v for k, v in (data.get("costs") or {}).items() if k in CostConfig.__dataclass_fields__
    })
    risk = RiskConfig(**{
        k: v for k, v in (data.get("risk") or {}).items() if k in RiskConfig.__dataclass_fields__
    })
    risk.leverage = research.profile.validate_leverage(risk.leverage)
    if not research.profile.allow_short and cost.fee_scenario != "zero":
        pass  # fees on spot are still per-side commissions; nothing special to do
    return research, cost, risk


def config_manifest(cfgs: tuple[ResearchConfig, CostConfig, RiskConfig]) -> dict[str, Any]:
    research, cost, risk = cfgs
    return {
        "research": asdict(research),
        "costs": asdict(cost),
        "risk": asdict(risk),
        "sha": None,  # filled by caller with code revision
    }


def dump_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
