"""Market profiles: spot BTCFDUSD and USD-M perpetual BTCUSDT.

Spot and futures are never mixed into one training dataset. Each profile fixes
the trading constraints that features, labels, fee calculations, and execution
assumptions must respect.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketProfile:
    name: str                      # "spot" | "futures"
    market_type: str               # "spot" | "usdm_perp"
    symbol: str                    # canonical uppercase symbol, e.g. BTCFDUSD
    allow_short: bool              # spot: False (no silent short selling)
    leverage_cap: float            # spot: 1.0; futures: configurable, capped
    has_funding: bool              # futures only
    funding_interval_hours: int    # 8 for USD-M perps (typical)
    default_min_notional: float    # exchange minimum order notional (fallback)
    base_data_path: str            # archive path segment under data.binance.vision
    ws_base: str                   # public websocket base url
    rest_base: str                 # public rest base url

    @property
    def is_derivative(self) -> bool:
        return self.market_type != "spot"

    def validate_leverage(self, leverage: float) -> float:
        if leverage <= 0:
            raise ValueError(f"leverage must be positive, got {leverage}")
        capped = min(leverage, self.leverage_cap)
        return capped


SPOT_BTCFDUSD = MarketProfile(
    name="spot",
    market_type="spot",
    symbol="BTCFDUSD",
    allow_short=False,
    leverage_cap=1.0,
    has_funding=False,
    funding_interval_hours=0,
    default_min_notional=5.0,
    base_data_path="data/spot",
    ws_base="wss://stream.binance.com:9443",
    rest_base="https://api.binance.com",
)

FUTURES_BTCUSDT = MarketProfile(
    name="futures",
    market_type="usdm_perp",
    symbol="BTCUSDT",
    allow_short=True,
    leverage_cap=2.0,   # Profile B: support configurable leverage, including 2x
    has_funding=True,
    funding_interval_hours=8,
    default_min_notional=100.0,
    base_data_path="data/futures/um",
    ws_base="wss://fstream.binance.com",
    rest_base="https://fapi.binance.com",
)

MARKETS: dict[str, MarketProfile] = {
    "spot": SPOT_BTCFDUSD,
    "futures": FUTURES_BTCUSDT,
}


def get_market(name: str) -> MarketProfile:
    key = (name or "").strip().lower()
    if key not in MARKETS:
        raise ValueError(
            f"unknown market '{name}'. Supported: {sorted(MARKETS)} "
            "(spot=BTCFDUSD, futures=USD-M BTCUSDT perpetual)"
        )
    return MARKETS[key]


def list_markets() -> list[dict]:
    return [
        {
            "name": p.name,
            "market_type": p.market_type,
            "symbol": p.symbol,
            "allow_short": p.allow_short,
            "leverage_cap": p.leverage_cap,
            "has_funding": p.has_funding,
            "funding_interval_hours": p.funding_interval_hours,
            "default_min_notional": p.default_min_notional,
        }
        for p in MARKETS.values()
    ]
