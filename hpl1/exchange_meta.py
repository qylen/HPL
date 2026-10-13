"""Exchange trading-rule metadata (tick size, quantity step, min notional).

Tries the official public REST endpoints first. fapi.binance.com often returns
HTTP 418/451 to cloud IPs (geo-block); in that case we fall back to documented
exchange-realistic defaults. Defaults are ASSUMPTIONS and are overridable via
config; every object records its source.
"""
from __future__ import annotations

from dataclasses import dataclass

import requests

# Verified-against-docs fallbacks (source: Binance spot/futures exchangeInfo,
# BTCUSDT / BTCFDUSD filters, as of the 2024 research window).
DEFAULTS = {
    ("futures", "BTCUSDT"): {"tick_size": "0.10", "step_size": "0.001", "min_notional": "100"},
    ("spot", "BTCFDUSD"): {"tick_size": "0.01", "step_size": "0.00001", "min_notional": "5"},
}


@dataclass
class SymbolRules:
    market: str
    symbol: str
    tick_size: str
    step_size: str
    min_notional: float
    source: str  # "rest" | "fallback" | "config"

    def to_dict(self) -> dict:
        return {
            "market": self.market, "symbol": self.symbol,
            "tick_size": self.tick_size, "step_size": self.step_size,
            "min_notional": self.min_notional, "source": self.source,
        }


def fetch_symbol_rules(market: str, symbol: str, session: requests.Session | None = None,
                       timeout: float = 10.0) -> SymbolRules:
    """Fetch exchange trading rules; fall back to documented defaults on block/failure."""
    from .markets import get_market

    profile = get_market(market)
    symbol = symbol.upper()
    sess = session or requests.Session()
    url = f"{profile.rest_base}/{'api/v3' if market == 'spot' else 'fapi/v1'}/exchangeInfo"
    try:
        resp = sess.get(url, params={"symbol": symbol}, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            filters = {f["filterType"]: f for f in data.get("symbols", [{}])[0].get("filters", [])}
            price = filters.get("PRICE_FILTER", {}).get("tickSize")
            lot = filters.get("LOT_SIZE", {}).get("stepSize")
            if market == "futures":
                min_notional = float(filters.get("MIN_NOTIONAL", {}).get("notional", 100))
            else:
                min_notional = float(filters.get("NOTIONAL", {}).get("minNotional", 5))
            if price and lot:
                return SymbolRules(market, symbol, price, lot, min_notional, source="rest")
    except (requests.RequestException, ValueError, KeyError, IndexError):
        pass
    d = DEFAULTS.get((market, symbol))
    if d is None:
        raise ValueError(f"no trading-rule defaults for {market}/{symbol}; add them to config")
    return SymbolRules(market, symbol, d["tick_size"], d["step_size"], float(d["min_notional"]),
                       source="fallback (REST unavailable or geo-blocked)")
