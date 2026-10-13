"""Market profiles, cost config, YAML config validation."""
import pytest

from hpl1.config import (
    CostConfig,
    ResearchConfig,
    RiskConfig,
    build_research_config,
    load_config,
)
from hpl1.markets import MARKETS, get_market, list_markets


def test_profiles_never_mixed():
    assert set(MARKETS) == {"spot", "futures"}
    assert MARKETS["spot"].symbol == "BTCFDUSD"
    assert MARKETS["futures"].symbol == "BTCUSDT"
    assert MARKETS["spot"].market_type != MARKETS["futures"].market_type


def test_spot_profile_forbids_short_and_leverage():
    spot = get_market("spot")
    assert spot.allow_short is False
    assert spot.leverage_cap == 1.0
    assert spot.has_funding is False
    with pytest.raises(ValueError):
        spot.validate_leverage(0.0)
    assert spot.validate_leverage(5.0) == 1.0  # capped


def test_futures_profile_supports_2x():
    fut = get_market("futures")
    assert fut.allow_short is True
    assert fut.validate_leverage(2.0) == 2.0
    assert fut.has_funding is True
    assert fut.funding_interval_hours == 8


def test_unknown_market_rejected():
    with pytest.raises(ValueError):
        get_market("options")


def test_zero_fee_is_explicit_scenario_not_default():
    default = CostConfig()
    assert default.taker_fee_bps == 5.0  # never silently zero
    zero = CostConfig(fee_scenario="zero")
    assert zero.taker_fee_bps == 0.0 and zero.maker_fee_bps == 0.0
    assert CostConfig(fee_scenario="user").taker_fee_bps == 5.0


def test_cost_config_rejects_negative_fees():
    with pytest.raises(ValueError):
        CostConfig(taker_fee_bps=-1)
    with pytest.raises(ValueError):
        CostConfig(fee_scenario="nonsense")


def test_round_trip_cost_includes_both_sides_and_spread():
    c = CostConfig(taker_fee_bps=5.0, slippage_bps=1.0)
    assert c.round_trip_cost_bps(spread_bps=2.0) == pytest.approx(2 * 6.0 + 2.0)
    assert CostConfig(use_spread=False).round_trip_cost_bps(10.0) == pytest.approx(12.0)


def test_research_config_validation():
    with pytest.raises(ValueError):
        ResearchConfig(train_fraction=0.9, validation_fraction=0.9)
    with pytest.raises(ValueError):
        ResearchConfig(start_date="2024-3-25")
    cfg = ResearchConfig(market="spot", data_types=["fundingRate", "aggTrades"])
    assert "fundingRate" not in cfg.data_types  # auto-excluded: futures-only dataset
    assert cfg.symbol == "BTCFDUSD"
    assert cfg.profile.market_type == "spot"


def test_yaml_config_load_and_rejects_live_trading(tmp_path):
    good = tmp_path / "research.yaml"
    good.write_text("""
research:
  market: futures
  start_date: "2024-03-25"
  end_date: "2024-03-29"
costs:
  taker_fee_bps: 4.0
  slippage_bps: 0.5
risk:
  leverage: 2.0
""")
    research, cost, risk = build_research_config(load_config(good))
    assert research.market == "futures"
    assert cost.taker_fee_bps == 4.0
    assert risk.leverage == 2.0
    bad = tmp_path / "evil.yaml"
    bad.write_text("live_trading_enabled: true\n")
    with pytest.raises(ValueError, match="live_trading_enabled"):
        load_config(bad)
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_risk_config_defaults_are_conservative():
    r = RiskConfig()
    assert r.max_daily_loss_pct == 3.0
    assert r.max_drawdown_pct == 10.0
    assert r.cooldown_seconds >= 1
    with pytest.raises(ValueError):
        RiskConfig(starting_capital=0)
    with pytest.raises(ValueError):
        RiskConfig(position_fraction=1.5)


def test_list_markets_shape():
    rows = list_markets()
    assert len(rows) == 2
    assert all({"name", "symbol", "allow_short", "leverage_cap"} <= set(r) for r in rows)
