import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "network: tests hitting real Binance endpoints (opt-in)")
    config.addinivalue_line(
        "markers", "slow: tests taking more than a few seconds")


def pytest_collection_modifyitems(config, items):
    if config.getoption("-m") == "network":
        return
    skip_network = pytest.mark.skip(reason="network test; run with -m network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)
