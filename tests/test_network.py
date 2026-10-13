"""Optional network tests against real Binance endpoints.

Run explicitly:  python -m pytest -q -m network
These verify archive availability facts and REST reachability. The futures
REST endpoint (fapi.binance.com) geo-blocks many cloud IPs with HTTP 418/451;
that is an environmental constraint, not a code failure -> skip with reason.
"""
import pytest
import requests

pytestmark = pytest.mark.network

ARCHIVE = "https://data.binance.vision/"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"


def test_archive_head_futures_bookticker_available():
    url = (ARCHIVE + "data/futures/um/daily/bookTicker/BTCUSDT/"
           "BTCUSDT-bookTicker-2024-03-25.zip")
    r = requests.head(url, timeout=20, allow_redirects=True)
    assert r.status_code == 200
    assert int(r.headers.get("Content-Length", 0)) > 50_000_000


def test_archive_checksum_file_available():
    url = (ARCHIVE + "data/futures/um/daily/aggTrades/BTCUSDT/"
           "BTCUSDT-aggTrades-2024-03-25.zip.CHECKSUM")
    r = requests.get(url, timeout=20)
    assert r.status_code == 200
    digest = r.text.split()[0]
    assert len(digest) in (32, 64)


def test_s3_listing_spot_aggtrades_btcfdusd():
    r = requests.get(S3, params={"delimiter": "/", "prefix":
                                 "data/spot/daily/aggTrades/BTCFDUSD/"}, timeout=20)
    assert r.status_code == 200
    from hpl1.binance_archive import parse_s3_listing

    keys, _ = parse_s3_listing(r.text)
    zips = [k for k in keys if k.key.endswith(".zip")]
    assert len(zips) > 30  # spot aggTrades history exists for BTCFDUSD


def test_s3_listing_spot_has_no_book_data():
    """Verified fact: spot publishes NO bookTicker history — the strongest
    valid research path for spot is trade-flow only."""
    r = requests.get(S3, params={"delimiter": "/", "prefix":
                                 "data/spot/daily/bookTicker/BTCFDUSD/"}, timeout=20)
    assert r.status_code == 200
    from hpl1.binance_archive import parse_s3_listing

    keys, _ = parse_s3_listing(r.text)
    assert not [k for k in keys if k.key.endswith(".zip")]


def test_spot_rest_exchange_info_reachable():
    r = requests.get("https://api.binance.com/api/v3/exchangeInfo",
                     params={"symbol": "BTCFDUSD"}, timeout=20)
    if r.status_code in (418, 451):
        pytest.skip("spot REST geo-blocked from this network (HTTP %d)" % r.status_code)
    assert r.status_code == 200
    filters = {f["filterType"]: f for f in r.json()["symbols"][0]["filters"]}
    assert "tickSize" in filters["PRICE_FILTER"]


def test_futures_rest_exchange_info_or_geoblock():
    r = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo",
                     params={"symbol": "BTCUSDT"}, timeout=20)
    if r.status_code in (418, 451):
        pytest.skip("fapi REST geo-blocked from this network (HTTP %d) — "
                    "exchange_meta falls back to documented defaults" % r.status_code)
    assert r.status_code == 200
    filters = {f["filterType"]: f for f in r.json()["symbols"][0]["filters"]}
    assert "tickSize" in filters["PRICE_FILTER"]


def test_futures_bookticker_discontinued_after_2024_03():
    """Verified fact: futures bookTicker daily archives END at 2024-03-30."""
    url = (ARCHIVE + "data/futures/um/daily/bookTicker/BTCUSDT/"
           "BTCUSDT-bookTicker-2024-04-15.zip")
    r = requests.head(url, timeout=20, allow_redirects=True)
    assert r.status_code in (404, 403), (
        "expected futures bookTicker to be unavailable after 2024-03-30; "
        "if this now exists, update docs/DOWNLOAD_RECORD.md")
