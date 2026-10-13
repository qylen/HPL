"""Archive URL building, S3 listing parse, downloader retry/checksum/resume/guard."""
import io
import zipfile

import pytest

from hpl1.binance_archive import (
    archive_key,
    estimate_download,
    iter_dates,
    parse_s3_listing,
)
from hpl1.downloader import download_one, download_range, sha256_of_file

S3_XML = """<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>data.binance.vision</Name>
  <IsTruncated>false</IsTruncated>
  <Contents>
    <Key>data/futures/um/daily/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2024-03-25.zip</Key>
    <Size>150000000</Size>
  </Contents>
  <Contents>
    <Key>data/futures/um/daily/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2024-03-26.zip</Key>
    <Size>151000000</Size>
  </Contents>
  <Contents>
    <Key>data/spot/daily/aggTrades/BTCFDUSD/BTCFDUSD-aggTrades-2024-03-25.zip</Key>
    <Size>40000000</Size>
  </Contents>
  <Contents>
    <Key>data/futures/um/monthly/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2024-03.zip</Key>
    <Size>4000000000</Size>
  </Contents>
</ListBucketResult>"""


def test_archive_key_patterns():
    k = archive_key("futures", "BTCUSDT", "bookTicker", "2024-03-25")
    assert k.key == "data/futures/um/daily/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2024-03-25.zip"
    assert k.url.endswith(".zip") and k.url.startswith("https://data.binance.vision/")
    assert k.checksum_url == k.url + ".CHECKSUM"
    s = archive_key("spot", "BTCFDUSD", "aggTrades", "2024-03-25")
    assert s.key == "data/spot/daily/aggTrades/BTCFDUSD/BTCFDUSD-aggTrades-2024-03-25.zip"
    m = archive_key("futures", "BTCUSDT", "bookTicker", "2024-03", period="monthly")
    assert m.key == "data/futures/um/monthly/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2024-03.zip"


def test_archive_key_rejects_futures_dtype_on_spot():
    with pytest.raises(ValueError):
        archive_key("spot", "BTCFDUSD", "fundingRate", "2024-03-25")


def test_parse_s3_listing():
    keys, marker = parse_s3_listing(S3_XML)
    assert marker is None
    assert len(keys) == 4
    fut_daily = next(k for k in keys if k.market == "futures" and k.period == "daily")
    assert fut_daily.dtype == "bookTicker" and fut_daily.size == 150000000
    spot = next(k for k in keys if k.market == "spot")
    assert spot.dtype == "aggTrades" and spot.size == 40000000
    monthly = [k for k in keys if k.period == "monthly"]
    assert monthly and monthly[0].date_label == "2024-03" and monthly[0].size == 4000000000


def test_iter_dates_inclusive_and_ordered():
    days = list(iter_dates("2024-03-25", "2024-03-29"))
    assert [d.isoformat() for d in days] == ["2024-03-25", "2024-03-26", "2024-03-27",
                                             "2024-03-28", "2024-03-29"]
    with pytest.raises(ValueError):
        list(iter_dates("2024-03-29", "2024-03-25"))


class FakeResponse:
    def __init__(self, status=200, content=b"", headers=None, text=""):
        self.status_code = status
        self._content = content
        self.headers = headers or {}
        self.text = text

    def iter_content(self, chunk_size=1 << 20):
        yield self._content

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code}")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeSession:
    """Session double: serves checksum + file bytes with scripted failures."""

    def __init__(self, file_bytes, sha, failures=0, statuses=None):
        self.file_bytes = file_bytes
        self.sha = sha
        self.failures_left = failures
        self.get_calls = 0
        self.statuses = statuses or {}

    def get(self, url, timeout=None, headers=None, stream=True):
        self.get_calls += 1
        if url.endswith(".CHECKSUM"):
            return FakeResponse(content=b"", text=f"{self.sha}  file.zip")
        remaining = self.failures_left > 0
        if remaining:
            self.failures_left -= 1
            return FakeResponse(status=500)
        rng = "bytes=%d-" % int(headers.get("Range", "bytes=-").split("=")[1].split("-")[0]) \
            if headers and "Range" in headers else None
        if rng and self.statuses.get("support_range", True):
            start = int(rng.split("=")[1].split("-")[0])
            return FakeResponse(status=206, content=self.file_bytes[start:])
        return FakeResponse(status=200, content=self.file_bytes)


def _make_zip(seed: int = 1) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("X.csv", "update_id,best_bid_price,best_bid_qty,best_ask_price,"
                             "best_ask_qty,transaction_time,event_time\n1,2,3,4,5,6,7\n" * seed)
    return buf.getvalue()


def test_download_one_success_with_checksum(tmp_path):
    import hashlib

    data = _make_zip(3)
    sha = hashlib.sha256(data).hexdigest()
    key = archive_key("futures", "BTCUSDT", "bookTicker", "2024-03-25")
    sess = FakeSession(data, sha)
    result = download_one(key, tmp_path, sess, max_retries=2)
    assert result.status == "ok"
    assert result.checksum_verified is True
    assert (tmp_path / key.key.split("/")[-1]).read_bytes() == data
    assert not (tmp_path / "X.zip.part").exists()


def test_download_one_retries_then_succeeds(tmp_path):
    import hashlib

    data = _make_zip(2)
    sha = hashlib.sha256(data).hexdigest()
    key = archive_key("futures", "BTCUSDT", "bookTicker", "2024-03-25")
    sess = FakeSession(data, sha, failures=2)
    result = download_one(key, tmp_path, sess, max_retries=5)
    assert result.status == "ok"
    assert result.attempts == 3  # two failures then success


def test_download_one_checksum_mismatch_fails(tmp_path):
    import hashlib

    data = _make_zip(2)
    wrong = hashlib.sha256(b"other").hexdigest()
    key = archive_key("futures", "BTCUSDT", "bookTicker", "2024-03-25")
    sess = FakeSession(data, wrong)
    result = download_one(key, tmp_path, sess, max_retries=2)
    assert result.status == "failed"
    assert "checksum mismatch" in (result.error or "")


def test_download_one_missing_checksum_reports_nocrc(tmp_path):
    key = archive_key("futures", "BTCUSDT", "bookTicker", "2024-03-25")

    class NoCrc(FakeSession):
        def get(self, url, **kw):
            if url.endswith(".CHECKSUM"):
                return FakeResponse(status=404)
            return super().get(url, **kw)

    sess = NoCrc(_make_zip(1), "x")
    result = download_one(key, tmp_path, sess, max_retries=1)
    assert result.status == "checksum_missing"
    assert result.checksum_verified is False


def test_download_one_skips_existing_verified(tmp_path):
    import hashlib

    data = _make_zip(1)
    sha = hashlib.sha256(data).hexdigest()
    key = archive_key("futures", "BTCUSDT", "bookTicker", "2024-03-25")
    dest = tmp_path / key.key.split("/")[-1]
    dest.write_bytes(data)
    sess = FakeSession(data, sha)
    result = download_one(key, tmp_path, sess)
    assert result.status == "skipped_existing"
    assert sess.get_calls == 1  # only the checksum fetch


def test_download_one_resumes_part_file(tmp_path):
    import hashlib

    data = _make_zip(5)
    sha = hashlib.sha256(data).hexdigest()
    key = archive_key("futures", "BTCUSDT", "bookTicker", "2024-03-25")
    dest = tmp_path / key.key.split("/")[-1]
    part = dest.with_suffix(".zip.part")
    part.write_bytes(data[: len(data) // 2])  # interrupted download
    sess = FakeSession(data, sha, statuses={"support_range": True})
    result = download_one(key, tmp_path, sess, max_retries=2)
    assert result.status == "ok"
    assert result.bytes_downloaded == pytest.approx(len(data) - len(data) // 2, rel=0.2)


def test_download_range_guard_requires_acknowledgement(tmp_path):
    import hashlib

    data = _make_zip(20)
    sha = hashlib.sha256(data).hexdigest()

    class GuardSession(FakeSession):
        def head(self, url, timeout=None, allow_redirects=True):
            class H:
                headers = {"Content-Length": str(900 * 1024 * 1024)}  # 900MB > guard

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False
            return H()

    with pytest.raises(PermissionError, match="acknowledge"):
        download_range("futures", "BTCUSDT", "bookTicker", "2024-03-25", "2024-03-25",
                       tmp_path, session=GuardSession(data, sha),
                       large_download_allow_mb=500.0, acknowledged=False)


def test_sha256_of_file(tmp_path):
    import hashlib

    p = tmp_path / "x.bin"
    p.write_bytes(b"hello")
    assert sha256_of_file(p) == hashlib.sha256(b"hello").hexdigest()


def test_estimate_download_counts_missing_dates(tmp_path):
    class EstSession:
        """Prefix-aware listing double: returns only keys under the requested prefix."""

        def get(self, url, timeout=None):
            import urllib.parse

            prefix = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["prefix"][0]
            keys, _ = parse_s3_listing(S3_XML)
            matching = [k for k in keys if k.key.startswith(prefix)]

            contents = "".join(
                f"<Contents><Key>{k.key}</Key><Size>{k.size}</Size></Contents>" for k in matching)
            xml = ('<?xml version="1.0" encoding="UTF-8"?>'
                   '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                   f"<IsTruncated>false</IsTruncated>{contents}</ListBucketResult>")
            return FakeResponse(text=xml)

    est = estimate_download("futures", "BTCUSDT", "bookTicker",
                            "2024-03-25", "2024-03-27", session=EstSession())
    assert est["files_found"] == 2  # 25th and 26th exist in the XML
    assert est["files_missing"] == 1  # 27th does not
    assert est["missing_dates"] == ["2024-03-27"]
    assert est["approx_total_mb"] == pytest.approx(301.0, rel=0.01)
    assert "no files were downloaded" in est["note"]
