"""Download the monthly funding archive for 2024-03 and record it."""
import sys
from pathlib import Path

sys.path.insert(0, "/home/z/my-project/HPL")

import requests

from hpl1.binance_archive import archive_key
from hpl1.downloader import download_one, print_results

RAW = Path("/home/z/my-project/HPL/data/raw")
key = archive_key("futures", "BTCUSDT", "fundingRate", "2024-03", period="monthly")
dest = RAW / "futures" / "BTCUSDT" / "fundingRate"
results = [download_one(key, dest, requests.Session(), max_retries=5)]
print_results(results, "futures/BTCUSDT/fundingRate monthly 2024-03")
assert results[0].status == "ok", results[0]
print("size:", results[0].size, "bytes")
