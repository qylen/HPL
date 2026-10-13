# Historical Download Record

Every file below was genuinely downloaded from https://data.binance.vision/
and verified against the official `.CHECKSUM` sha256 digest. Nothing was
assumed about availability — every fact was probed (see tests/test_network.py).

## Completed downloads (16 files, ~1.14 GB)

| Market | Symbol | Dataset | Files | Range | Validation |
|---|---|---|---|---|---|
| futures | BTCUSDT | bookTicker daily | 5 | 2024-03-25..29 | sha256 verified |
| futures | BTCUSDT | aggTrades daily | 5 | 2024-03-25..29 | sha256 verified |
| futures | BTCUSDT | fundingRate monthly | 1 | 2024-03 | sha256 verified |
| spot | BTCFDUSD | aggTrades daily | 5 | 2024-03-25..29 | sha256 verified |

Ingestion integrity across the futures window: 70,540,368 applied book
events; 0 duplicate update ids; 0 out-of-order events; 0 crossed quotes;
431,984/432,000 seconds with book events (99.996%).

## Verified availability facts (probed, not assumed)

- **Spot publishes NO book history at all** — no bookTicker, no depth, for any
  symbol/date (S3 listing returns empty; network test `test_s3_listing_spot_has_no_book_data`).
  Spot research is trade-flow-only by necessity. Book features are never fabricated for spot.
- **Futures bookTicker daily archives END at 2024-03-30** (dataset
  discontinued by Binance; monthly series ends 2024-04). Any book-feature
  research window must lie within 2023-05..2024-03. Network test
  `test_futures_bookticker_discontinued_after_2024_03` asserts the 404.
- **fundingRate is monthly-only in the archive** (daily URLs 404). The
  downloader/ingester handle this explicitly; this was discovered live and is
  not an error condition.
- Futures aggTrades + bookTicker + spot aggTrades remain current and
  checksummed for all probed dates.
- `fapi.binance.com` REST returns HTTP 418 from many cloud IPs (geo-block);
  `exchange_meta.fetch_symbol_rules` falls back to documented defaults and
  labels their source. Spot REST is reachable.

## What was NOT downloaded (and why)

- Futures bookTicker outside 2024-03-25..29: sufficient for the research
  window; extending is a config change (`start_date`/`end_date`).
- bookDepth (futures): exists in the archive but is a coarse
  percentage-level snapshot — insufficient for top-of-book OFI research; not used.
- klines: benchmarks only; not needed for the one-second hypothesis.
- Spot book data: does not exist (see above).

Re-verify any time: `python -m hpl1 data validate --market futures --symbol BTCUSDT`.
