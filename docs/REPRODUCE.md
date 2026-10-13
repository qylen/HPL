# Exact Reproduction Commands

Python 3.10+ required. Install: `pip install -r requirements.txt`
(or `pip install -e ".[dev]"`). All commands run from the repository root.
Total wall-clock for the full workflow: ~25–40 minutes plus ~1.14 GB download.

## 1. Environment check

    python -m hpl1 env
    python -m hpl1 markets

## 2. Download and verify historical data (futures, ~915 MB)

    python -m hpl1 data inspect  --market futures --symbol BTCUSDT --dtype bookTicker
    python -m hpl1 data estimate --market futures --symbol BTCUSDT --dtype bookTicker \
        --start 2024-03-25 --end 2024-03-29
    python -m hpl1 data download --market futures --symbol BTCUSDT \
        --dtypes bookTicker,aggTrades,fundingRate --start 2024-03-25 --end 2024-03-29 --yes

fundingRate is monthly-only in the archive; the monthly file for 2024-03 is
fetched by the build step below. To pre-fetch it explicitly:

    python scripts/download_funding_monthly.py    # repo-relative: scripts/ outside package

## 3. Download spot data (~217 MB)

    python -m hpl1 data download --market spot --symbol BTCFDUSD \
        --dtypes aggTrades --start 2024-03-25 --end 2024-03-29 --yes

## 4. Verify every archive against official checksums

    python -m hpl1 data validate --market futures --symbol BTCUSDT
    python -m hpl1 data validate --market spot   --symbol BTCFDUSD

## 5. Full research pipeline (ingest -> features -> 6-model comparison ->
##    leakage suite -> cost-aware backtest -> statistics -> reports)

    python -m hpl1 research --market futures --symbol BTCUSDT \
        --start 2024-03-25 --end 2024-03-29
    python -m hpl1 research --market spot --symbol BTCFDUSD \
        --start 2024-03-25 --end 2024-03-29

Outputs: `artifacts/research/<market>_<symbol>/` — research_report.json/md,
backtest_report.json, backtest_zero_fee.json, fee_sensitivity.csv,
quality_report.json/md, leakage_suite.json, training_manifest.json,
best_model.joblib, test_predictions.parquet.

## 6. Individual steps (equivalent to the above, granular)

    python -m hpl1 build    --market futures --symbol BTCUSDT --start 2024-03-25 --end 2024-03-29
    python -m hpl1 compare  --market futures --symbol BTCUSDT --models zero,ridge,logistic,hist,law,regime
    python -m hpl1 backtest --market futures --symbol BTCUSDT --taker-fee-bps 5 --slippage-bps 1
    python -m hpl1 backtest --market futures --symbol BTCUSDT --fee-scenario zero
    python -m hpl1 evaluate --market futures --symbol BTCUSDT
    python -m hpl1 validate-leakage --market futures --symbol BTCUSDT

## 7. Statistical verification on real data

    python -m hpl1 validate-leakage --market futures --symbol BTCUSDT
    # label alignment, truncation invariance, split disjointness, planted-leak detection

## 8. Synthetic end-to-end checks (no network)

    python -m pytest -q                      # 132 offline tests
    python -m hpl1 synthetic-e2e             # planted archive -> recovered
    python -m hpl1 synthetic-e2e --noise     # control: no fake edge
    python -m hpl1 synthetic-e2e --corrupt   # duplicates/out-of-order/crossed/gaps
    python -m hpl1 demo --rows 20000 --directory artifacts/demo

## 9. Network tests (optional, real endpoints)

    python -m pytest -q -m network

## 10. Paper trading (public streams only, NO real orders)

    python -m hpl1 paper --market futures --symbol BTCUSDT \
        --model artifacts/research/futures_BTCUSDT/best_model.joblib --duration-seconds 300
    python -m hpl1 health                      # live health snapshot
    python -m hpl1 kill                        # manual kill switch

## Expected results (futures, seed 13)

OOS R² ≈ 0.100, correlation ≈ 0.318, non-zero directional accuracy ≈ 0.807,
zero trades at 5 bps taker + 1 bps slippage, gross zero-fee edge ≈ +241%
(unconstrained measurement, ≈ +0.148 bps/s). Numbers vary slightly with
library versions; materially different numbers indicate a pipeline defect.

## Config file

All of the above accept `--config config/research.yaml` (validated; see the
file for every tunable, including your real fee tier).
