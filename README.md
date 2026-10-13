# HPL-1: Hidden Predictive Law research toolkit

HPL-1 is a tested, reproducible Python system for researching whether public
crypto market-microstructure data contains a repeatable **one-second**
predictive relationship — and for measuring honestly whether that relationship
survives realistic execution costs. It downloads official Binance historical
archives, builds leakage-guarded features and labels, compares six models
under chronological validation, runs a full-mechanics cost-aware backtester,
and paper-trades live public streams with the same code path.

**This is research software. It contains no live-order execution client, places
no orders, manages no exchange account, and promises no profitability.**

> Verified headline result (futures BTCUSDT, 2024-03-25..29, untouched test
> split): OOS R² 0.100, correlation 0.318, non-zero directional accuracy
> 80.7% — statistical predictability is real. Net backtest at 5 bps taker +
> 1 bps slippage: **zero trades** — predicted edges (~1–2 bps) never clear the
> ~12 bps round-trip hurdle. Predictability is not profitability.
> See docs/RESEARCH_FINDINGS.md.

## What's inside

| Component | Where |
|---|---|
| Historical downloader (S3 discovery, estimates, SHA-256 checksums, resume, retry, manifests, size guard) | `hpl1/binance_archive.py`, `hpl1/downloader.py` |
| Streaming ingestion (ZIP → per-second Parquet; dedup/out-of-order/crossed handling; Cont-style OFI) | `hpl1/ingest.py`, `hpl1/dataset.py` |
| One-second features + labels (28 full / 17 trade-only; staleness invalidation) | `hpl1/features.py` |
| Six models (zero, ridge, logistic, hist, hidden-law, regime) | `hpl1/models.py`, `hpl1/training.py` |
| Leakage-detection suite (label alignment, truncation invariance, split disjointness, planted-leak detector) | `hpl1/validation.py` |
| Cost-aware backtester (spread/slippage/commission/funding split, Decimal rounding, halts, fee curves) | `hpl1/backtest.py` |
| Statistics (HAC, block bootstrap, time-shift placebo, Ljung–Box, Holm, calibration) | `hpl1/statistics.py`, `hpl1/evaluate.py` |
| Live paper engine (shared features/costs, dual-stream fallback, kill switch, health, state resume) | `hpl1/live.py` |
| Risk limits + kill switch | `hpl1/risk.py` |
| Data-quality reports | `hpl1/quality.py` |
| Market profiles (spot BTCFDUSD / futures BTCUSDT, never mixed) | `hpl1/markets.py` |
| Tests: 132 offline + 6 network (unit, integration, end-to-end on synthetic Binance-format archives) | `tests/` |

## Quick start

```bash
python -m pip install -r requirements.txt

python -m hpl1 env                                   # environment check
python -m hpl1 markets                               # supported market profiles
python -m pytest -q                                  # offline test suite (132 tests)

# historical research (downloads ~1.1 GB, checksum-verifies, ingests,
# trains/compares 6 models, runs the leakage suite, backtests, reports)
python -m hpl1 data download --market futures --symbol BTCUSDT \
    --dtypes bookTicker,aggTrades,fundingRate --start 2024-03-25 --end 2024-03-29 --yes
python -m hpl1 research --market futures --symbol BTCUSDT --start 2024-03-25 --end 2024-03-29

python -m hpl1 research --market spot --symbol BTCFDUSD --start 2024-03-25 --end 2024-03-29

# paper trading on live public streams (NO real orders)
python -m hpl1 paper --market futures --model artifacts/research/futures_BTCUSDT/best_model.joblib
python -m hpl1 health
python -m hpl1 kill

# synthetic end-to-end verification (no network)
python -m hpl1 synthetic-e2e            # planted signal must be recovered
python -m hpl1 synthetic-e2e --noise    # control: no fake edge allowed
```

Full command reference and exact reproduction steps: **docs/REPRODUCE.md**.
Validated configuration: `config/research.yaml` (fee tiers, leverage caps,
risk limits, splits — malformed or live-trading configs are rejected).

## Market profiles

| | Spot | USD-M Perpetual |
|---|---|---|
| Symbol | BTCFDUSD | BTCUSDT |
| Directions | long only | long + short |
| Leverage cap | 1x | 2x (configurable below the cap) |
| Funding | none | 8 h, applied in backtest/paper |
| Historical book data | **does not exist** (verified) → trade-only features | bookTicker through 2024-03-30 |
| Min notional default | 5 | 100 |

Spot and futures observations are never mixed in one dataset.

## Data integrity and leakage safety

- Every archive is SHA-256-verified against Binance's official `.CHECKSUM`.
- Ingestion counts and skips duplicate update ids, out-of-order events and
  crossed quotes; nothing is silently dropped.
- Features use only information available at the prediction timestamp; labels
  are exactly `log(price[t+1]/price[t])`. The automated leakage suite (run on
  every `research` invocation) also proves the checks themselves catch planted
  leakage.
- Selection happens on the validation split only; the test split is evaluated
  once, for one model.
- Stale quotes are forward-filled but explicitly flagged invalid; trading
  fails closed on invalid data.

## Backtesting model

Market orders buy at `ask·(1+slippage)` and sell at `bid·(1−slippage)`;
the spread cost is the bid/ask distance inside those prices, commission is
charged per side on notional, funding is applied at funding timestamps —
each component tracked separately, nothing double-counted. Quantity/price are
rounded to the exchange step/tick with Decimal arithmetic. Position sizing,
turnover caps, cooldowns, daily-loss and drawdown halts, min-notional and
final liquidation are all configurable (config/research.yaml). Limit orders
are NOT modelled (no honest fill model exists without calibration data) —
maker economics are an explicit open hypothesis.

## Fee scenarios

Zero fees are an explicit scenario (`--fee-scenario zero` or
`fee_scenario: zero`), never a default. `fee_sensitivity.csv` reports the fee
curve (fees isolated, spread paid): 0 bps → gross edge visible; ≥1 bps → no
trades. Replace `taker_fee_bps` with your account's actual tier.

## Documentation

- docs/RESEARCH_FINDINGS.md — the full research report (results above)
- docs/ENGINEERING_REPORT.md — what was built, changed, verified
- docs/DOWNLOAD_RECORD.md — exactly which data was obtained and validated
- docs/LIMITATIONS.md — 13 documented limitations + remediation path
- docs/REPRODUCE.md — exact commands for the entire workflow
- docs/SECURITY.md — checklist and threat model
- docs/DEPLOYMENT.md — deployment, rollback, backups
- docs/RUNBOOK_PAPER_TRADING.md — operating the paper engine
- docs/PRODUCTION_READINESS.md — readiness checklist
- docs/BASELINE.md — the original repository's audit baseline

## Safety

Default mode is research/paper. There is no authenticated exchange client in
this codebase; live trading is structurally impossible. The paper engine
polls a kill switch every second, fails closed on stale data, and enforces
hard position/notional/leverage/daily-loss limits before any simulated order.

## Development

```bash
python -m pytest -q            # offline suite
python -m pytest -q -m network # real-endpoint tests (optional)
ruff check hpl1 tests
```

CI runs ruff + compile + the offline suite on Python 3.10/3.11/3.12
(.github/workflows/ci.yml). Docker: `docker compose build` (research and
paper services, non-root).

## Legacy JSONL path

The original `collect` (WebSocket recorder) → `build-features` → `train` →
`backtest-legacy` workflow still works for locally collected JSONL data,
including the `demo` command on synthetic data.
