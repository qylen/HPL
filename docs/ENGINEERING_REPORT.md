# HPL-1 Engineering Report

**Delivery date:** 2026-10-12 · **Repository:** github.com/qylen/HPL (extended in place,
history preserved) · **Verification basis:** everything below was executed in this
environment; nothing is claimed from memory.

## 1. What was implemented

Starting from the original 12-file skeleton (WS collector, JSONL features, ridge/hist,
unit-position backtest, 6 tests), the repository now contains a complete, tested
research and paper-trading system:

- **Market profiles** — spot BTCFDUSD (long-only, no leverage, no funding) and USD-M
  BTCUSDT perpetual (two-sided, 2x-capped leverage, 8 h funding) — never mixed.
- **Historical acquisition** from data.binance.vision: S3 discovery, size estimation,
  SHA-256-verified resumable downloads with retry/jitter, manifests, >500 MB guard,
  monthly fundingRate handling.
- **Two-stage memory-bounded pipeline**: streaming ZIPs → per-second Parquet buckets
  (70.5 M events in ~6 min) → 28 full / 17 trade-only features with staleness
  invalidation and per-dataset quality reports.
- **Six models** (zero, ridge, logistic, hist, hidden-law, regime) with chronological
  60/20/20 splits, purge + embargo, validation-only selection, saved manifests.
- **Leakage suite**: label alignment, truncation invariance, split disjointness,
  planted-leak detection — run on the real dataset (passed) and in tests.
- **Cost-aware backtester**: cash/quantity accounting, spread/slippage/commission/funding
  as separate components (no double counting), Decimal tick/step rounding, leverage
  caps, cooldown, turnover cap, daily-loss/drawdown halts, final liquidation,
  per-regime stats, fee-sensitivity curves, zero-fee edge measurement.
- **Statistical validation**: HAC, moving-block bootstrap, time-shift placebo,
  Ljung–Box, Holm correction, validation-fit directional calibration.
- **Live paper trading**: WS engine with the SAME feature builder and cost model as
  research, dual trade-stream fallback (a real futures aggTrade outage on this network
  was detected and worked around), per-second kill-switch polling, JSONL audit logs,
  health snapshots, state resume, graceful shutdown.
- **Production**: CI (3 Pythons), Docker + compose, validated configs, 10 docs.

## 2. Source files changed

Rewritten: `cli.py`, `features.py`, `models.py`, `training.py`, `backtest.py`,
`synthetic.py`, `collector.py`, `tests/*`. New: `markets.py`, `config.py`,
`binance_archive.py`, `downloader.py`, `ingest.py`, `dataset.py`, `quality.py`,
`validation.py`, `statistics.py`, `risk.py`, `evaluate.py`, `live.py`,
`exchange_meta.py`, `logging_setup.py`, 9 test files, CI, Docker, configs, docs.
All original tests retained and passing. Full history in git log.

## 3. Binance data actually obtained

16 files, ~1.14 GB, all SHA-256-verified (docs/DOWNLOAD_RECORD.md):
futures BTCUSDT bookTicker + aggTrades + fundingRate (monthly), spot BTCFDUSD
aggTrades — 2024-03-25..29 → 432,000 seconds, 99.996% book coverage,
0 duplicates/out-of-order/crossed.

## 4. Tests

- `python -m pytest -q` → **132 passed** (offline, no network, no credentials).
- `python -m pytest -q -m network` → **6 passed, 1 skipped** (fapi geo-block
  HTTP 418 — environmental, skip with reason).
- `ruff check hpl1 tests` → clean; `python -m compileall -q hpl1` → clean.

Defects found and fixed during testing: inverted OFI ask-side sign (from the
original code), contemporaneous planted signal, pred_frame price misalignment
after dropna (would have corrupted every backtest), pandas-3 headerless-CSV
header swallowing, non-monotonic bucket rollover under clock skew, kill switch
not polled during warmup, unbounded boundary loop on clock jumps, fundingRate
monthly-only 404s, duplicate-column crash, NaN-mask guards in quality reports.

## 5. End-to-end commands

See docs/REPRODUCE.md (research, spot, leakage, synthetic-e2e, paper, health, kill).

## 6. Real-data results

Futures BTCUSDT test split (86,358 untouched seconds): **OOS R² 0.0998**,
**correlation 0.318**, **non-zero directional accuracy 80.7%**, conditional
means ±0.15 bps, placebo-clean (shifted |corr| p95 0.195), block-bootstrap CI
[0.303, 0.333]. Net backtest at 5 bps taker + 1 bps slippage: **zero trades,
0.0%** — predicted edges never clear the 12 bps hurdle. Zero-fee unconstrained
gross edge: +241.6% (≈ +0.148 bps/s; HAC z 48.9). Fee curve: 0.5 bps fee
already destroys it. Spot (trade-only): R² 0.066, corr 0.257.

## 7. Was an edge found?

**Statistical predictability: YES — strongly supported.** **Net tradable edge at
realistic taker costs: NO — rejected for this window.** The gross signal content
(~0.15 bps/s) is an order of magnitude below taker costs. Maker-execution
economics are the untested follow-up hypothesis. No tuning was performed to
change these conclusions.

## 8. Security and deployment status

No credentials anywhere; no execution client (live trading is structurally
impossible); fail-closed staleness/kill/halt behaviour tested end-to-end;
kill switch verified live (trip ≤1 s, restart required). CI/Docker/configs/
runbook/rollback delivered. Paper trading is production-ready as a service.

## 9. Known limitations and blockers

Docs/LIMITATIONS.md (13 items). Headline: one 5-day window, no local receipt
timestamps in archives, top-of-book only, taker-only fills, spot has no book
history, futures book history ended 2024-03-30.

## 10. Path from paper trading to carefully controlled live deployment

1. **Evidence period:** run paper trading continuously for ≥4 weeks; require
   the health/decision logs to show stable data quality and behaviour matching
   the backtest (which, today, means staying flat at taker costs).
2. **Maker hypothesis first:** specify and calibrate a conservative limit-order
   fill model; re-run fee sensitivity with maker fees; only proceed if a net
   positive maker scenario survives out-of-sample and walk-forward tests.
3. **Testnet adapter:** implement an authenticated TESTNET-only execution
   client in a separate module (never imported by research/paper paths),
   with env-var credentials, reconciliation, duplicate-order protection.
4. **Operator acknowledgement:** a signed record (config + doc) naming the
   operator, date, scope, and the exact evidence relied upon.
5. **Live pilot:** smallest possible notional, hard caps enforced in code
   (position/notional/leverage/daily-loss), kill switch rehearsed, monitoring
   on every decision, automatic halt on any state mismatch.
6. **Continuous verification:** exchange state reconciliation before every
   resume; weekly re-run of the research pipeline; any drift → halt and
   re-evaluation. Live readiness is re-earned continuously, not conferred
   once.

**This delivery does not claim live-trading readiness.**
