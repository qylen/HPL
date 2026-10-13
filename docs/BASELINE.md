# HPL-1 Baseline Report (Phase A)

Date: 2026-10-12 · Code revision: 30a6182 (original upload) · Python 3.13.5

## Initial test results

`python -m pytest -q` → **6 passed** (tests/test_features.py: 2, tests/test_training.py: 2, tests/test_backtest.py: 2) in 1.29s.

## Current capabilities (preserved going forward)

- Public Binance **USD-M futures** WebSocket collector (`bookTicker` + `aggTrade`, no API key, reconnect + seq continuity).
- JSONL → one-second feature frame (20 features) with next-second mid-return labels; quote-anchored timeline; EPS guards on zero denominators.
- Ridge / HistGradientBoosting models; chronological 60/20/20 split with 1-row boundary purge; artifact persistence via joblib; feature diagnostics (std coefficients / validation permutation importance).
- `{-1,0,+1}` cost-aware backtest: per-side taker fee + slippage + half-spread, round-trip threshold, final liquidation, equity/drawdown/turnover stats, Newey-West HAC test.
- Synthetic frame generator + `demo` CLI command.
- Honest README disclaimers (no live orders, no profitability claims).

## Defects and gaps identified (to be fixed/implemented)

**Defects in existing code:**

1. **OFI ask-side sign convention is inverted** vs the standard Cont–Kukanov–Stoikov best-level definition: an ask-price *drop* (buyers lifting the offer) contributes **negative** OFI (`-aq_new`) instead of positive (`+aq_old`). Bid side is correct. This corrupts the central order-flow feature. Fix + unit test.
2. **Synthetic planted signal is contemporaneous, not predictive**: `target_return[i]` is generated from `ofi_norm[i]`, but in the real pipeline the label at row *i* is the return over second *i → i+1`. The demo therefore validates the wrong alignment. Fix: plant the signal one second ahead (`y[i+1] = f(ofi[i])`) and add a regression test.
3. **No event deduplication / ordering by exchange update id** (JSONL collector path cannot dedup book updates; reconnect duplicates are silently kept).
4. **Stale quotes are forward-filled indefinitely** across gaps; `quote_age_ms` exists as a feature but no invalidation flag, and no trading halt on stale data.
5. `build_feature_frame` dead code: `start_ms` computed twice (trades range ignored).
6. Single-model training only — no model **comparison/selection on validation** despite the README's claims about selection hygiene.
7. Backtester has no capital/quantity/notional accounting (unit positions only), no min-notional, no tick/step rounding, no cooldown, no daily-loss/drawdown halts, no turnover caps, no funding, no maker fee scenario, no per-regime breakdown, no fee-sensitivity output.

**Missing functionality (the main build):**

- Historical data acquisition from data.binance.vision (discovery, estimate, checksum-verified resumable downloads, manifests) — the #1 priority.
- Spot market support (BTCFDUSD profile); market profiles in general (fees/leverage/funding per market).
- Funding-rate ingestion and accounting; leverage/margin constraints for futures.
- Two-stage memory-bounded pipeline (streaming ZIP → per-second Parquet → features) for ~13M events/day.
- Models: zero benchmark, logistic directional baseline, hidden-law interaction model, regime-aware model.
- Leakage-detection suite; embargo in splits.
- Statistical validation: block bootstrap, time-shift placebo, Holm correction, Ljung–Box, calibration.
- Data-quality reports per dataset.
- Live paper-trading engine (shared features/risk/accounting), kill switch, healthcheck, JSONL audit logs.
- Config file system with validation; rotating logs; disk-space checks.
- CI, Docker, deployment/security/runbook docs.

## Plan

Build in place on this repository (git history preserved), keeping the legacy JSONL path (`collect`→`build`→`train`→`backtest`) working while adding the archive-based research pipeline as the primary path. Phases B–K per the specification.
