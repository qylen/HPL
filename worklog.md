# HPL-1 Worklog (Phases A–K)

Task ID: HPL-1 full execution
Agent: Super Z (main agent, no delegation)
Started: 2026-10-12 14:5x UTC+8 · Finished: 2026-10-12 17:0x UTC+8

---
Task: Execute the complete HPL-1 specification (A–K): inspect the original
repo, build the historical Binance pipeline, models, leakage guards,
cost-aware backtester, statistics, paper trading, tests, CI/Docker/docs, run
everything on real data, audit honestly.

Work Log:
- Phase A: cloned github.com/qylen/HPL (commit 30a6182), read all 12 source
  files + 3 test files, ran the 6 original tests (6 passed, 1.3 s). Wrote
  docs/BASELINE.md: 7 defects (inverted OFI ask-side sign, contemporaneous
  planted signal, no dedup by update id, unbounded stale ffill, dead code,
  no model selection, unit-position backtester) + missing functionality list.
- Phase B: markets.py (spot BTCFDUSD / futures BTCUSDT profiles), config.py
  (validated YAML; live_trading_enabled rejected), binance_archive.py (S3
  discovery, 6/7-segment key parsing), downloader.py (sha256 CHECKSUM
  verification, .part resume via Range, retry+jitter, manifests, 500 MB
  guard), exchange_meta.py (REST with geo-block fallback),
  logging_setup.py (rotating logs + disk checks).
- Phase C: ingest.py — streaming ZIP→Parquet with header-peek reader
  (futures header vs spot headerless), vectorized monotonic-update-id book
  application, corrected Cont OFI, per-second aggregation; dataset.py
  (full-grid assembly); features.py rewrite (28 full/17 trade-only features,
  staleness/gap flags, reindex onto missing seconds) keeping the legacy
  JSONL API; quality.py reports.
- Phase D: models.py (6 models, save/load, deterministic), training.py
  (chronological splits purge 1 s/embargo 30 s, validation-only selection,
  test evaluated once), validation.py (leakage suite + planted-leak
  detector).
- Phase E: backtest.py full engine (separate cost components, Decimal
  rounding, funding, halts, min-notional, turnover cap, final liquidation,
  fee sensitivity) with the legacy {-1,0,+1} backtest preserved; risk.py
  (kill switch + hard limits).
- Phase F: statistics.py (HAC, Ljung–Box, moving-block bootstrap, time-shift
  placebo, Holm, calibration), evaluate.py (battery + research reports).
- Phase G: live.py paper engine (dual @aggTrade/@trade subscription — a real
  futures aggTrade stream outage on this network was observed and worked
  around, markPrice funding, per-second boundary ticker task, kill-switch
  polling every second, state resume, health.json, JSONL audit logs);
  collector.py upgraded to multi-market + watchdog.
- Phase H: complete CLI (env/markets/data inspect|estimate|download|validate/
  build/compare/backtest/evaluate/research/validate-leakage/paper/health/
  kill/demo/synthetic-e2e + legacy commands), CI (3 Pythons), Docker +
  compose, configs, 10 docs, README rewrite.
- Phase I: test suite grown 6 → 132 offline + 6 network tests. Defects found
  & fixed while testing: pandas-3 fillna(method=) removal, ndarray.clip(lower)
  ordering, any() on arrays, headerless CSV first-row swallowing, 7-segment
  S3 keys, resume Range parsing in test double, single-row RegimeRidge
  broadcast (regression-tested), boundary-loop hang on clock jumps (guard
  added), duplicate-bucket race under clock skew (monotonic rollover + dedup),
  kill switch not polled during warmup, quality-report unguarded columns,
  test fixtures using ms instead of s.
- Phase J (real data): downloaded 16 archives (futures bookTicker 800 MB +
  aggTrades 114 MB + monthly funding 939 B; spot aggTrades 217 MB), all
  SHA-256-verified via `data validate` (after fixing its date-token parsing).
  Discovered fundingRate is monthly-only (daily 404s) → monthly ingest path.
  Ran the full research pipeline on BOTH markets. Critical bug found via the
  contradiction between corr=0.318 and a negative zero-fee backtest:
  pred_frame price columns were misaligned (df.loc on a re-indexed work
  frame) — fixed + regression-tested (e2e now asserts positive zero-fee PnL
  on planted data). Metric fixes: directional accuracy over non-zero returns
  (62.9% zero-tick seconds), conditional means, validation-fit calibration
  (Brier 0.129 vs 0.190 base), fee curve with slippage isolated to zero,
  unconstrained edge-measurement risk config.
  Paper trading verified live: futures 46k events/105 s, 87 decisions,
  correctly flat (0.1–2 bps predictions vs 12 bps hurdle), zero errors after
  the clock-skew fixes; kill switch verified end-to-end (trips ≤1 s, restart
  required); spot session verified (48 decisions, 0 errors).
- Phase K: ruff configured and clean, compileall clean, full suite green,
  docs completed, final audit + zip.

Stage Summary:
- Futures BTCUSDT 2024-03-25..29 (untouched test 86,358 s): OOS R² 0.0998,
  corr 0.318, non-zero dir acc 80.7%, cond means ±0.148 bps, placebo-clean
  (shifted p95 0.195), block-bootstrap CI [0.303,0.333], leakage suite
  passed. Net backtest 5 bps taker + 1 bps slip: ZERO trades (correct).
  Zero-fee unconstrained gross edge +241.6% ≈ +0.148 bps/s (HAC z 48.9);
  0.5 bps fee already destroys it. Spot trade-only: R² 0.066, corr 0.257.
- Verdict: statistical predictability SUPPORTED; net taker profitability
  REJECTED; maker economics UNRESOLVED (next falsifiable hypothesis).
- Tests: 132 offline passed, 6 network passed, 1 skipped (fapi geo-block),
  ruff clean, compile clean.
- No live-order execution client exists anywhere in the codebase.
