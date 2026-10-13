# HPL-1 Research Findings Report

**Status: COMPLETE — real historical data, fully reproducible.**
This report distinguishes verified facts from assumptions. Nothing here is a
promise of profitability.

## 1. What was hypothesized

Order-flow imbalance, liquidity, spread, volatility, trade flow and regime
changes at second *t* predict the next-second mid-price return
r(t+1) = log(mid(t+1)/mid(t)) well enough to trade profitably after realistic
taker execution costs on Binance BTCUSDT USD-M perpetual futures (and, with
reduced feature coverage, BTCFDUSD spot).

## 2. What data was used

Genuinely downloaded from https://data.binance.vision/ and SHA-256-verified
against official `.CHECKSUM` files (see docs/DOWNLOAD_RECORD.md):

| Dataset | Market | Range | Size | Integrity |
|---|---|---|---|---|
| bookTicker (daily) | futures BTCUSDT | 2024-03-25..29 | 800 MB | 70,540,368 applied events; 0 duplicate ids, 0 out-of-order, 0 crossed quotes |
| aggTrades (daily) | futures BTCUSDT | 2024-03-25..29 | 114 MB | checksum-verified |
| fundingRate (monthly) | futures BTCUSDT | 2024-03 | 939 B | checksum-verified (daily files 404 — monthly-only dataset) |
| aggTrades (daily) | spot BTCFDUSD | 2024-03-25..29 | 217 MB | checksum-verified |

Assembled: **432,000 one-second observations** (5 full UTC days per market);
431,999 usable labeled rows on futures (99.999%); spot coverage 98.1%
(trade-flow only: spot publishes no book history — verified, not assumed).

Timestamp semantics: exchange timestamps (bookTicker `transaction_time`,
trades `transact_time`). Archives contain no local receipt time; live
collection records it separately (see docs/LIMITATIONS.md).

## 3. Which models were compared

Six candidates, identical chronological conditions (60/20/20 split,
purge 1s, embargo 30s, selection on validation only):

zero benchmark, ridge (standardized), logistic directional, histogram gradient
boosting, hidden-law interaction (I, I·L, I·σ + controls), volatility-regime
ridge (terciles fit on train only).

## 4. Which results were held out from tuning

The final 20% test split (86,358 seconds) was evaluated ONCE for the single
validation-selected model (hist). No other model, threshold or configuration
touched it. The leakage suite (label alignment, truncation invariance, split
disjointness, planted-leak detection) passed on the real dataset.

## 5. What worked and what failed

### Futures BTCUSDT (book + trade features)

| Metric (untouched test set) | Value |
|---|---|
| OOS R² vs zero forecast | **0.0998** |
| prediction/return correlation | **0.318** |
| directional accuracy (non-zero returns) | **80.7%** |
| conditional mean return given pred>0 / pred<0 | +0.147 / −0.149 bps |
| zero-return seconds (tick quantization) | 62.9% |
| placebo (time-shifted) |corr| p95 | 0.195 → real corr 0.318 is alignment-dependent (not a timestamp artifact) |
| block-bootstrap corr CI (block 60s, 300 reps) | 0.303 .. 0.333 |
| HAC z (gross mean, naive sign strategy) | 37.6 |
| calibration | Brier 0.129 vs 0.190 base-rate (validation-fit logistic) |

**Economics:** predicted edges (~1–2 bps) never clear the ~12 bps round-trip
hurdle (2×5 bps taker + 2×1 bps slippage + spread): **zero trades, net return
0.0%** — the correct behaviour, not a bug.

Fee sensitivity (fees isolated, spread paid, unconstrained sizing — an edge
measurement, not a tradable configuration):

| taker fee | trades | net |
|---|---|---|
| 0 bps | 30,080 | **+241.6%** (HAC z 48.9; ≈ +0.148 bps/s gross, compounded) |
| 0.5 bps | 28 | −0.06% |
| ≥ 1 bps | 0 | 0.0% |

### Spot BTCFDUSD (trade-only features; no book history exists)

OOS R² 0.066, corr 0.257, conditional means ±0.073 bps, placebo-clean. The
drop from futures (R² 0.100, ±0.148 bps) quantifies how much signal lives in
the book features that spot history cannot provide.

## 6. Statistical uncertainty and known limitations

- One contiguous 5-day window (2024-03-25..29) — no walk-forward across
  regimes; a single-window result is vulnerable to period effects.
- Historical archives lack local receipt timestamps; real trading adds
  latency, adverse selection and queue effects none of which are modelled.
- Top-of-book only: no depth beyond best level, no queue-position modelling.
- Taker-only execution; maker fill economics are the untested follow-up.
- Multiple-testing control: Holm-corrected p-values across the battery; the
  battery was specified once, not iterated against the test set.

## 7. Verdict

1. **Statistical predictability: SUPPORTED.** Strong, alignment-dependent,
   out-of-sample one-second structure on real Binance data (R²≈0.10,
   corr≈0.32, conditional means ±0.15 bps).
2. **Net profitability after realistic taker costs: REJECTED** for this
   window. The gross edge (~0.15 bps/s) is an order of magnitude smaller than
   the taker cost hurdle (~12 bps round trip). Zero trades is the honest
   outcome.
3. **Maker-execution economics: UNRESOLVED.** At 0–0.5 bps effective costs
   the signal has measurable content; whether a passive fill model can
   capture it is the next falsifiable hypothesis, not a conclusion.

Reproduce with: `python -m hpl1 research --market futures --symbol BTCUSDT
--start 2024-03-25 --end 2024-03-29` (see docs/REPRODUCE.md).
