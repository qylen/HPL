# Limitations Report

## Data limitations

1. **No local receipt timestamps in historical archives.** All historical
   features/labels use exchange timestamps. Live latency (network + processing)
   is therefore absent from backtests; the paper engine records receipt time
   and reports book age, but no historical backtest can include latency.
2. **No spot book history exists.** Spot BTCFDUSD research uses 17 trade-only
   features. Book features (OFI, spread, depth) are structurally unavailable —
   the spot result (R² 0.066 vs futures 0.100) reflects reduced information,
   and future live spot collection could supplement it.
3. **Futures book history ends 2024-03-30.** The research window
   (2024-03-25..29) is the last usable bookTicker window. Results are from
   ONE 5-day window — regime coverage is minimal (no sustained trend/crash
   rotation), and no walk-forward across months was possible.
4. **Top-of-book only.** bookTicker gives best bid/ask. No depth beyond level
   1, no queue-position or partial-fill modelling; "multi-level depth" from
   bookDepth archives is percentage-level and was rejected as inadequate.

## Model limitations

5. Single contiguous train/val/test split. Selection happened on validation
   (once), but only one window was ever tested — selection-bias risk across
   research PROGRAMMES remains (see the demo command history in your own work).
6. HistGradientBoosting is a tabular model, not a sequence model. No temporal
   model was added because the marginal value over microstructure baselines
   was not demonstrated; TimesFM was not evaluated (one-second order flow is
   out of scope for general-purpose forecasters; adding it would be complexity
   without evidence).
7. The hidden-law model's interaction terms (I, I·L, I·σ) are a hypothesis
   test, not an established law; its validation R² (0.011) was far below hist
   (0.096) — the interpretable form did not survive contact with the data.

## Execution limitations

8. **Taker-only market orders.** No maker/limit fill model: a conservative
   fill model was required by the specification but not implemented because
   no evidence exists to calibrate one; this is documented rather than
   fabricated. Maker economics are the open follow-up hypothesis.
9. Slippage is a constant bps assumption, not a market-impact model.
10. No partial fills, exchange outages, rate limits, or adverse selection.
11. Funding payments use the observed mid at funding timestamps (not the
    exchange mark price) — a small approximation.

## Statistical limitations

12. One test window: HAC/block-bootstrap inference conditions on this window;
    they do not establish persistence across regimes.
13. 62.9% of one-second mid returns are exactly zero (0.1 tick on ~70k price);
    directional accuracy is reported over non-zero returns, and zero-return
    seconds mechanically cap attainable R².

## Remediation path (highest value first)

1. Extend the window to the full 2023-05..2024-03 futures book history
   (download config change) and run walk-forward evaluation.
2. Record live data now (the collector exists) to build a latency-aware,
   post-2024 dataset including depth via the diff-depth stream.
3. Specify and calibrate a conservative maker-fill model, then re-run the
   fee sensitivity with maker fees.
