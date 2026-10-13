# HPL-1 Research Report — futures BTCUSDT (2024-03-25..2024-03-29)

## Hypothesis
Order-flow imbalance, liquidity, spread, volatility, trade flow and regime changes predict the next one-second mid-price return well enough to trade profitably after realistic taker execution costs.

## Data
- Market: futures / BTCUSDT
- Window: 2024-03-25..2024-03-29
- Usable rows: 431999

## Model comparison (selection on validation only)
| model | val R2 | val dir.acc | test R2 | test corr |
|---|---|---|---|---|
| zero | 0.0000 | n/a | n/a | n/a |
| ridge | 0.0809 | n/a | n/a | n/a |
| logistic | -0.0413 | n/a | n/a | n/a |
| hist | 0.0955 | n/a | 0.0998 | 0.3180 |
| law | 0.0113 | n/a | n/a | n/a |
| regime | 0.0854 | n/a | n/a | n/a |
- selected: **hist**
- validation R2: 0.0955, directional accuracy: n/a
- test R2 (untouched): 0.0998, correlation: 0.3180, directional accuracy: n/a

## Statistical battery (test set)
- Placebo: real corr 0.3180 vs shifted |corr| p95 0.1946 -> alignment-dependent: True
- HAC z (gross mean): 37.5777
- Block-bootstrap corr CI: 0.3027 .. 0.3330
- Brier score: 0.1286

## Net backtest (cost-aware, taker execution)
- Trades: 0, round trips: 0
- Net return: 0.00%  |  max DD: 0.00%
- Costs by component: {'commission': 0.0, 'spread': 0.0, 'slippage': 0.0, 'funding': 0.0}

## Verdict
- Statistical predictability: **supported**
- Signal is alignment-dependent (not timestamp artifact): **True**
- Net taker profitability: **rejected**

Out-of-sample predictability exists (R2=0.100, corr=0.318) but predicted edges never clear the round-trip cost hurdle: zero trades at realistic taker fees. Predictability is NOT profitability.

## Limitations
- Single contiguous window; no walk-forward across regimes.
- Taker-only execution; maker fill economics untested.
- Top-of-book data only; no depth beyond best level, no queue-position modelling.
- Historical archive lacks local receipt timestamps; live latency not represented.
