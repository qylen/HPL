# HPL-1: Hidden Predictive Law research toolkit

HPL-1 is a Python research pipeline for testing whether public crypto market-microstructure data contains a repeatable, one-second predictive relationship. It collects public USD-M Futures best-bid/ask and aggregate-trade events, builds features, trains chronological forecasting models, and evaluates a cost-aware hypothetical strategy.

**This is research software, not a profitable trading system. It does not place orders, manage an exchange account, or guarantee an edge.** The demo contains a deliberately planted synthetic signal so the software pipeline can be smoke-tested; demo results are not market evidence.

## Included

- Public Binance USD-M Futures WebSocket collector (`bookTicker` + `aggTrade`; no API key).
- Best-level order-flow imbalance (OFI), trade-flow, liquidity, spread, volatility, and clock-time features.
- One-second-ahead mid-price log-return labels, with features aligned to information available at the prediction timestamp.
- Ridge and histogram gradient-boosting regressors.
- Chronological train/validation/test split; the final model is fit on train + validation, not the test period.
- Cost-aware position backtest with per-side taker fees, slippage, spread crossing, turnover, and final liquidation costs.
- Newey-West/HAC test for mean net per-second returns (diagnostic only; not a proof of profitability).
- Ridge standardized-coefficient or validation permutation-importance report, plus test-residual autocorrelation diagnostics.
- Synthetic data generator and tests.

## 1. Set up

Python 3.10+ is recommended. From this directory:

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe -m pip install -e ".[dev]"` and run Python using `.\.venv\Scripts\python.exe`.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

You can use `pip install -r requirements.txt` instead of editable installation, then invoke commands as `python -m hpl1 ...` from this directory.

## 2. Collect market data

Collect BTCUSDT public perpetual-futures events for one hour:

```bash
python -m hpl1 collect --symbol btcusdt --duration-seconds 3600 --output data/raw/events.jsonl
```

Run until Ctrl+C by omitting the duration or setting it to zero:

```bash
python -m hpl1 collect --symbol btcusdt --output data/raw/events.jsonl
```

Then build the feature table:

```bash
python -m hpl1 build --input data/raw/events.jsonl --output data/processed/features.csv
```

The collector only records public data. It reconnects after transient errors and appends to its JSONL file. A reconnect or dropped-data interval can still affect labels and should be checked during data quality review. The collector does not reconstruct full depth-of-book; its OFI feature is calculated from best bid/ask price and quantity updates only.

**Data volume and history:** a one-second horizon is noisy and sensitive to latency. A brief capture is only a pipeline check, not enough to establish an edge. Capture varied market conditions and inspect timestamps, missing intervals, duplicate events, spread, and quote freshness before research. The order-flow features here are based on one exchange/one instrument; cross-exchange arbitrage, funding, liquidation feeds, and fees beyond the configured backtest costs are not implemented.

## 3. Train a model

Baseline ridge model:

```bash
python -m hpl1 train \
  --data data/processed/features.csv \
  --model ridge \
  --output artifacts/hpl1.joblib \
  --predictions artifacts/test_predictions.csv
```

On Windows PowerShell, run the same command on one line or use PowerShell's backtick line continuation.

Alternative nonlinear model:

```bash
python -m hpl1 train --data data/processed/features.csv --model hist --output artifacts/hpl1_hist.joblib --predictions artifacts/test_predictions_hist.csv
```

The default split is chronological: first 60% train, next 20% validation, final 20% test. The selected model family/hyperparameters are not automatically optimized against the test set. Keep the final test untouched while developing. If you experiment with many models or thresholds, use a later, independent test period or nested walk-forward testing to limit selection bias.

A minimum row check prevents tiny datasets from being treated as credible research. In practice, far more data is normally needed. A higher prediction correlation or accuracy alone is not proof of tradable profitability. `artifacts/hpl1.feature_diagnostics.csv` ranks standardized ridge coefficients or validation-set permutation importance; these are descriptive diagnostics, not causal explanations. Test residual autocorrelation is reported as an exploratory check, and any model changes inspired by final-test patterns require a new untouched test period.

## 4. Run the cost-aware test backtest

```bash
python -m hpl1 backtest \
  --predictions artifacts/test_predictions.csv \
  --output artifacts/backtest.csv \
  --report artifacts/backtest_report.json \
  --taker-fee-bps 5 \
  --slippage-bps 1 \
  --min-edge-bps 1 \
  --risk-buffer-bps 1
```

Replace the fee assumption with your actual account's fee tier. Basis points are 0.01%; e.g. 5 bps = 0.05% per side. Slippage is an assumed additional per-side cost. These values are illustrative defaults and should be adjusted to your exchange, instrument, order type, and execution measurements.

The policy uses positions `-1`, `0`, `+1`. It requires the predicted absolute return to exceed an approximate round-trip cost threshold, deducts one-way spread/fee/slippage costs for each position change, and charges to liquidate any final open position. The outputs are not an order-execution simulator: queue position, partial fills, market impact, funding, liquidation mechanics, exchange outages, latency and adverse selection are not fully modelled. Do not take the backtest as an expected return forecast.

`backtest_report.json` includes total return, drawdown, turnover, active time, costs, a per-second mean return, and a Newey-West/HAC statistic. The HAC p-value is an approximate diagnostic; repeated model/threshold selection can invalidate naive significance, and statistically nonzero gross predictions may have no net value.

## 5. Check that installation works

Run tests:

```bash
python -m pytest -q
```

Run the whole model/evaluation software path against synthetic data:

```bash
python -m hpl1 demo --rows 20000 --directory artifacts/demo
```

The synthetic dataset has a deliberately planted dependency to verify that the training pipeline works. It is **not real crypto data**, does not demonstrate a real edge, and should never be used to estimate likely live returns.

## Data and timing conventions

- `event_time_ms` is exchange event time; `received_time_ms` is local receipt time; `seq` preserves local receive ordering when event timestamps tie.
- `buyer_is_maker=true` is treated as an aggressive sell; `false` as an aggressive buy.
- One row represents one UTC second bucket. Quote and trade quantities in the Binance USD-M feed use the quantity units supplied by the exchange.
- Features for bucket `t` use quote/trade events timestamped in that bucket and prior buckets. `target_return[t] = log(mid[t+1] / mid[t])` and the final row has no target.
- In the current implementation, `mid[t]` is the last observed top-of-book mid-price inside the second bucket; empty quote buckets carry forward the last observed book state. For rigorous work, audit gaps/reconnects and consider excluding intervals affected by data outages or stale feed states.
- OFI is best-level OFI and its scale depends on instrument quantity units. It is not full depth-of-book OFI.

## Suggested next research steps

1. Add capture-health metadata and exclude reconnect/data-gap intervals.
2. Compare prediction performance with the zero-return and simple OFI baselines on the same timestamps.
3. Use walk-forward tests across multiple periods and correct for multiple hypotheses.
4. Test whether feature coefficients/importance and conditional performance remain stable by spread, depth and volatility regime.
5. Add event-driven execution simulation and compare intended mid-price returns with actual paper fill prices.
6. Only after robust paper results should execution integration be considered; this project intentionally includes no live-order module.

## CLI help

```bash
python -m hpl1 --help
python -m hpl1 collect --help
python -m hpl1 train --help
python -m hpl1 backtest --help
```
