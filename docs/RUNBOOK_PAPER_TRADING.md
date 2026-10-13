# Paper-Trading Runbook

## Start

    python -m hpl1 paper --market futures --symbol BTCUSDT \
        --model artifacts/research/futures_BTCUSDT/best_model.joblib \
        [--config config/paper.yaml]

Runs until Ctrl+C (graceful: state saved, final heartbeat written).
`data/paper/latest` symlinks to the active run directory.

## What the engine does every second

1. Aggregates bookTicker/aggTrade/@trade events into the CURRENT second's
   bucket (update-id monotonic; crossed quotes rejected; dual-stream trade
   dedup).
2. On each second boundary: closes the bucket, appends to a 900-second
   rolling frame, rebuilds THE SAME features as historical research, runs the
   model, applies the cost hurdle and risk limits, simulates fills at
   book prices ± slippage with taker fees.
3. Polls the kill switch every second — including during warmup.
4. Heartbeat every 60 s: health.json + equity log.

## Files in the run directory

| File | Contents |
|---|---|
| observations.jsonl | sampled raw events (1 per 200) |
| decisions.jsonl | every prediction + action + reason |
| orders.jsonl / fills.jsonl | every simulated order/fill |
| equity.jsonl | heartbeat equity snapshots |
| state.json | restart-resumable account state |
| health.json | live health snapshot |
| kill_switch | manual kill file (created by `hpl1 kill`) |
| logs/hpl1.live.log | rotating structured log |

## Monitoring

- `hpl1 health` — expect book_valid=true, events increasing (~200–600/s on
  futures BTCUSDT), book_age < 5 s, kill_switch.tripped=false.
- Drift signals: mean |prediction_bps| far from the training value; decision
  rate collapsing (feed problem); skip rate rising (stale data).

## Halting

- Manual: `hpl1 kill --run-dir data/paper/latest` — halts within ~1 s.
  Removing the file does NOT resume: restart the process (documented,
  deliberate). Ctrl+C is graceful.
- Automatic: daily loss > 3%, drawdown > 10%, or stale data > 30 s.

## After a halt

1. Read the kill reason in health.json.
2. If data-related: verify the exchange status page; restart.
3. If loss-related: do NOT restart without review — re-run the research
   pipeline on fresh data first.
4. Positions open at halt are kept in state.json; the next session resumes
   them (and the cost model will flatten them if the signal is absent).

## Expected behaviour with the current trained models

Predictions are ~0.1–2 bps against a ~12 bps round-trip hurdle: the engine
stays flat (`flat_below_hurdle`) and places NO orders. This is the correct
outcome; a paper session that suddenly starts trading means either the
config's fee tier changed or the model/data has drifted — investigate.
