# Security Checklist and Threat Model

## Default posture

- Default operating mode is research or paper trading. **This codebase
  contains NO authenticated exchange client and NO live-order execution path
  — live trading is structurally impossible, not merely disabled.**
- `load_config` rejects `live_trading_enabled: true` outright.
- Example configs ship with trading disabled semantics only.

## Secrets

- No API keys are required or used anywhere: historical downloads and paper
  trading use public endpoints/streams only.
- No secrets are hardcoded, logged, or committed (audited: the strings
  "api_key"/"secret" appear only in this document).
- If an execution adapter is ever added: keys must come from environment
  variables or a secrets manager, never files in the repo; read-only data
  credentials must remain separate from trading credentials; withdrawal
  scopes must never be requested; credentials must never appear in logs.

## Fail-closed behaviour (implemented and tested)

- Stale/invalid market data forces positions flat and blocks new decisions
  (backtest `halt_stale`; paper engine `valid` gate).
- Kill switch: manual file (`hpl1 kill`) polled EVERY second (including during
  warmup) plus automatic triggers (daily loss, drawdown, stale data).
  Restart required after a trip; the switch never silently un-trips.
- Model/feature schema mismatch refuses to trade (logged, skipped).
- Duplicate order protection: paper engine deduplicates stream events by
  trade id+timestamp and ignores book update-id regressions after reconnects.
- State reconciliation: paper account state persists across restarts
  (state.json); positions are re-loaded, not re-opened.

## Hard limits enforced before any simulated order

Max position size, max notional, leverage cap (profile-enforced: spot 1x,
futures 2x), min notional, turnover cap, cooldown, daily-loss halt,
drawdown halt, Decimal tick/step rounding.

## Threat model summary

| Threat | Mitigation |
|---|---|
| Malformed config silently enabling trading | config validator rejects live flags; no execution client exists |
| Compromised feed / bad data | staleness + validity gates; crossed-quote rejection; fail-closed halts |
| Runaway strategy | kill switch (manual + automatic), turnover/drawdown/daily-loss halts |
| Secrets leakage | no credentials exist in the codebase; documented policy for any future adapter |
| Dependency compromise | pinned requirements; CI runs ruff + compile + tests on 3 Python versions |
| Public exposure | no dashboards, no listening ports, no admin interfaces at all |

## Testnet note

Binance spot/futures testnets exist for ORDER placement, which this codebase
does not do. No testnet adapter is included; adding one later requires the
same safeguards as live trading (separate credentials, explicit operator
acknowledgement, reconciliation, kill switch).
