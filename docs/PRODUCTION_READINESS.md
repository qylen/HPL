# Production Readiness Checklist

## Implemented and verified

- [x] Clear package structure (`hpl1/`, 21 modules, `pyproject.toml`).
- [x] Pinned dependencies (requirements.txt) + dev extras.
- [x] Validated YAML configuration; malformed or live-trading configs rejected.
- [x] Structured logs with rotation (5 MB × 5 backups).
- [x] Error reporting: every failure path logged with context; boundary-tick
      errors never kill the loop.
- [x] Health checks: `hpl1 health` + health.json heartbeat (book age, event
      rate, decisions, equity, kill status).
- [x] Service restart: state.json resume; docker-compose `restart: unless-stopped`.
- [x] Data-drift monitoring hooks: decision logs carry prediction/hurdle;
      training manifest records the training distribution.
- [x] Model-version tracking: manifest (config, seed, splits, days, git SHA)
      saved beside every model artifact.
- [x] Disk-space checks before downloads and paper runs (fail closed < 1 GB).
- [x] Durable research artifacts under artifacts/research/<market>_<symbol>/.
- [x] Docker + docker-compose (research + paper services, non-root user).
- [x] CI: GitHub Actions — ruff, compileall, offline pytest on 3.10/3.11/3.12,
      CLI smoke (demo).
- [x] Deployment checklist + rollback plan (docs/DEPLOYMENT.md).
- [x] Paper-trading runbook (docs/RUNBOOK_PAPER_TRADING.md).
- [x] Security review + threat model (docs/SECURITY.md).
- [x] Backup/retention guidance (docs/DEPLOYMENT.md).
- [x] No dashboards or listening ports; nothing to expose.

## Known gaps (documented, non-blocking for research/paper use)

- [ ] No type-checking job (mypy) in CI — lint + compile + tests only.
- [ ] No published package (PyPI); install is repo-local.
- [ ] Funding uses mid price, not mark price (small approximation).
- [ ] No maker-fill model (taker-only execution simulation).
- [ ] Single 5-day research window ingested; the pipeline supports the full
      2023-05..2024-03 futures book window via config.

## Live trading: NOT READY (by design)

No execution client exists. Activation would additionally require: an
authenticated adapter with the safeguards in docs/SECURITY.md, testnet
integration tests, exchange state reconciliation, operator acknowledgement
records, and a paper-trading track record. See docs/ENGINEERING_REPORT.md §10.
