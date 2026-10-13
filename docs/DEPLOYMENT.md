# Deployment Checklist and Rollback Plan

## Developer machine (verified path)

1. `git clone <repo> && cd HPL`
2. `python -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python -m pytest -q` — must pass (offline suite).
5. Follow docs/REPRODUCE.md.

## Linux server

1. Provision ≥2 CPU / 4 GB RAM / ≥20 GB free disk (5-day research data needs
   ~1.2 GB raw + ~1 GB processed; the full 2023-05..2024-03 book history is
   ~50 GB compressed — check before downloading).
2. Install Python 3.10–3.12 and the package (or use Docker: `docker compose
   build`).
3. Disk-space guard: `hpl1 paper` and the research pipeline check free space
   (`logging_setup.check_disk_space`, min 1 GB) and fail closed when low.
4. Run the research pipeline once to produce `best_model.joblib`.
5. Start paper trading under a supervisor:
   - Docker: `docker compose up -d paper` (restart: unless-stopped).
   - systemd: `Restart=on-failure`, `ExecStart=python -m hpl1 paper ...`.
6. Confirm health: `hpl1 health` shows book_valid=true, events increasing.

## Pre-deployment checklist

- [ ] `python -m pytest -q` green; `python -m pytest -q -m network` green/skipped-with-reason.
- [ ] `ruff check hpl1 tests` clean; `python -m compileall -q hpl1` clean.
- [ ] Archive checksums validated (`hpl1 data validate`).
- [ ] Config validated; fee tier replaced with the account's real schedule.
- [ ] Kill-switch procedure rehearsed (`hpl1 kill` → halt within ~1 s verified).
- [ ] Logs rotating (RotatingFileHandler 5 MB × 5) and disk-space check green.
- [ ] Data drift monitoring active (paper health.json: book age, event rate,
      prediction distribution vs training manifest).
- [ ] Model version recorded (training_manifest.json + git SHA in artifacts).

## Rollback plan

1. **Bad model / drift detected:** `hpl1 kill` → paper engine halts ≤1 s.
   Swap `best_model.joblib` to the previous artifact (every compare run
   saves `model_<name>.joblib` plus manifest), remove the kill file, restart.
2. **Bad data day:** delete `data/processed/<market>/<symbol>/<dtype>/<day>.parquet`
   and re-run `hpl1 build --rebuild`. Raw archives are never modified.
3. **Bad release:** `git checkout <previous-tag>` and restart the service;
   processed Parquet schemas are stable across the 1.x series.
4. **Full reset:** `rm -rf data/processed artifacts` and re-run the pipeline;
   raw data (validated) is retained — regeneration is deterministic given the
   same seed (13).

## Backup and retention

- Code: git (including the local phase commits).
- Raw archives: re-downloadable from the public archive; retain if bandwidth
  matters. Checksums make re-validation cheap.
- Research artifacts (models, manifests, reports): small; back up the whole
  `artifacts/research/` tree with each release.
- Paper logs (JSONL): rotate weekly; retain 90 days for post-mortems.

## What deployment does NOT include

Any live-order execution. See docs/SECURITY.md for the path a future
execution adapter would have to satisfy before activation.
