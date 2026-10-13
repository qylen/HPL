"""Command-line interface for the HPL-1 research and paper-trading pipeline."""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _print_json(value) -> None:
    print(json.dumps(value, indent=2, default=str))


def _add_market_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--market", default="futures", choices=["spot", "futures"])
    p.add_argument("--symbol", default=None, help="defaults to the profile symbol")
    p.add_argument("--config", default=None, help="YAML config file (validated)")


def _resolve(args, cli_overrides: dict | None = None):
    """Merge YAML config + CLI overrides into (research, cost, risk) configs."""
    from .config import (
        build_research_config,
        load_config,
    )

    data = load_config(args.config) if getattr(args, "config", None) else {}
    research, cost, risk = build_research_config(data)
    if getattr(args, "market", None):
        research.market = args.market
        research.symbol = None
    if getattr(args, "symbol", None):
        research.symbol = args.symbol.upper()
    if getattr(args, "start", None):
        research.start_date = args.start
    if getattr(args, "end", None):
        research.end_date = args.end
    research.__post_init__()
    for key, val in (cli_overrides or {}).items():
        if val is not None:
            setattr(research, key, val)
    risk.leverage = research.profile.validate_leverage(risk.leverage)
    return research, cost, risk


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hpl1",
        description=("Research one-second crypto order-flow predictability; historical "
                     "archive pipeline, cost-aware backtests and PAPER trading only. "
                     "This toolkit contains NO live-order execution client."),
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    env = sub.add_parser("env", help="Show environment and dependency versions")
    markets = sub.add_parser("markets", help="List supported market profiles")
    markets.add_argument("--json", action="store_true")

    d = sub.add_parser("data", help="Historical archive operations (data.binance.vision)")
    dsub = d.add_subparsers(dest="data_command", required=True)
    inspect = dsub.add_parser("inspect", help="List archive availability for a symbol/dtype")
    _add_market_args(inspect)
    inspect.add_argument("--dtype", default="bookTicker",
                         choices=["bookTicker", "aggTrades", "trades", "klines", "fundingRate", "bookDepth"])
    inspect.add_argument("--monthly", action="store_true")
    estimate = dsub.add_parser("estimate", help="Estimate download size for a date range")
    _add_market_args(estimate)
    estimate.add_argument("--dtype", default="bookTicker")
    estimate.add_argument("--start", required=True)
    estimate.add_argument("--end", required=True)
    download = dsub.add_parser("download", help="Download daily archives with checksums")
    _add_market_args(download)
    download.add_argument("--dtypes", default="bookTicker,aggTrades,fundingRate")
    download.add_argument("--start", required=True)
    download.add_argument("--end", required=True)
    download.add_argument("--yes", action="store_true",
                          help="acknowledge large downloads above the configured guard")
    validate = dsub.add_parser("validate", help="Re-validate checksums of downloaded archives")
    _add_market_args(validate)

    build = sub.add_parser("build", help="Ingest archives and assemble per-second datasets + quality report")
    _add_market_args(build)
    build.add_argument("--start", default=None)
    build.add_argument("--end", default=None)
    build.add_argument("--rebuild", action="store_true")

    compare = sub.add_parser("compare", help="Train and compare candidate models (selection on validation)")
    _add_market_args(compare)
    compare.add_argument("--models", default="zero,ridge,logistic,hist,law,regime")
    compare.add_argument("--start", default=None)
    compare.add_argument("--end", default=None)

    backtest = sub.add_parser("backtest", help="Cost-aware backtest on the stored test predictions")
    _add_market_args(backtest)
    backtest.add_argument("--taker-fee-bps", type=float, default=None)
    backtest.add_argument("--slippage-bps", type=float, default=None)
    backtest.add_argument("--fee-scenario", default=None, choices=["user", "zero", "promo"])
    backtest.add_argument("--leverage", type=float, default=None)

    evaluate = sub.add_parser("evaluate", help="Statistical battery + research report from stored artifacts")
    _add_market_args(evaluate)

    research = sub.add_parser("research", help="Full pipeline: build -> compare -> leakage -> backtest -> report")
    _add_market_args(research)
    research.add_argument("--start", default=None)
    research.add_argument("--end", default=None)
    research.add_argument("--models", default=None)
    research.add_argument("--download-missing", action="store_true")
    research.add_argument("--yes", action="store_true")

    leakage = sub.add_parser("validate-leakage", help="Run the leakage-detection suite on the dataset")
    _add_market_args(leakage)

    paper = sub.add_parser("paper", help="Run the live-data PAPER trading engine (no real orders)")
    _add_market_args(paper)
    paper.add_argument("--model", default="artifacts/research/best_model.joblib")
    paper.add_argument("--duration-seconds", type=float, default=0)
    paper.add_argument("--run-dir", default=None)
    paper.add_argument("--taker-fee-bps", type=float, default=None)
    paper.add_argument("--leverage", type=float, default=None)

    health = sub.add_parser("health", help="Show paper-service health snapshot")
    health.add_argument("--run-dir", default="data/paper/latest")
    kill = sub.add_parser("kill", help="Create the manual kill-switch file for a paper run")
    kill.add_argument("--run-dir", default="data/paper/latest")

    demo = sub.add_parser("demo", help="Run the pipeline on planted synthetic data (not market evidence)")
    demo.add_argument("--rows", type=int, default=20_000)
    demo.add_argument("--directory", default="artifacts/demo")
    demo.add_argument("--model", choices=["ridge", "hist"], default="ridge")
    demo.add_argument("--taker-fee-bps", type=float, default=0.0)
    demo.add_argument("--slippage-bps", type=float, default=0.0)
    demo.add_argument("--noise", action="store_true", help="no planted signal (control dataset)")

    synthetic_e2e = sub.add_parser("synthetic-e2e", help="Full pipeline on synthetic Binance-format archives")
    synthetic_e2e.add_argument("--directory", default="artifacts/synthetic_e2e")
    synthetic_e2e.add_argument("--noise", action="store_true")
    synthetic_e2e.add_argument("--corrupt", action="store_true", help="inject duplicates/out-of-order/crossed/gaps")

    tests = sub.add_parser("test", help="Print the automated test invocation")

    # ---- legacy commands (JSONL collection path) ----
    collect = sub.add_parser("collect", help="Record public market events to JSONL (legacy path)")
    collect.add_argument("--symbol", default="btcusdt")
    collect.add_argument("--market", default="futures", choices=["spot", "futures"])
    collect.add_argument("--output", default="data/raw/events.jsonl")
    collect.add_argument("--duration-seconds", type=float, default=0)
    legacy_build = sub.add_parser("build-features", help="Build features from collected JSONL (legacy)")
    legacy_build.add_argument("--input", default="data/raw/events.jsonl")
    legacy_build.add_argument("--output", default="data/processed/features.csv")
    train = sub.add_parser("train", help="Legacy single-model training on a feature CSV")
    train.add_argument("--data", default="data/processed/features.csv")
    train.add_argument("--model", choices=["ridge", "hist"], default="ridge")
    train.add_argument("--output", default="artifacts/hpl1.joblib")
    train.add_argument("--predictions", default="artifacts/test_predictions.csv")
    legacy_bt = sub.add_parser("backtest-legacy", help="Legacy {-1,0,+1} backtest on prediction CSVs")
    legacy_bt.add_argument("--predictions", default="artifacts/test_predictions.csv")
    legacy_bt.add_argument("--output", default="artifacts/backtest.csv")
    legacy_bt.add_argument("--report", default="artifacts/backtest_report.json")
    legacy_bt.add_argument("--taker-fee-bps", type=float, default=5.0)
    legacy_bt.add_argument("--slippage-bps", type=float, default=1.0)
    legacy_bt.add_argument("--min-edge-bps", type=float, default=1.0)
    legacy_bt.add_argument("--risk-buffer-bps", type=float, default=0.0)
    legacy_bt.add_argument("--hac-lags", type=int, default=60)
    return parser


# --------------------------------------------------------------------- helpers

def _artifact_dir(research) -> Path:
    d = Path(research.artifacts_dir) / f"{research.market}_{research.symbol}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_feature_frame(research):
    from .dataset import load_feature_frame

    return load_feature_frame(research.market, research.symbol, research.start_date,
                              research.end_date, research.processed_dir)


def _feature_set(research) -> str:
    return "trade_only" if research.profile.market_type == "spot" else "full"


# ------------------------------------------------------------------- commands

def _cmd_env(_args) -> None:
    import sklearn
    import yaml

    from . import __version__

    _print_json({
        "hpl1_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "pyyaml": yaml.__version__,
        "live_order_execution_client": "ABSENT BY DESIGN (research/paper only)",
    })


def _cmd_markets(args) -> None:
    from .markets import list_markets

    if args.json:
        _print_json(list_markets())
    else:
        for m in list_markets():
            print(f"{m['name']:9s} {m['symbol']:10s} type={m['market_type']:11s} "
                  f"short={'yes' if m['allow_short'] else 'no ':3s} lev_cap={m['leverage_cap']} "
                  f"funding={'yes' if m['has_funding'] else 'no'}")


def _cmd_data_inspect(args) -> None:
    from .binance_archive import list_archive_files
    from .markets import get_market

    profile = get_market(args.market)
    symbol = (args.symbol or profile.symbol).upper()
    keys = list_archive_files(args.market, symbol, args.dtype, period="monthly" if args.monthly else "daily")
    if not keys:
        print(f"[inspect] NO archives found for {args.market}/{symbol}/{args.dtype} "
              f"({'monthly' if args.monthly else 'daily'}).")
        print("          This dataset may not exist for this market (e.g. spot publishes "
              "no book history). Nothing was downloaded.")
        return
    print(f"[inspect] {args.market}/{symbol}/{args.dtype} "
          f"({'monthly' if args.monthly else 'daily'}): {len(keys)} archive(s)")
    for k in keys[:10]:
        print(f"  {k.date_label}  {k.size / 1e6:9.1f} MB  {k.key}")
    if len(keys) > 10:
        print(f"  ... and {len(keys) - 10} more (first={keys[0].date_label}, last={keys[-1].date_label})")


def _cmd_data_estimate(args) -> None:
    from .binance_archive import estimate_download
    from .markets import get_market

    profile = get_market(args.market)
    symbol = (args.symbol or profile.symbol).upper()
    _print_json(estimate_download(args.market, symbol, args.dtype, args.start, args.end))


def _cmd_data_download(args) -> None:
    from .downloader import download_range, print_results, save_manifest
    from .markets import get_market

    profile = get_market(args.market)
    symbol = (args.symbol or profile.symbol).upper()
    research, _, _ = _resolve(args)
    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    if profile.market_type == "spot":
        dtypes = [d for d in dtypes if d != "fundingRate"]
    all_results = []
    for dtype in dtypes:
        results, manifest = download_range(
            args.market, symbol, dtype, args.start, args.end, research.raw_dir,
            max_concurrency=research.max_download_concurrency, acknowledged=args.yes,
            large_download_allow_mb=research.large_download_allow_mb)
        print_results(results, f"{args.market}/{symbol}/{dtype}")
        all_results.extend(results)
        mdir = _artifact_dir(research) / "downloads" / dtype
        save_manifest(manifest, mdir / f"manifest_{args.start}_{args.end}.json")
    failed = [r for r in all_results if r.status == "failed"]
    _print_json({"files": len(all_results), "failed": len(failed),
                 "raw_dir": str(Path(research.raw_dir))})
    if failed:
        sys.exit(1)


def _cmd_data_validate(args) -> None:
    from .binance_archive import archive_key
    from .downloader import sha256_of_file

    research, _, _ = _resolve(args)
    symbol = research.symbol
    ok = missing = failed = 0
    for dtype_dir in (Path(research.raw_dir) / research.market / symbol).glob("*"):
        if not dtype_dir.is_dir():
            continue
        dtype = dtype_dir.name
        for z in sorted(dtype_dir.glob("*.zip")):
            # SYMBOL-dtype-YYYY-MM-DD.zip (daily) or SYMBOL-dtype-YYYY-MM.zip (monthly)
            tail = z.stem.split("-")
            n_sym = len(symbol.split("-"))
            date_parts = tail[n_sym + 1:]  # skip symbol and dtype tokens
            period = "monthly" if len(date_parts) == 2 else "daily"
            day = "-".join(date_parts)
            key = archive_key(research.market, symbol, dtype, day, period=period)
            import requests

            try:
                resp = requests.get(key.checksum_url, timeout=15)
                expected = resp.text.split()[0].lower() if resp.status_code == 200 else None
            except requests.RequestException:
                expected = None
            if expected is None:
                missing += 1
                print(f"NOCRC {z.name} (no official checksum)")
                continue
            if sha256_of_file(z) == expected:
                ok += 1
                print(f"OK    {z.name}")
            else:
                failed += 1
                print(f"FAIL  {z.name} (checksum mismatch — re-download recommended)")
    _print_json({"ok": ok, "no_checksum": missing, "failed": failed})
    if failed:
        sys.exit(1)


def _cmd_build(args) -> None:
    from .binance_archive import iter_dates
    from .dataset import build_dataset_days, load_dataset
    from .features import build_features
    from .ingest import ingest_archive
    from .quality import quality_markdown, quality_report

    research, _, _ = _resolve(args)
    profile = research.profile
    dtypes = [d for d in research.data_types
              if (Path(research.raw_dir) / research.market / research.symbol / d).exists()]
    if not dtypes:
        print(f"[build] no raw archives under {research.raw_dir}/{research.market}/{research.symbol}; "
              "run `hpl1 data download` first.")
        sys.exit(1)
    ingest_stats_total = {"dup_events": 0, "out_of_order": 0, "crossed_quotes": 0, "applied_events": 0}
    from .ingest import ingest_funding_monthly

    for dtype in dtypes:
        if dtype == "fundingRate":
            # fundingRate is published monthly-only in the public archive
            months = sorted({day.isoformat()[:7]
                             for day in iter_dates(research.start_date, research.end_date)})
            days = [d.isoformat() for d in iter_dates(research.start_date, research.end_date)]
            for month in months:
                try:
                    r = ingest_funding_monthly(research.market, research.symbol, month, days,
                                               research.raw_dir, research.processed_dir,
                                               rebuild=args.rebuild)
                    print(f"[ingest] fundingRate {month}: {r}")
                except FileNotFoundError as exc:
                    print(f"[ingest] WARNING missing monthly funding archive {month}: {exc}")
            continue
        for day in iter_dates(research.start_date, research.end_date):
            d = day.isoformat()
            try:
                r = ingest_archive(research.market, research.symbol, dtype, d,
                                   research.raw_dir, research.processed_dir, rebuild=args.rebuild)
                if r["status"] == "ok":
                    print(f"[ingest] {dtype} {d}: rows={r['rows']} stats={r['stats']}")
                    if r["stats"]:
                        for k in ingest_stats_total:
                            ingest_stats_total[k] += r["stats"].get(k, 0)
            except FileNotFoundError as exc:
                print(f"[ingest] WARNING missing archive for {dtype} {d}: {exc}")
    summaries = build_dataset_days(research.market, research.symbol, research.start_date,
                                   research.end_date, research.processed_dir, rebuild=args.rebuild)
    for s in summaries:
        print(f"[dataset] {s['day']}: {s}")
    frame = load_dataset(research.market, research.symbol, research.start_date,
                         research.end_date, research.processed_dir)
    feature_set = _feature_set(research)
    feats = build_features(frame, feature_set=feature_set)
    report = quality_report(feats, ingest_stats=ingest_stats_total)
    adir = _artifact_dir(research)
    (adir / "quality_report.json").write_text(json.dumps(report, indent=2, default=str))
    (adir / "quality_report.md").write_text(
        quality_markdown(report, title=f"Data quality — {research.market} {research.symbol} "
                                       f"{research.start_date}..{research.end_date}"))
    feats.to_parquet(adir / "feature_frame.parquet", index=False)
    _print_json({"feature_set": feature_set, "rows": len(feats), **report})


def _cmd_compare(args) -> None:
    from .features import feature_columns_for
    from .training import compare_models

    research, _, _ = _resolve(args, {"models": [m.strip() for m in args.models.split(",")]})
    feats = _load_feature_frame(research)
    feature_set = _feature_set(research)
    fcols = [c for c in feature_columns_for(feature_set) if c in feats.columns]
    adir = _artifact_dir(research)
    out = compare_models(feats, fcols, research.models, seed=research.random_seed,
                         train_fraction=research.train_fraction,
                         validation_fraction=research.validation_fraction,
                         purge=research.purge_seconds, embargo=research.embargo_seconds,
                         artifact_dir=str(adir),
                         run_manifest={"market": research.market, "symbol": research.symbol,
                                       "days": f"{research.start_date}..{research.end_date}"})
    best_path = out["saved"].get(out["best"])
    if best_path:
        import shutil

        shutil.copy(best_path, adir / "best_model.joblib")
    summary = {name: {"validation_r2": r["validation"].get("oos_r2_vs_zero"),
                      "test_r2": r.get("test", {}).get("oos_r2_vs_zero")}
               for name, r in out["results"].items()}
    _print_json({"selected": out["best"], "summary": summary, "manifest": out["manifest"]})


def _run_backtest_with(research, cost, risk, fee_override=None, slip_override=None,
                       scenario=None, leverage=None) -> dict:
    from .backtest import run_cost_backtest
    from .exchange_meta import fetch_symbol_rules
    from .models import load_model

    if fee_override is not None:
        cost.taker_fee_bps = float(fee_override)
    if slip_override is not None:
        cost.slippage_bps = float(slip_override)
    if scenario:
        cost.fee_scenario = scenario
        cost.__post_init__()
    if leverage is not None:
        risk.leverage = leverage
    adir = _artifact_dir(research)
    pred = pd.read_parquet(adir / "test_predictions.parquet")
    model, _ = load_model(adir / "best_model.joblib")
    rules = fetch_symbol_rules(research.market, research.symbol)
    out = run_cost_backtest(pred, cost, risk, research.profile,
                            tick_size=rules.tick_size, step_size=rules.step_size,
                            min_notional=rules.min_notional)
    return out


def _cmd_backtest(args) -> None:

    research, cost, risk = _resolve(args)
    if args.fee_scenario:
        cost.fee_scenario = args.fee_scenario
    out = _run_backtest_with(research, cost, risk, fee_override=args.taker_fee_bps,
                             slip_override=args.slippage_bps, leverage=args.leverage)
    adir = _artifact_dir(research)
    out["curve"].to_parquet(adir / "backtest_curve.parquet", index=False)
    out["trades"].to_csv(adir / "backtest_trades.csv", index=False)
    out["round_trips"].to_csv(adir / "backtest_round_trips.csv", index=False)
    (adir / "backtest_report.json").write_text(json.dumps(out["report"], indent=2, default=str))
    _print_json(out["report"])


def _cmd_evaluate(args) -> None:
    from .evaluate import research_report, statistical_battery, write_report_artifacts
    from .models import load_model

    research, cost, risk = _resolve(args)
    adir = _artifact_dir(research)
    pred = pd.read_parquet(adir / "test_predictions.parquet")
    with open(adir / "training_manifest.json") as fh:
        manifest = json.load(fh)
    quality = json.loads((adir / "quality_report.json").read_text())
    battery = statistical_battery(pred["prediction"].to_numpy(), pred["actual_return"].to_numpy())
    bt_path = adir / "backtest_report.json"
    backtest = json.loads(bt_path.read_text()) if bt_path.exists() else None
    model, _ = load_model(adir / "best_model.joblib")
    report = research_report(research.market, research.symbol,
                             f"{research.start_date}..{research.end_date}",
                             comparison={"results": {}, "best": manifest.get("selected_model", "n/a")},
                             backtest={"report": backtest} if backtest else None,
                             battery=battery, quality=quality,
                             config_meta={"manifest": manifest, "costs": vars(cost), "risk": vars(risk)})
    report["selected_model"] = manifest.get("selected_model")
    report["test_metrics"] = manifest.get("test_metrics", {})
    paths = write_report_artifacts(report, adir)
    _print_json({"report_paths": paths, "verdict": report["verdict"]})


def _cmd_research(args) -> None:
    """One command: build -> compare -> leakage suite -> backtest -> battery -> report."""
    from .backtest import fee_sensitivity, run_cost_backtest
    from .dataset import load_dataset
    from .evaluate import research_report, statistical_battery, write_report_artifacts
    from .exchange_meta import fetch_symbol_rules
    from .features import build_features, feature_columns_for
    from .training import compare_models
    from .validation import run_leakage_suite

    research, cost, risk = _resolve(args, {"models": None} if not args.models else
                                    {"models": [m.strip() for m in args.models.split(",")]})
    adir = _artifact_dir(research)
    t_start = time.time()

    # 1. data present?
    raw_root = Path(research.raw_dir) / research.market / research.symbol
    missing_archives = []
    from .binance_archive import iter_dates

    for dtype in research.data_types:
        for day in iter_dates(research.start_date, research.end_date):
            f = raw_root / dtype / f"{research.symbol}-{dtype}-{day.isoformat()}.zip"
            if dtype == "fundingRate":
                # fundingRate is published monthly-only in the public archive
                f = raw_root / dtype / f"{research.symbol}-{dtype}-{day.isoformat()[:7]}.zip"
            if not f.exists():
                missing_archives.append(str(f))
    if missing_archives and args.download_missing:
        args.dtypes = ",".join(research.data_types)
        args.yes = args.yes or True
        _cmd_data_download(args)
        missing_archives = []
    if missing_archives and not args.download_missing:
        print(f"[research] {len(missing_archives)} archive(s) missing; "
              "run with --download-missing or `hpl1 data download` first.")
        sys.exit(1)

    # 2. ingest + features (reuse build command logic)
    args.rebuild = False
    _cmd_build(args)
    raw_frame = load_dataset(research.market, research.symbol, research.start_date,
                             research.end_date, research.processed_dir)
    feature_set = _feature_set(research)
    feats = build_features(raw_frame, feature_set=feature_set)
    fcols = [c for c in feature_columns_for(feature_set) if c in feats.columns]

    # 3. leakage suite
    def _bf(df):
        return build_features(df, feature_set=feature_set)

    leakage = run_leakage_suite(feats, build_fn=_bf, raw_df=raw_frame, feature_columns=fcols)
    (adir / "leakage_suite.json").write_text(json.dumps(leakage, indent=2, default=str))
    print(f"[research] leakage suite passed={leakage['passed']}")

    # 4. model comparison
    out = compare_models(feats, fcols, research.models, seed=research.random_seed,
                         train_fraction=research.train_fraction,
                         validation_fraction=research.validation_fraction,
                         purge=research.purge_seconds, embargo=research.embargo_seconds,
                         artifact_dir=str(adir),
                         run_manifest={"market": research.market, "symbol": research.symbol,
                                       "days": f"{research.start_date}..{research.end_date}",
                                       "leakage_passed": leakage["passed"]})
    import shutil

    shutil.copy(out["saved"][out["best"]], adir / "best_model.joblib")
    (adir / "training_manifest.json").write_text(json.dumps(out["manifest"], indent=2, default=str))

    # 5. cost-aware backtest (user fee scenario)
    rules = fetch_symbol_rules(research.market, research.symbol)
    pred_frame = out["predictions"]
    bt = run_cost_backtest(pred_frame, cost, risk, research.profile,
                           tick_size=rules.tick_size, step_size=rules.step_size,
                           min_notional=rules.min_notional)
    bt["curve"].to_parquet(adir / "backtest_curve.parquet", index=False)
    bt["trades"].to_csv(adir / "backtest_trades.csv", index=False)
    (adir / "backtest_report.json").write_text(json.dumps(bt["report"], indent=2, default=str))

    # 6. fee sensitivity + explicit zero-fee research scenario.
    # These are EDGE MEASUREMENTS: an unconstrained risk configuration isolates
    # the cost dimension (turnover caps/cooldowns would truncate the signal).
    # The main backtest above uses the configured (conservative) risk limits.
    from .config import CostConfig, RiskConfig

    edge_risk = RiskConfig(
        starting_capital=risk.starting_capital,
        leverage=risk.leverage,
        position_fraction=1.0,
        cooldown_seconds=0,
        max_turnover_per_day=1e9,
        max_daily_loss_pct=0.0,   # halts disabled for measurement
        max_drawdown_pct=0.0,
        min_edge_bps=0.0,
        stale_quote_max_seconds=risk.stale_quote_max_seconds,
    )
    fee_table = fee_sensitivity(pred_frame, cost, edge_risk, research.profile,
                                tick_size=rules.tick_size, step_size=rules.step_size,
                                min_notional=rules.min_notional)
    fee_table.to_csv(adir / "fee_sensitivity.csv", index=False)
    zero_cost = CostConfig(taker_fee_bps=0.0, maker_fee_bps=0.0,
                           slippage_bps=0.0, use_spread=cost.use_spread,
                           fee_scenario="zero")  # gross edge: fees AND slippage zeroed
    bt_zero = run_cost_backtest(pred_frame, zero_cost, edge_risk, research.profile,
                                tick_size=rules.tick_size, step_size=rules.step_size,
                                min_notional=rules.min_notional)
    bt_zero["report"]["measurement_note"] = (
        "UNCONSTRAINED risk config (no turnover cap/cooldown/halts): measures the "
        "raw economic content of the signal net of spread only. Not a tradable "
        "strategy configuration.")
    (adir / "backtest_zero_fee.json").write_text(json.dumps(bt_zero["report"], indent=2, default=str))

    # 7. statistical battery + research report
    val_preds = out.get("validation_predictions") or {}
    battery = statistical_battery(pred_frame["prediction"].to_numpy(),
                                  pred_frame["actual_return"].to_numpy(),
                                  seed=research.random_seed,
                                  calibration_fit=(val_preds.get("prediction"),
                                                   val_preds.get("actual"))
                                  if val_preds.get("prediction") is not None else None)
    quality = json.loads((adir / "quality_report.json").read_text())
    report = research_report(research.market, research.symbol,
                             f"{research.start_date}..{research.end_date}",
                             comparison=out, backtest=bt, battery=battery, quality=quality,
                             config_meta={"costs": vars(cost), "risk": vars(risk),
                                          "seed": research.random_seed},
                             fee_table=fee_table)
    paths = write_report_artifacts(report, adir)
    _print_json({
        "elapsed_seconds": round(time.time() - t_start, 1),
        "selected_model": out["best"],
        "leakage_passed": leakage["passed"],
        "test_metrics": out["results"][out["best"]]["test"],
        "backtest_net_return_pct": bt["report"]["total_net_return_pct"],
        "backtest_trades": bt["report"]["n_trades"],
        "zero_fee_net_return_pct": bt_zero["report"]["total_net_return_pct"],
        "fee_sensitivity": fee_table.to_dict(orient="records"),
        "verdict": report["verdict"],
        "report_paths": paths,
    })


def _cmd_validate_leakage(args) -> None:
    from .dataset import load_dataset
    from .features import build_features, feature_columns_for
    from .training import compare_models  # noqa: F401  (splits checked when run)
    from .validation import (
        run_leakage_suite,
    )

    research, _, _ = _resolve(args)
    raw = load_dataset(research.market, research.symbol, research.start_date,
                       research.end_date, research.processed_dir)
    feature_set = _feature_set(research)
    feats = build_features(raw, feature_set=feature_set)
    fcols = [c for c in feature_columns_for(feature_set) if c in feats.columns]
    suite = run_leakage_suite(feats, build_fn=lambda df: build_features(df, feature_set=feature_set),
                              raw_df=raw, feature_columns=fcols)
    _print_json(suite)
    if not suite["passed"]:
        sys.exit(1)


def _cmd_paper(args) -> None:
    from .live import run_paper_session
    from .models import load_model

    research, cost, risk = _resolve(args)
    if args.taker_fee_bps is not None:
        cost.taker_fee_bps = float(args.taker_fee_bps)
    if args.leverage is not None:
        risk.leverage = args.leverage
    model, meta = load_model(args.model)
    run_dir = args.run_dir or f"data/paper/{research.market}_{int(time.time())}"
    Path("data/paper").mkdir(parents=True, exist_ok=True)
    try:
        Path("data/paper/latest").unlink()
    except (OSError, FileNotFoundError):
        pass
    try:
        Path("data/paper/latest").symlink_to(Path(run_dir).resolve(), target_is_directory=True)
    except (OSError, NotImplementedError):
        pass
    summary = asyncio.run(run_paper_session(research.profile, model, cost, risk, run_dir,
                                            duration_seconds=args.duration_seconds))
    _print_json(summary)


def _cmd_health(args) -> None:
    h = Path(args.run_dir) / "health.json"
    if not h.exists():
        print(f"no health snapshot at {h}; is the paper engine running?")
        sys.exit(1)
    _print_json(json.loads(h.read_text()))


def _cmd_kill(args) -> None:
    p = Path(args.run_dir) / "kill_switch"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print(f"kill switch created: {p} — the running paper engine halts within ~1s; "
          "restart required to resume (see docs/RUNBOOK_PAPER_TRADING.md)")


def _cmd_demo(args) -> None:
    from .backtest import backtest_files
    from .synthetic import generate_synthetic_dataset
    from .training import train_and_evaluate

    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    dataset = directory / "synthetic_features.csv"
    artifact = directory / "model.joblib"
    predictions = directory / "test_predictions.csv"
    generate_synthetic_dataset(dataset, rows=args.rows, planted=not args.noise)
    trained = train_and_evaluate(dataset, args.model, artifact, predictions)
    backtested = backtest_files(predictions, directory / "backtest.csv",
                                directory / "backtest_report.json",
                                taker_fee_bps=args.taker_fee_bps,
                                slippage_bps=args.slippage_bps)
    _print_json({"note": ("Synthetic data with a planted PREDICTIVE signal checks software "
                          "flow only; it is NOT evidence of a real-market edge."
                          if not args.noise else
                          "Control dataset with NO planted signal; the pipeline must NOT "
                          "report a profitable edge here."),
                 "training": trained, "backtest": backtested, "directory": str(directory)})


def _cmd_synthetic_e2e(args) -> None:
    from .backtest import run_cost_backtest
    from .config import CostConfig
    from .dataset import build_dataset_days, load_dataset
    from .features import build_features, feature_columns_for
    from .markets import get_market
    from .synthetic import write_synthetic_archive
    from .training import compare_models

    directory = Path(args.directory)
    raw_dir = directory / "raw"
    proc_dir = directory / "processed"
    info = write_synthetic_archive(raw_dir, market="futures", symbol="SYNUSD",
                                   day="2025-06-01", seconds=7200,
                                   planted=not args.noise,
                                   gap_seconds=40 if args.corrupt else 0,
                                   duplicate_events=25 if args.corrupt else 0,
                                   out_of_order_events=15 if args.corrupt else 0,
                                   crossed_quotes=10 if args.corrupt else 0)
    from .ingest import ingest_archive

    for dtype in ("bookTicker", "aggTrades", "fundingRate"):
        r = ingest_archive("futures", "SYNUSD", dtype, "2025-06-01", raw_dir, proc_dir)
        print(f"[e2e ingest] {dtype}: {r['stats']}")
    build_dataset_days("futures", "SYNUSD", "2025-06-01", "2025-06-01", proc_dir)
    raw = load_dataset("futures", "SYNUSD", "2025-06-01", "2025-06-01", proc_dir)
    feats = build_features(raw, feature_set="full")
    fcols = [c for c in feature_columns_for("full") if c in feats.columns]
    out = compare_models(feats, fcols, ["zero", "ridge", "hist", "law"], seed=13,
                         train_fraction=0.6, validation_fraction=0.2, purge=1, embargo=30)
    from .config import RiskConfig

    cost = CostConfig(taker_fee_bps=0.0, slippage_bps=0.0, fee_scenario="zero")
    risk = RiskConfig(starting_capital=10_000)
    bt = run_cost_backtest(out["predictions"], cost, risk, get_market("futures"),
                           tick_size="0.01", step_size="0.001", min_notional=5.0)
    _print_json({
        "planted": info["planted"], "corrupt": args.corrupt,
        "test_metrics": out["results"][out["best"]]["test"],
        "zero_cost_backtest": {k: bt["report"][k] for k in
                               ("n_trades", "n_round_trips", "total_net_return_pct")},
        "note": "synthetic archives verify pipeline mechanics; not market evidence",
    })


def _cmd_test(_args) -> None:
    print("Offline suite:   python -m pytest -q")
    print("Network suite:   python -m pytest -q -m network   (requires internet)")
    print("Lint:            ruff check hpl1 tests")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    import logging

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    if args.command == "env":
        _cmd_env(args)
    elif args.command == "markets":
        _cmd_markets(args)
    elif args.command == "data":
        {"inspect": _cmd_data_inspect, "estimate": _cmd_data_estimate,
         "download": _cmd_data_download, "validate": _cmd_data_validate}[args.data_command](args)
    elif args.command == "build":
        _cmd_build(args)
    elif args.command == "compare":
        _cmd_compare(args)
    elif args.command == "backtest":
        _cmd_backtest(args)
    elif args.command == "evaluate":
        _cmd_evaluate(args)
    elif args.command == "research":
        _cmd_research(args)
    elif args.command == "validate-leakage":
        _cmd_validate_leakage(args)
    elif args.command == "paper":
        _cmd_paper(args)
    elif args.command == "health":
        _cmd_health(args)
    elif args.command == "kill":
        _cmd_kill(args)
    elif args.command == "demo":
        _cmd_demo(args)
    elif args.command == "synthetic-e2e":
        _cmd_synthetic_e2e(args)
    elif args.command == "test":
        _cmd_test(args)
    elif args.command == "collect":
        from .collector import collect_events

        try:
            asyncio.run(collect_events(args.symbol, args.output, args.duration_seconds,
                                       market=args.market))
        except KeyboardInterrupt:
            print("Collection interrupted by user.")
    elif args.command == "build-features":
        from .features import build_from_jsonl, save_feature_csv

        frame = build_from_jsonl(args.input)
        save_feature_csv(frame, args.output)
        _print_json({"rows": len(frame), "output": args.output})
    elif args.command == "train":
        from .training import train_and_evaluate

        _print_json(train_and_evaluate(args.data, args.model, args.output, args.predictions))
    elif args.command == "backtest-legacy":
        from .backtest import backtest_files

        _print_json(backtest_files(args.predictions, args.output, args.report,
                                   args.taker_fee_bps, args.slippage_bps,
                                   args.min_edge_bps, args.risk_buffer_bps, args.hac_lags))


if __name__ == "__main__":
    main()
