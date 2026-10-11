"""Command-line interface for the HPL-1 research pipeline."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from .backtest import backtest_files
from .collector import collect_events
from .features import build_from_jsonl, save_feature_csv
from .synthetic import generate_synthetic_dataset
from .training import train_and_evaluate


def _print_json(value) -> None:
    print(json.dumps(value, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hpl1",
        description="Research one-second crypto order-flow predictability. This toolkit does not place live orders.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="Record public Binance USD-M Futures bookTicker and aggTrade events")
    collect.add_argument("--symbol", default="btcusdt", help="Symbol, e.g. btcusdt or ethusdt")
    collect.add_argument("--output", default="data/raw/events.jsonl")
    collect.add_argument("--duration-seconds", type=float, default=0, help="0 means run until Ctrl+C")

    build = sub.add_parser("build", help="Build one-second features and next-second labels from the event log")
    build.add_argument("--input", default="data/raw/events.jsonl")
    build.add_argument("--output", default="data/processed/features.csv")

    train = sub.add_parser("train", help="Train and evaluate a chronological holdout model")
    train.add_argument("--data", default="data/processed/features.csv")
    train.add_argument("--model", choices=["ridge", "hist"], default="ridge")
    train.add_argument("--output", default="artifacts/hpl1.joblib")
    train.add_argument("--predictions", default="artifacts/test_predictions.csv")
    train.add_argument("--train-fraction", type=float, default=0.60)
    train.add_argument("--validation-fraction", type=float, default=0.20)
    train.add_argument("--ridge-alpha", type=float, default=10.0)

    backtest = sub.add_parser("backtest", help="Run cost-aware backtest on untouched test predictions")
    backtest.add_argument("--predictions", default="artifacts/test_predictions.csv")
    backtest.add_argument("--output", default="artifacts/backtest.csv")
    backtest.add_argument("--report", default="artifacts/backtest_report.json")
    backtest.add_argument("--taker-fee-bps", type=float, default=5.0, help="Actual per-side taker fee in basis points")
    backtest.add_argument("--slippage-bps", type=float, default=1.0, help="Assumed additional slippage per side in basis points")
    backtest.add_argument("--min-edge-bps", type=float, default=1.0)
    backtest.add_argument("--risk-buffer-bps", type=float, default=0.0)
    backtest.add_argument("--hac-lags", type=int, default=60)

    demo = sub.add_parser("demo", help="Run the complete training/backtest pipeline on planted synthetic data (not market evidence)")
    demo.add_argument("--rows", type=int, default=20_000)
    demo.add_argument("--directory", default="artifacts/demo")
    demo.add_argument("--model", choices=["ridge", "hist"], default="ridge")
    demo.add_argument("--taker-fee-bps", type=float, default=0.0)
    demo.add_argument("--slippage-bps", type=float, default=0.0)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "collect":
        try:
            asyncio.run(collect_events(args.symbol, args.output, args.duration_seconds))
        except KeyboardInterrupt:
            print("Collection interrupted by user.")
    elif args.command == "build":
        frame = build_from_jsonl(args.input)
        save_feature_csv(frame, args.output)
        _print_json({"rows": len(frame), "output": args.output, "features": 20, "labeled_rows": int(frame["target_return"].notna().sum())})
    elif args.command == "train":
        result = train_and_evaluate(
            args.data, args.model, args.output, args.predictions,
            args.train_fraction, args.validation_fraction, args.ridge_alpha,
        )
        _print_json(result)
    elif args.command == "backtest":
        result = backtest_files(
            args.predictions, args.output, args.report,
            args.taker_fee_bps, args.slippage_bps, args.min_edge_bps,
            args.risk_buffer_bps, args.hac_lags,
        )
        _print_json(result)
    elif args.command == "demo":
        directory = Path(args.directory)
        directory.mkdir(parents=True, exist_ok=True)
        dataset = directory / "synthetic_features.csv"
        artifact = directory / "model.joblib"
        predictions = directory / "test_predictions.csv"
        backtest_csv = directory / "backtest.csv"
        backtest_report = directory / "backtest_report.json"
        generate_synthetic_dataset(dataset, rows=args.rows)
        trained = train_and_evaluate(dataset, args.model, artifact, predictions)
        backtested = backtest_files(
            predictions, backtest_csv, backtest_report,
            taker_fee_bps=args.taker_fee_bps, slippage_bps=args.slippage_bps,
        )
        _print_json({"note": "Synthetic data with a planted signal only checks software flow; it is not evidence of a real-market edge.", "training": trained, "backtest": backtested, "directory": str(directory)})


if __name__ == "__main__":
    main()
