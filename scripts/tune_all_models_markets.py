#!/usr/bin/env python
"""Orchestrate hyperparameter tuning for all models across all markets.

Usage:
    python scripts/tune_all_models_markets.py \\
        --sport nba \\
        --season 2025-26 \\
        --csv data/raw/nba.csv \\
        --start 2024-11-01 \\
        --end 2024-12-31

This runs tune-model for each model (bradley-terry, elo, gssd, poisson, toor)
automatically, tuning across all three markets (ML, SPREAD, TOTAL) per model,
and activates tuned params immediately.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Default models to tune (in order)
DEFAULT_MODELS = ["bradley-terry", "elo", "gssd", "poisson", "toor"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune all models across all markets for a sport/season.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic: tune all models for a date range
  python scripts/tune_all_models_markets.py \\
    --sport nba --season 2025-26 \\
    --csv data/raw/nba.csv \\
    --start 2024-11-01 --end 2024-12-31

  # With custom grid
  python scripts/tune_all_models_markets.py \\
    --sport nba --season 2025-26 \\
    --csv data/raw/nba.csv \\
    --start 2024-11-01 --end 2024-12-31 \\
    --grid-file custom_grid.json

  # Tune specific models only
  python scripts/tune_all_models_markets.py \\
    --sport nba --season 2025-26 \\
    --csv data/raw/nba.csv \\
    --start 2024-11-01 --end 2024-12-31 \\
    --models elo,bradley-terry

  # With parallelization
  python scripts/tune_all_models_markets.py \\
    --sport nba --season 2025-26 \\
    --csv data/raw/nba.csv \\
    --start 2024-11-01 --end 2024-12-31 \\
    --jobs 4
        """,
    )

    parser.add_argument(
        "--sport",
        required=True,
        help="Sport identifier (e.g., nba, nhl, nfl).",
    )
    parser.add_argument(
        "--season",
        required=True,
        help="Season identifier (e.g., 2025-26).",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="CSV path containing historical games for tuning.",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Backtest start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="Backtest end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of models to tune (default: all). "
        "Options: bradley-terry, elo, gssd, poisson, toor.",
    )
    parser.add_argument(
        "--market",
        default=None,
        help="Market(s) to tune (ML, SPREAD, TOTAL). Defaults to all when omitted.",
    )
    parser.add_argument(
        "--window",
        default="expanding",
        choices=["expanding", "rolling"],
        help="Training window type (default: expanding).",
    )
    parser.add_argument(
        "--rolling-days",
        type=int,
        default=None,
        help="Rolling window size in days (required if --window=rolling).",
    )
    parser.add_argument(
        "--rolling-games",
        type=int,
        default=None,
        help="Rolling window size in games (optional alternative for rolling).",
    )
    parser.add_argument(
        "--grid-file",
        default=None,
        help="Optional JSON file defining parameter grids.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Optional SQLite DB path override.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel jobs per model (default: 1).",
    )
    parser.add_argument(
        "--no-activate",
        action="store_true",
        help="Skip activating tuned params (don't persist to DB).",
    )
    parser.add_argument(
        "--allow-worse",
        action="store_true",
        help="Allow worse results than default parameters.",
    )
    parser.add_argument(
        "--metric-overrides",
        default=None,
        help="Optional JSON map of market->metric overrides (e.g., '{\"ML\":\"log_loss\"}').",
    )

    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Validate provided arguments."""
    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("CSV file not found: %s", csv_path)
        sys.exit(1)

    # Validate dates
    try:
        datetime.strptime(args.start, "%Y-%m-%d")
        datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError as e:
        logger.error("Invalid date format (expected YYYY-MM-DD): %s", e)
        sys.exit(1)

    if args.window == "rolling" and args.rolling_days is None and args.rolling_games is None:
        logger.error("--rolling-days or --rolling-games required when --window=rolling")
        sys.exit(1)

    if args.grid_file and not Path(args.grid_file).exists():
        logger.error("Grid file not found: %s", args.grid_file)
        sys.exit(1)


def _resolve_models(models_arg: Optional[str]) -> list[str]:
    """Parse and validate models from argument."""
    if models_arg is None:
        return DEFAULT_MODELS
    models = [m.strip().lower() for m in models_arg.split(",")]
    valid = set(DEFAULT_MODELS)
    invalid = [m for m in models if m not in valid]
    if invalid:
        logger.warning("Skipping invalid models: %s", invalid)
        models = [m for m in models if m in valid]
    if not models:
        logger.error("No valid models specified. Valid options: %s", ", ".join(DEFAULT_MODELS))
        sys.exit(1)
    return models


def _build_tune_model_cmd(
    model: str,
    args: argparse.Namespace,
    activate: bool,
) -> list[str]:
    """Build the tune-model CLI command for a specific model."""
    cmd = [
        "python",
        "-m",
        "src.cli.pipeline",
        "tune-model",
        "--sport",
        args.sport,
        "--season",
        args.season,
        "--model",
        model,
        "--csv",
        args.csv,
        "--start",
        args.start,
        "--end",
        args.end,
        "--window",
        args.window,
    ]

    if args.rolling_days:
        cmd.extend(["--rolling-days", str(args.rolling_days)])
    if args.rolling_games:
        cmd.extend(["--rolling-games", str(args.rolling_games)])
    if args.market:
        cmd.extend(["--market", args.market])
    if args.grid_file:
        cmd.extend(["--grid-file", args.grid_file])
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    if args.db:
        cmd.extend(["--db", args.db])
    if args.jobs and args.jobs > 1:
        cmd.extend(["--jobs", str(args.jobs)])
    if args.allow_worse:
        cmd.append("--allow-worse")
    if args.metric_overrides:
        cmd.extend(["--market-metrics", args.metric_overrides])
    if activate:
        cmd.append("--activate")

    return cmd


def _run_tune_model(
    model: str,
    args: argparse.Namespace,
    model_index: int,
    total_models: int,
    activate: bool,
) -> tuple[str, bool, str]:
    """Run tune-model for a single model. Returns (model, success, summary)."""
    logger.info(
        "=== [%d/%d] Tuning model: %s ===",
        model_index + 1,
        total_models,
        model,
    )

    cmd = _build_tune_model_cmd(model, args, activate)
    logger.debug("Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True,
        )
        logger.info("✓ Model %s tuning completed successfully.", model)
        return (model, True, "success")
    except subprocess.CalledProcessError as e:
        logger.error("✗ Model %s tuning failed with exit code %d", model, e.returncode)
        return (model, False, f"exit_code={e.returncode}")
    except Exception as e:
        logger.error("✗ Model %s tuning error: %s", model, e)
        return (model, False, str(e))


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    models = _resolve_models(args.models)
    activate = not args.no_activate

    logger.info("=" * 70)
    logger.info("TUNE ALL MODELS ACROSS ALL MARKETS")
    logger.info("=" * 70)
    logger.info("Sport: %s", args.sport)
    logger.info("Season: %s", args.season)
    logger.info("Date range: %s to %s", args.start, args.end)
    logger.info("Window: %s", args.window)
    if args.rolling_days:
        logger.info("Rolling days: %d", args.rolling_days)
    if args.rolling_games:
        logger.info("Rolling games: %d", args.rolling_games)
    logger.info("Models to tune: %s", ", ".join(models))
    logger.info("Activate params: %s", "yes" if activate else "no")
    logger.info("=" * 70)

    results: list[tuple[str, bool, str]] = []
    for idx, model in enumerate(models):
        success, summary = _run_tune_model(
            model,
            args,
            idx,
            len(models),
            activate,
        )[:2]
        result_summary = _run_tune_model(model, args, idx, len(models), activate)[2]
        results.append((model, success, result_summary))
        logger.info("")

    # Summary
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    succeeded = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    for model, success, summary in results:
        status = "✓" if success else "✗"
        logger.info("%s %s: %s", status, model, summary)

    logger.info("")
    logger.info("Results: %d succeeded, %d failed (out of %d total)", len(succeeded), len(failed), len(results))

    if failed:
        logger.warning("Failed models: %s", ", ".join([m for m, _, _ in failed]))
        sys.exit(1)
    else:
        logger.info("All models tuned successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
