"""Standalone CLI for historical calibration system.

This is a completely independent command-line interface for fitting calibrators
on historical data. It is NOT integrated with the betting pipeline.

Usage:
    python -m calibration.standalone_cli fit \
        --db data/db/nba/2025-26.db \
        --sport nba \
        --season 2025-26 \
        --models bradley-terry elo \
        --markets ML SPREAD TOTAL \
        --method auto \
        --source-id historical

This generates calibrators saved to outputs/calibrators/<sport>/<season>/<source>/<market>/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    """Add src directory to Python path so imports work."""
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    src_str = str(src_dir)
    root_str = str(repo_root)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    if root_str not in sys.path:
        insert_at = sys.path.index(src_str) + 1 if src_str in sys.path else 0
        sys.path.insert(insert_at, root_str)


# Ensure src is on path before any other imports
_ensure_src_on_path()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("tmp_calibration.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

_LOG = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Standalone calibration system for historical model predictions. "
            "Completely independent from betting pipeline."
        ),
    )

    parser.add_argument(
        "--db",
        required=True,
        help="Path to SQLite database (e.g., data/db/nba/2025-26.db)",
    )
    parser.add_argument(
        "--sport",
        required=True,
        help="Sport identifier (e.g., nba, nfl, mlb, nhl). Case-insensitive.",
    )
    parser.add_argument(
        "--season",
        required=True,
        help="Season identifier (e.g., 2025-26, 2024). Used for organizing artifacts.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help=(
            "Space-separated list of model names (kept for compatibility). "
            "Ensemble membership now follows the ensemble config for each market."
        ),
    )
    parser.add_argument(
        "--markets",
        nargs="*",
        default=["ML", "SPREAD", "TOTAL"],
        help="Markets to calibrate (default: ML SPREAD TOTAL). Options: ML, SPREAD, TOTAL",
    )
    parser.add_argument(
        "--source-id",
        default="historical",
        help="Identifier for saved calibrators (default: historical). "
        "Allows multiple calibration runs to coexist.",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "isotonic", "platt"],
        default="auto",
        help=(
            "Calibration method (default: auto). "
            "For ML market: auto chooses isotonic if n>=500 else platt. "
            "For SPREAD/TOTAL: always uses distribution calibrator."
        ),
    )
    parser.add_argument(
        "--start-date",
        help="Optional start date filter (YYYY-MM-DD). Only include games on/after this date.",
    )
    parser.add_argument(
        "--end-date",
        help="Optional end date filter (YYYY-MM-DD). Only include games on/before this date.",
    )

    return parser.parse_args(argv)


def _main(argv: list[str] | None = None) -> int:
    """Main entry point for standalone calibration CLI."""
    args = _parse_args(argv)

    _LOG.info("=" * 80)
    _LOG.info("STANDALONE CALIBRATION SYSTEM")
    _LOG.info("=" * 80)
    _LOG.info(f"Database: {args.db}")
    _LOG.info(f"Sport: {args.sport}, Season: {args.season}")
    _LOG.info(f"Models: {args.models}")
    _LOG.info(f"Markets: {args.markets}")
    _LOG.info(f"Source ID: {args.source_id}")
    _LOG.info(f"Method: {args.method}")
    if args.start_date:
        _LOG.info(f"Start Date: {args.start_date}")
    if args.end_date:
        _LOG.info(f"End Date: {args.end_date}")

    try:
        # Convert market strings to Market enums
        from markets.base import Market

        markets = []
        for market_str in args.markets:
            market_upper = market_str.strip().upper()
            try:
                markets.append(Market[market_upper])
            except KeyError:
                _LOG.error(f"Unknown market: {market_str}. Options: ML, SPREAD, TOTAL")
                return 1

        # Import and run calibration
        from calibration.historical_calibration import calibrate_sport_season

        results = calibrate_sport_season(
            args.db,
            sport=args.sport,
            season=args.season,
            models=args.models,
            markets=markets,
            source_id=args.source_id,
            method=args.method,
            start_date=args.start_date,
            end_date=args.end_date,
        )

        _LOG.info("=" * 80)
        _LOG.info("CALIBRATION COMPLETE")
        _LOG.info("=" * 80)

        for market_name, (calibrator, saved_path) in results.items():
            status = f"OK {saved_path}" if saved_path else "OK Fitted (not saved)"
            _LOG.info(f"{market_name}: {status}")
            _LOG.info(f"  Metadata: {calibrator.metadata}")

        _LOG.info("=" * 80)
        _LOG.info(f"Fitted {len(results)} market(s) successfully")
        return 0

    except Exception as e:
        _LOG.exception(f"Calibration failed: {e}")
        _LOG.info("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(_main())
