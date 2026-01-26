"""CLI interface for the standalone calibration system."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tmp_calibration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

_LOG = logging.getLogger(__name__)


def main():
    """CLI entry point for calibration subsystem."""
    parser = argparse.ArgumentParser(
        description="Ensemble Calibration System - Independent from betting pipeline"
    )
    
    parser.add_argument("--db", required=True, help="Path to database")
    parser.add_argument("--sport", required=True, help="Sport identifier (e.g., nba)")
    parser.add_argument("--season", required=True, help="Season identifier (e.g., 2025-26)")
    parser.add_argument(
        "--models",
        required=True,
        nargs="+",
        help="List of models to ensemble (e.g., bradley-terry elo)"
    )
    parser.add_argument(
        "--ensemble-id",
        default="ensemble_v1",
        help="Ensemble identifier for saved calibrators"
    )
    parser.add_argument(
        "--start-date",
        help="Start date for historical games (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        help="End date for historical games (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--method",
        default="auto",
        choices=["auto", "isotonic", "platt"],
        help="Calibration method"
    )
    
    args = parser.parse_args()
    
    _LOG.info("=" * 80)
    _LOG.info("ENSEMBLE CALIBRATION SYSTEM")
    _LOG.info("=" * 80)
    _LOG.info(f"Database: {args.db}")
    _LOG.info(f"Sport: {args.sport}, Season: {args.season}")
    _LOG.info(f"Models: {args.models}")
    _LOG.info(f"Ensemble ID: {args.ensemble_id}")
    _LOG.info(f"Method: {args.method}")
    
    try:
        from src.calibration.calibration_engine import calibrate_ensemble
        
        calibrators = calibrate_ensemble(
            args.db,
            sport=args.sport,
            season=args.season,
            models=args.models,
            ensemble_id=args.ensemble_id,
            start_date=args.start_date,
            end_date=args.end_date,
            method=args.method,
        )
        
        _LOG.info("=" * 80)
        _LOG.info("CALIBRATION COMPLETE")
        _LOG.info("=" * 80)
        for market, path in calibrators.items():
            _LOG.info(f"{market}: {path}")
        
    except Exception as e:
        _LOG.exception(f"Calibration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
