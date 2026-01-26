#!/usr/bin/env python3
"""
Independent Calibration CLI

Usage:
  python calibration_cli.py fit --sport nba --season 2025-26 --market ML --ensemble ensemble_ml_v1
  python calibration_cli.py fit --sport nba --season 2025-26 --market SPREAD --ensemble ensemble_spread_v1
  python calibration_cli.py fit --sport nba --season 2025-26 --market TOTAL --ensemble ensemble_total_v1
  python calibration_cli.py apply --input bets_sheet.xlsx --calibrators-dir data/calibrators --output bets_calibrated.xlsx
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from src.calibration_system import EnsembleCalibrator, save_calibration_result, load_calibration


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
_LOG = logging.getLogger(__name__)


def fit_calibrators(args):
    """Fit calibrators for ensemble predictions."""
    db_path = args.db or f"data/db/{args.sport}/{args.season}.db"
    calibrator_system = EnsembleCalibrator(db_path)
    
    markets = [args.market] if args.market else ['ML', 'SPREAD', 'TOTAL']
    ensemble_ids = {
        'ML': args.ensemble_ml or 'ensemble_ml_v1',
        'SPREAD': args.ensemble_spread or 'ensemble_spread_v1',
        'TOTAL': args.ensemble_total or 'ensemble_total_v1',
    }
    
    results = []
    for market in markets:
        try:
            ensemble_id = ensemble_ids.get(market)
            if not ensemble_id:
                _LOG.warning(f"No ensemble ID for {market}, skipping")
                continue
            
            result, calibrator = calibrator_system.fit_calibrator(
                sport=args.sport,
                season=args.season,
                market=market,
                ensemble_id=ensemble_id,
                calibrator_type=args.calibrator_type,
                start_date=date.fromisoformat(args.start_date) if args.start_date else None,
                end_date=date.fromisoformat(args.end_date) if args.end_date else None,
            )
            
            save_calibration_result(result, args.output_dir)
            results.append(result)
            
            print(f"\n✓ {market}: log_loss {result.metric_before:.4f} → {result.metric_after:.4f}")
        
        except Exception as e:
            _LOG.error(f"Failed to calibrate {market}: {e}")
            if args.strict:
                raise
    
    print(f"\nCalibrated {len(results)} markets")


def apply_calibrators(args):
    """Apply calibrators to BETS sheet predictions."""
    # Load BETS sheet
    bets_df = pd.read_excel(args.input, sheet_name='BETS')
    print(f"Loaded {len(bets_df)} BETS predictions")
    
    calibrators_dir = Path(args.calibrators_dir)
    calibrator_system = EnsembleCalibrator(args.db or "data/db/nba/2025-26.db")
    
    # Create calibrated_model_prob column
    bets_df['calibrated_model_prob'] = bets_df['model_prob'].copy()
    
    # Apply calibrators by market type
    for market_type in ['ML', 'spread', 'total']:
        market_rows = bets_df['market_type'] == market_type
        if not market_rows.any():
            continue
        
        # Find calibrator file
        ensemble_id = {
            'ML': 'ensemble_ml_v1',
            'spread': 'ensemble_spread_v1',
            'total': 'ensemble_total_v1',
        }[market_type]
        
        calibrator_file = calibrators_dir / f"{market_type}_{ensemble_id}_*.json"
        matches = list(calibrators_dir.glob(f"{market_type}_{ensemble_id}_*.json"))
        
        if not matches:
            _LOG.warning(f"No calibrator found for {market_type}")
            continue
        
        calibrator_file = matches[0]
        _LOG.info(f"Loading calibrator from {calibrator_file}")
        
        try:
            calibrator_type, calibrator = load_calibration(calibrator_file)
            
            # Apply to market rows
            probs = bets_df.loc[market_rows, 'model_prob'].values
            calibrated_probs = calibrator_system.apply_calibrator(calibrator, probs)
            bets_df.loc[market_rows, 'calibrated_model_prob'] = calibrated_probs
            
            print(f"✓ Applied {calibrator_type} calibrator to {market_rows.sum()} {market_type} predictions")
        
        except Exception as e:
            _LOG.error(f"Failed to apply calibrator for {market_type}: {e}")
    
    # Save output
    output_path = args.output
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        bets_df.to_excel(writer, sheet_name='BETS', index=False)
    
    print(f"✓ Saved calibrated predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Independent ensemble calibration system"
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Fit calibrators
    fit_parser = subparsers.add_parser('fit', help='Fit calibrators on historical data')
    fit_parser.add_argument('--sport', required=True, help='Sport (e.g., nba)')
    fit_parser.add_argument('--season', required=True, help='Season (e.g., 2025-26)')
    fit_parser.add_argument('--market', help='Market (ML, SPREAD, TOTAL) - all if not specified')
    fit_parser.add_argument('--ensemble-ml', help='ML ensemble ID')
    fit_parser.add_argument('--ensemble-spread', help='SPREAD ensemble ID')
    fit_parser.add_argument('--ensemble-total', help='TOTAL ensemble ID')
    fit_parser.add_argument('--calibrator-type', default='temperature', choices=['temperature', 'isotonic'])
    fit_parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    fit_parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    fit_parser.add_argument('--db', help='Database path')
    fit_parser.add_argument('--output-dir', default='data/calibrators', help='Output directory for calibrator files')
    fit_parser.add_argument('--strict', action='store_true', help='Fail if any market fails')
    fit_parser.set_defaults(func=fit_calibrators)
    
    # Apply calibrators
    apply_parser = subparsers.add_parser('apply', help='Apply calibrators to BETS sheet')
    apply_parser.add_argument('--input', required=True, help='Input BETS Excel file')
    apply_parser.add_argument('--output', required=True, help='Output Excel file')
    apply_parser.add_argument('--calibrators-dir', default='data/calibrators', help='Directory with calibrator files')
    apply_parser.add_argument('--db', help='Database path (for reference)')
    apply_parser.set_defaults(func=apply_calibrators)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
