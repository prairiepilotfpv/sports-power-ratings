#!/usr/bin/env python3
"""Generate and save ensemble predictions for historical games for calibration."""

import sys
from pathlib import Path

# Ensure src is on path
repo_root = Path(__file__).resolve().parent
src_dir = repo_root / "src"
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(repo_root))

import pandas as pd
import sqlite3
from datetime import datetime, timezone
from data.repository import init_db
from data.bets_repository import save_bets_predictions

def generate_historical_predictions_for_calibration(
    sport: str = "nba",
    season: str = "2025-26",
):
    """Generate ensemble predictions for all games and save for calibration."""
    from data.paths import db_path_for
    from pipelines.schedule import build_schedule_excel_report
    import tempfile
    
    db_path = db_path_for(sport, season)
    init_db(db_path)
    
    print(f"[{sport}/{season}] Generating ensemble predictions for calibration...")
    
    # Build schedule report which generates predictions
    # Use a temp directory to avoid cluttering outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        result_path = build_schedule_excel_report(
            db_path=str(db_path),
            sport=sport,
            season=season,
            division=None,
            conference=None,
            model=None,
            output_path=Path(tmpdir),
            upcoming_only=False,  # Include ALL games, not just upcoming
            model_params=None,
            model_params_file=None,
            tuned_metric=None,
            as_of_date=None,
            bets_model=None,
            strict=False,
        )
        
        print(f"[{sport}/{season}] Generated schedule report: {result_path}")
        
        # Load the BETS sheet from the generated Excel
        bets_df = pd.read_excel(result_path, sheet_name="BETS")
        print(f"[{sport}/{season}] Loaded {len(bets_df)} BETS rows from Excel")
    
    # Clear existing predictions
    conn = sqlite3.connect(str(db_path))
    print(f"[{sport}/{season}] Clearing existing predictions...")
    conn.execute('DELETE FROM bets_predictions')
    conn.commit()
    conn.close()
    
    # Save the predictions with ensemble sources
    print(f"[{sport}/{season}] Saving predictions to database...")
    n_saved = save_bets_predictions(
        str(db_path),
        bets_df=bets_df,
        sport=sport,
        season=season,
    )
    
    print(f"[{sport}/{season}] Saved {n_saved} predictions")
    
    # Show what we have
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("""
        SELECT market_type, market_forecast_source, COUNT(*) as count
        FROM bets_predictions
        GROUP BY market_type, market_forecast_source
        ORDER BY market_type, market_forecast_source
    """)
    
    print(f"\n[{sport}/{season}] Database state:")
    for market_type, source, count in cursor:
        print(f"  {market_type:10} | {source:30} | {count}")
    
    # Now check which predictions have completed games
    cursor = conn.execute("""
        SELECT 
            bp.market_type,
            COUNT(DISTINCT bp.game_id) as predicted_games,
            COUNT(DISTINCT CASE WHEN g.home_score IS NOT NULL THEN bp.game_id END) as completed_games
        FROM bets_predictions bp
        LEFT JOIN games g ON bp.game_id = g.game_id
        GROUP BY bp.market_type
    """)
    
    print(f"\n[{sport}/{season}] Prediction vs Completed Games:")
    for market_type, predicted, completed in cursor:
        print(f"  {market_type:10}: {predicted:4} predicted, {completed:4} completed")
    
    conn.close()
    
    print(f"\n[{sport}/{season}] Done!")

if __name__ == "__main__":
    generate_historical_predictions_for_calibration()
