import sys
sys.path.insert(0, 'src')
import sqlite3
import pandas as pd
from pathlib import Path
from pipelines.ensemble_weight_validation import validate_ensemble_ml_weights

db_path = Path('data/db/nba/2025-26.db')

# For demo: manually mark one prediction game as completed
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get one prediction and manually add a completed game
cursor.execute('''
    SELECT bp.game_id FROM bets_predictions bp 
    WHERE bp.sport = ? AND bp.season = ? AND bp.prediction_date = ?
    LIMIT 1
''', ('nba', '2025-26', '2026-01-24'))

result = cursor.fetchone()
if result:
    game_id = result[0]
    print(f'Demo: Marking prediction game {game_id} as completed...')
    
    # Update a game to have scores (for demo)
    cursor.execute('''
        UPDATE games SET home_score = 105, away_score = 100
        WHERE game_id = ?
    ''', (game_id,))
    conn.commit()
    print('  Done.')

conn.close()

# Now run validation
print('\nRunning validation...\n')
result = validate_ensemble_ml_weights(
    db_path=db_path,
    sport='nba',
    season='2025-26',
    market='ML',
    days_back=7,
)

if result:
    print('\nValidation Result:')
    print(f"  Total games: {result['n_games']}")
    print(f"  Prediction dates covered: {result['n_prediction_dates']}")
    print(f"  Date range: {result['date_range_start']} to {result['date_range_end']}")
