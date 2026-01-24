import sys
sys.path.insert(0, 'src')

import pandas as pd
from data.repository import load_games
from pathlib import Path

# Load from BETS sheet
bets_df = pd.read_excel('data/processed/nba/2025-26/schedule_with_projections.xlsx', sheet_name='BETS')
print('BETS sheet columns (first 15):', bets_df.columns.tolist()[:15])
print(f'BETS sheet rows: {len(bets_df)}')

if 'game_id' in bets_df.columns:
    print('\nBETS sheet game_ids (first 5):')
    print(bets_df[['game_id']].head())
else:
    print('\nNo game_id in BETS; showing first row columns:')
    print(bets_df.columns.tolist())

# Load from database
db_path = Path('data/db/nba/2025-26.db')
games = load_games(db_path, sport='nba', season='2025-26')
completed = [g for g in games if g.home_score is not None and g.away_score is not None]
print(f'\nDatabase: {len(completed)} completed games')

if completed:
    print('Database game_ids (first 5):')
    for g in completed[:5]:
        print(f'  {g.game_id}: {g.date} {g.home_team} vs {g.away_team}')
