import sqlite3
import os
DB_PATH = os.path.join('data','db','nhl','2025-26.db')
if not os.path.exists(DB_PATH):
    print('DB not found', DB_PATH)
    raise SystemExit(1)
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

def pragma_cols(t):
    cur.execute(f"PRAGMA table_info('{t}')")
    return [r[1] for r in cur.fetchall()]

print('--- games schema ---')
print(pragma_cols('games'))
print('\n--- forecast_snapshots schema ---')
print(pragma_cols('forecast_snapshots'))

# Count forecast_snapshots rows for model='elo'
try:
    cur.execute("SELECT COUNT(*) FROM forecast_snapshots WHERE model='elo'")
    total_elo = cur.fetchone()[0]
except Exception as e:
    total_elo = f'ERR: {e}'

# Count projected_total non-null and numeric-typed
try:
    cur.execute("SELECT COUNT(*) FROM forecast_snapshots WHERE model='elo' AND projected_total IS NOT NULL AND typeof(projected_total) IN ('real','integer')")
    proj_total_numeric = cur.fetchone()[0]
except Exception as e:
    proj_total_numeric = f'ERR: {e}'

# Count projected_home_score & projected_away_score numeric
try:
    cur.execute("SELECT COUNT(*) FROM forecast_snapshots WHERE model='elo' AND projected_home_score IS NOT NULL AND projected_away_score IS NOT NULL AND typeof(projected_home_score) IN ('real','integer') AND typeof(projected_away_score) IN ('real','integer')")
    proj_home_away_numeric = cur.fetchone()[0]
except Exception as e:
    proj_home_away_numeric = f'ERR: {e}'

# Join to games to see actual numeric home+away for those game_ids
try:
    cur.execute("SELECT COUNT(*) FROM forecast_snapshots f JOIN games g ON f.game_id = g.id WHERE f.model='elo' AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL AND typeof(g.home_score) IN ('real','integer') AND typeof(g.away_score) IN ('real','integer')")
    joined_actuals = cur.fetchone()[0]
except Exception as e:
    joined_actuals = f'ERR: {e}'

print('\n--- counts ---')
print('forecast_snapshots rows with model=elo:', total_elo)
print('forecast_snapshots projected_total numeric count:', proj_total_numeric)
print('forecast_snapshots projected_home+away numeric count:', proj_home_away_numeric)
print('forecast_snapshots join games with numeric actuals count:', joined_actuals)

conn.close()
