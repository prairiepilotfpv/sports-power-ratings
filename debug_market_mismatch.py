#!/usr/bin/env python3
import sqlite3
from pathlib import Path

db_path = Path('data/db/nba/2025-26.db')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check games schema
print("=" * 80)
print("GAMES TABLE SCHEMA")
print("=" * 80)
cursor.execute("PRAGMA table_info(games)")
columns = cursor.fetchall()
for col in columns:
    print(f"  {col[1]}: {col[2]}")

# Check games on 2026-01-25
print("\n" + "=" * 80)
print("GAMES IN DB FOR 2026-01-25")
print("=" * 80)
cursor.execute("""
SELECT game_id, home_team, away_team, date FROM games 
WHERE date >= '2026-01-25' AND date < '2026-01-26'
ORDER BY game_id
""")
games = cursor.fetchall()
print(f"Total: {len(games)} games")
for g in games:
    print(f"  {g[0]}: {g[2]} @ {g[1]} ({g[3]})")

# Check market line import errors
print("\n" + "=" * 80)
print("MARKET LINE IMPORT ERRORS")
print("=" * 80)
cursor.execute("""
SELECT COUNT(*), failure_reason FROM market_line_import_errors 
WHERE sport = 'nba' AND season = '2025-26'
GROUP BY failure_reason
""")
errors = cursor.fetchall()
for count, reason in errors:
    print(f"  {count}x {reason}")

# Get details on game_unmatched errors
print("\n" + "=" * 80)
print("GAME_UNMATCHED DETAILS")
print("=" * 80)
cursor.execute("""
SELECT row_data FROM market_line_import_errors 
WHERE sport = 'nba' AND season = '2025-26' AND failure_reason = 'game_unmatched'
LIMIT 10
""")
errors = cursor.fetchall()
import json
for row_data, in errors:
    row = json.loads(row_data)
    print(f"  Date: {row.get('game_date')}, Home: {row.get('team_home_raw')}, Away: {row.get('team_away_raw')}")

conn.close()
