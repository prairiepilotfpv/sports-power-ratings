import sys
sys.path.insert(0, 'src')
from src.data.betting_repository import import_market_csv
import sqlite3
DB='data/db/nhl/2025-26.db'
res = import_market_csv(
    db_path=DB,
    csv_path='data/raw/nhlToday.csv',
    snapshot_run_id='snap-20260114',
    sport='nhl',
    season='2025-26',
    default_book='DK',
    commit_matched=True,
)
print('IMPORT_RESULT:', res)
conn=sqlite3.connect(DB)
cur=conn.cursor()
cur.execute("SELECT COUNT(1) FROM market_snapshots WHERE snapshot_run_id = ?", ('snap-20260114',))
a=cur.fetchone()[0]
print('market_snapshots_committed_for_snap-20260114:', a)
cur.execute("SELECT COUNT(1) FROM market_snapshot_staging")
b=cur.fetchone()[0]
print('total_staging_rows:', b)
cur.execute("SELECT id, match_status, game_id, market_type, selection, line, odds, hold_reason FROM market_snapshot_staging ORDER BY id DESC LIMIT 5")
rows=cur.fetchall()
print('recent_staging_rows (up to 5):')
for r in rows:
    print(r)
conn.close()
