import sys
sys.path.insert(0, 'src')
from src.data.betting_repository import import_market_csv
import sqlite3
DB='data/db/nhl/2025-26.db'
res = import_market_csv(
    db_path=DB,
    csv_path='data/raw/nhlToday.csv',
    sport='nhl',
    season='2025-26',
    default_book='DK',
)
print('IMPORT_RESULT:', res)
conn=sqlite3.connect(DB)
cur=conn.cursor()
cur.execute("SELECT COUNT(1) FROM market_lines WHERE sport = ? AND season = ?", ('nhl', '2025-26'))
a=cur.fetchone()[0]
print('market_lines_rows:', a)
cur.execute("SELECT COUNT(1) FROM market_line_import_errors")
b=cur.fetchone()[0]
print('import_errors:', b)
cur.execute("SELECT failure_reason FROM market_line_import_errors ORDER BY id DESC LIMIT 5")
rows=cur.fetchall()
print('recent_import_errors (up to 5):')
for r in rows:
    print(r)
conn.close()
