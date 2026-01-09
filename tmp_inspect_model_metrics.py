import sqlite3
import pprint
from pathlib import Path

db = Path('data/db/nba/2025-26.db')
if not db.exists():
    print('DB not found at', db)
    raise SystemExit(1)

conn = sqlite3.connect(db)
cur = conn.execute("SELECT * FROM model_metrics WHERE sport=? AND season=?", ('nba', '2025-26'))
rows = cur.fetchall()
cols = [d[0] for d in cur.description] if cur.description else []
results = [dict(zip(cols, row)) for row in rows]
pp = pprint.PrettyPrinter(indent=2)
pp.pprint(results)
conn.close()
