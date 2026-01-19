import sqlite3, json
from pathlib import Path
p = Path('data/db/nhl/2025-26.db')
if not p.exists():
    print('DB not found:', p)
    raise SystemExit(1)
con = sqlite3.connect(p)
cur = con.cursor()
try:
    rows = cur.execute('SELECT model,market,params_json,source_run_id,updated_at FROM model_market_active_params').fetchall()
    print(json.dumps(rows, indent=2))
finally:
    con.close()
