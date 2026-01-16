import json
from src.ingest.paste_ingest import ingest_file

rows = ingest_file('data/raw/NBA-market.txt')
print('Parsed rows:', len(rows))
print(json.dumps(rows[:20], indent=2, ensure_ascii=False))
