import sqlite3
import os
import sys
from pprint import pprint

DB_PATH = os.path.join('data','db','nhl','2025-26.db')
if not os.path.exists(DB_PATH):
    print(f"ERROR: DB not found at {DB_PATH}")
    sys.exit(2)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("SELECT name,type,sql FROM sqlite_master WHERE type IN ('table','view') ORDER BY name")
items = cur.fetchall()

print('--- tables_and_views ---')
for name,typ,sql in items:
    print(name)

candidates = [name for name,_,_ in items if any(k in name.lower() for k in ('forecast','prediction','pred','schedule','backtest','tuning','rank'))]
if not candidates:
    candidates = [name for name,_,_ in items]

print('\n--- candidate_tables ---')
for t in candidates:
    print(t)

# columns of every candidate
print('\n--- schema_details ---')
all_info = {}
for t in candidates:
    try:
        cur.execute(f"PRAGMA table_info('{t}')")
        cols = cur.fetchall()
        colnames = [c[1] for c in cols]
        all_info[t] = colnames
        print(f"{t}: {colnames}")
    except Exception as e:
        print(f"PRAGMA failed for {t}: {e}")

# fields to look for
total_fields = ['pred_total','total_mean','total_sd','actual_total']
score_fields = ['home_score','away_score','home_pts','away_pts','hs','as']

print('\n--- totals_presence_and_counts ---')
results = []
for t,cols in all_info.items():
    has_model = any(c.lower()=='model' or 'model' in c.lower() for c in cols)
    has_season = any(c.lower()=='season' or 'season' in c.lower() for c in cols)
    has_sport = any(c.lower()=='sport' or 'league' in c.lower() for c in cols)
    found_totals = [c for c in cols if c in total_fields]
    found_scores = [c for c in cols if c in score_fields]
    entry = {'table':t,'cols':cols,'has_model':has_model,'has_season':has_season,'has_sport':has_sport,'found_totals':found_totals,'found_scores':found_scores}

    # Build a where clause candidate
    where_clauses = []
    if has_model:
        where_clauses.append("(model='elo' OR model LIKE '%elo%')")
    if has_season:
        where_clauses.append("(season='2025-26' OR season LIKE '%2025-26%')")
    elif 'season' not in cols:
        # try year columns
        where_clauses.append("1=1")

    where = ' AND '.join(where_clauses) if where_clauses else '1=1'

    # For each interesting field, try to count non-null typed numeric entries
    counts = {}
    for f in found_totals:
        try:
            q = f"SELECT COUNT(*) FROM '{t}' WHERE {where} AND {f} IS NOT NULL AND typeof({f}) IN ('real','integer')"
            cur.execute(q)
            cnum = cur.fetchone()[0]
            counts[f] = int(cnum)
        except Exception as e:
            counts[f] = f'ERR: {e}'

    # if no pred_total/total_mean, try home+away
    if not found_totals and found_scores:
        h = found_scores[0]
        a = found_scores[1] if len(found_scores)>1 else None
        if a:
            try:
                q = f"SELECT COUNT(*) FROM '{t}' WHERE {where} AND {h} IS NOT NULL AND {a} IS NOT NULL AND typeof({h}) IN ('real','integer') AND typeof({a}) IN ('real','integer')"
                cur.execute(q)
                counts['home_away_numeric'] = int(cur.fetchone()[0])
            except Exception as e:
                counts['home_away_numeric'] = f'ERR: {e}'

    entry['counts'] = counts
    results.append(entry)

pprint(results)

# Summarize definitive answer: check any table where pred_total or total_mean numeric count >0
suitable = False
missing = []
for r in results:
    for f,cnt in r['counts'].items():
        if isinstance(cnt,int) and cnt>0 and f in ('pred_total','total_mean','home_away_numeric'):
            suitable = True

# print final
print('\n--- SUMMARY ---')
print('Elo emits numeric totals suitable for mae_total tuning:' , 'YES' if suitable else 'NO')
if not suitable:
    # list which fields were absent or zero
    for r in results:
        if not r['found_totals'] and not r['found_scores']:
            missing.append((r['table'],'no total/score fields'))
        else:
            for f in total_fields:
                if f in r['cols']:
                    cnt = r['counts'].get(f,0)
                    if not (isinstance(cnt,int) and cnt>0):
                        missing.append((r['table'],f,'zero_or_non-numeric'))
            if 'home_away_numeric' in r['counts']:
                if not (isinstance(r['counts']['home_away_numeric'],int) and r['counts']['home_away_numeric']>0):
                    missing.append((r['table'],'home+away','zero_or_non-numeric'))

    pprint(missing)

conn.close()
