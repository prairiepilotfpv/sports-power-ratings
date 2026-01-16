import sqlite3
from pprint import pprint
conn=sqlite3.connect('data/db/nba/2025-26.db')
cur=conn.cursor()
# load staging rows
cols=[c[0] for c in cur.execute("PRAGMA table_info(market_snapshot_staging)")]
st_rows=[dict(zip(cols, r)) for r in cur.execute('SELECT * FROM market_snapshot_staging').fetchall()]
# target game
game_id='2026-01-14|Brooklyn Nets|New Orleans Pelicans'
home='New Orleans Pelicans'
away='Brooklyn Nets'
for mtype, sel in [('ML', away), ('ML', home), ('spread', away), ('spread', home)]:
    print('---',mtype,sel)
    sel_norm=str(sel).lower(); home_norm=str(home).lower(); away_norm=str(away).lower()
    for s in st_rows:
        try:
            if s.get('market_type')!=mtype: continue
            gdate=str(s.get('game_date') or '')[:10]
            if gdate!='2026-01-14': continue
            s_sel=str(s.get('selection') or '').lower()
            th=str(s.get('team_home_raw') or '').lower(); ta=str(s.get('team_away_raw') or '').lower()
            # selection equality
            if mtype!='total' and s_sel==sel_norm:
                print('sel eq ->',s['id'],s_sel,th,ta); break
            if mtype in ('ML','spread'):
                # home/away side
                if sel_norm==home_norm:
                    if (s_sel==home_norm or s_sel in home_norm or home_norm in th or home_norm in ta or s_sel in th or s_sel in ta):
                        print('side match ->',s['id'],s_sel,th,ta); break
                if sel_norm==away_norm:
                    if (s_sel==away_norm or s_sel in away_norm or away_norm in th or away_norm in ta or s_sel in th or s_sel in ta):
                        print('side match ->',s['id'],s_sel,th,ta); break
                if s_sel==sel_norm:
                    print('fallback sel eq ->',s['id'],s_sel,th,ta); break
            if mtype=='total':
                if ((home_norm in th or home_norm in ta) and (away_norm in th or away_norm in ta)):
                    print('total pair ->',s['id'],s_sel,th,ta); break
            # fallback game_id for totals or unspecified selections
            if s.get('game_id')==game_id:
                if mtype=='total' or s_sel in ('','over','under'):
                    print('fallback gid ->',s['id'],s_sel,th,ta); break
        except Exception as e:
            pass
conn.close()
print('done')
