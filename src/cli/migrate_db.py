"""DB migration CLI: analyze and optionally apply migrations across DB files.

Usage:
  python -m src.cli.migrate_db --db-dir data/db --apply --yes
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sqlite3
import sys
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
_SRC_STR = str(_SRC_DIR)
_ROOT_STR = str(_REPO_ROOT)
if _SRC_STR not in sys.path:
    sys.path.insert(0, _SRC_STR)
if _ROOT_STR not in sys.path:
    insert_at = sys.path.index(_SRC_STR) + 1 if _SRC_STR in sys.path else 0
    sys.path.insert(insert_at, _ROOT_STR)

from src.data.migrations import analyze_migration, apply_migrations


def find_dbs(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith('.db'):
                yield os.path.join(dirpath, fn)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--db-dir', default='data/db', help='Root directory containing sport DBs')
    p.add_argument('--apply', action='store_true', help='Apply migrations')
    p.add_argument('--yes', action='store_true', help='Assume yes for apply')
    args = p.parse_args()

    dbs = list(find_dbs(args.db_dir))
    if not dbs:
        print('No .db files found under', args.db_dir)
        return 1

    for db in dbs:
        print('---')
        print('DB:', db)
        conn = sqlite3.connect(db)
        try:
            report = analyze_migration(conn)
        except Exception as e:
            print('Analysis failed:', e)
            conn.close()
            continue

        counts = report.get('counts', {})
        print('Games total:', counts.get('games_total'))
        print('Would update games:', counts.get('would_update'))
        print('Collisions:', counts.get('collisions'))
        sample = report.get('sample_updates', [])
        if sample:
            print('Sample updates (rowid, old_gid, new_gid):')
            for s in sample[:10]:
                print(' ', s)

        if args.apply:
            if not args.yes:
                confirm = input(f'Apply migrations to {db}? [y/N] ')
                if confirm.lower() != 'y':
                    print('Skipping apply')
                    conn.close()
                    continue
            print('Applying migrations...')
            try:
                conn.execute('BEGIN')
                apply_migrations(conn)
                conn.commit()
                print('Applied migrations to', db)
            except Exception as e:
                conn.rollback()
                print('Failed to apply migrations:', e)
        conn.close()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
