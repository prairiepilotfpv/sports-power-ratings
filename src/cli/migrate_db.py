"""DB migration CLI: analyze and optionally apply migrations across DB files.

Usage:
  python -m src.cli.migrate_db --db-dir data/db --apply --yes
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from typing import Iterable

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
