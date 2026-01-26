#!/usr/bin/env python
"""Audit SQLite databases under data/db for basic health and junk signals.

Checks (per DB):
- PRAGMA integrity_check
- games table presence and row counts
- duplicate games (date/home/away)
- legacy game_id format (pipe-delimited)
- season consistency (filename vs games.season)
- suspicious text tokens in selected columns
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path


SEASON_RE = re.compile(r"^\d{4}-\d{2}$")
SUSPICIOUS_KEYWORDS = ["test", "tmp", "dummy", "fake", "experiment", "debug", "scratch"]
SUSPICIOUS_COLUMNS = {
    "bets": ["source", "notes", "workbook"],
    "market_snapshot_staging": ["source", "book"],
    "market_snapshots": ["source", "book"],
    "review_runs": ["notes"],
    "market_line_import_errors": ["source", "message", "raw_row"],
}


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _list_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return sorted(row[0] for row in rows)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({_quote_ident(table)})").fetchall()
    return {row[1] for row in rows}


def _count_rows(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {_quote_ident(table)}").fetchone()[0]


def _infer_seasons(conn: sqlite3.Connection) -> list[str]:
    try:
        rows = conn.execute(
            "SELECT DISTINCT season FROM games WHERE season IS NOT NULL ORDER BY season"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    seasons = [row[0] for row in rows if row and row[0]]
    return seasons


def _scan_suspicious(
    conn: sqlite3.Connection, table: str, column: str
) -> dict[str, int]:
    hits: dict[str, int] = {}
    col_expr = f"LOWER({_quote_ident(column)})"
    for kw in SUSPICIOUS_KEYWORDS:
        pattern = f"%{kw.lower()}%"
        sql = f"SELECT COUNT(*) FROM {_quote_ident(table)} WHERE {col_expr} LIKE ?"
        params = [pattern]
        if kw == "test":
            sql += f" AND {col_expr} NOT LIKE ?"
            params.append("%backtest%")
        try:
            count = conn.execute(sql, params).fetchone()[0]
        except sqlite3.OperationalError:
            continue
        if count:
            hits[kw] = count
    return hits


def _audit_db(db_path: Path) -> dict:
    result: dict = {
        "path": str(db_path),
        "integrity": None,
        "tables": {},
        "errors": [],
        "warnings": [],
        "notes": [],
    }
    try:
        with sqlite3.connect(db_path) as conn:
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            result["integrity"] = integrity
            if integrity != "ok":
                result["errors"].append(f"integrity_check={integrity}")

            tables = _list_tables(conn)
            for table in tables:
                result["tables"][table] = _count_rows(conn, table)

            total_rows = sum(result["tables"].values())
            if total_rows == 0:
                result["warnings"].append("no_rows_in_database")

            if "games" not in tables:
                result["errors"].append("missing_games_table")
                return result

            game_columns = _table_columns(conn, "games")
            game_count = result["tables"].get("games", 0)
            if game_count == 0:
                result["warnings"].append("games_table_empty")

            if {"date", "home_team", "away_team"}.issubset(game_columns):
                duplicates = conn.execute(
                    """
                    SELECT COUNT(*) FROM (
                        SELECT date, home_team, away_team, COUNT(*) AS cnt
                        FROM games
                        GROUP BY date, home_team, away_team
                        HAVING cnt > 1
                    )
                    """
                ).fetchone()[0]
                if duplicates:
                    result["errors"].append(f"duplicate_games={duplicates}")
            else:
                result["warnings"].append("games_missing_date_or_team_columns")

            if "game_id" in game_columns:
                pipe_ids = conn.execute(
                    "SELECT COUNT(*) FROM games WHERE CAST(game_id AS TEXT) LIKE '%|%'"
                ).fetchone()[0]
                if pipe_ids:
                    result["warnings"].append(f"legacy_game_id_pipe_format={pipe_ids}")

            seasons = _infer_seasons(conn)
            if seasons:
                if len(seasons) > 1:
                    result["warnings"].append(
                        f"multiple_seasons_in_games={seasons}"
                    )
                else:
                    result["notes"].append(f"season={seasons[0]}")
            else:
                result["warnings"].append("no_season_values_in_games")

            filename_season = db_path.stem
            if SEASON_RE.match(filename_season):
                if seasons and len(seasons) == 1 and seasons[0] != filename_season:
                    result["warnings"].append(
                        f"filename_season_mismatch={filename_season} vs {seasons[0]}"
                    )
            else:
                result["warnings"].append(f"filename_season_invalid={filename_season}")

            suspicious_hits: list[str] = []
            for table, columns in SUSPICIOUS_COLUMNS.items():
                if table not in tables:
                    continue
                existing = _table_columns(conn, table)
                for column in columns:
                    if column not in existing:
                        continue
                    hits = _scan_suspicious(conn, table, column)
                    for kw, count in hits.items():
                        suspicious_hits.append(f"{table}.{column} contains '{kw}': {count}")
            if suspicious_hits:
                result["warnings"].append("suspicious_text_matches")
                result["notes"].extend(suspicious_hits)
    except sqlite3.Error as exc:
        result["errors"].append(f"sqlite_error={exc}")
    return result


def _print_result(result: dict) -> None:
    print(f"DB: {result['path']}")
    if result["integrity"] is not None:
        print(f"  integrity: {result['integrity']}")
    if result["tables"]:
        print(f"  tables: {len(result['tables'])}")
        for table, count in sorted(result["tables"].items()):
            print(f"    {table}: {count}")
    if result["errors"]:
        for err in result["errors"]:
            print(f"  ERROR: {err}")
    if result["warnings"]:
        for warn in result["warnings"]:
            print(f"  WARN: {warn}")
    if result["notes"]:
        for note in result["notes"]:
            print(f"  note: {note}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit SQLite DB health under data/db")
    parser.add_argument(
        "--db-dir",
        default=None,
        help="Directory containing sport subfolders with .db files (default: data/db)",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        help="Optional JSON output path for audit results",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    db_dir = Path(args.db_dir) if args.db_dir else repo_root / "data" / "db"
    if not db_dir.exists():
        print(f"DB directory not found: {db_dir}")
        return 1

    db_files = sorted(db_dir.rglob("*.db"))
    if not db_files:
        print(f"No .db files found under {db_dir}")
        return 0

    results = [_audit_db(db_path) for db_path in db_files]
    for result in results:
        _print_result(result)
        print("")

    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote JSON results -> {args.json_path}")

    error_count = sum(1 for r in results if r["errors"])
    warn_count = sum(1 for r in results if r["warnings"])
    print(f"Audit complete: {len(results)} DBs, errors={error_count}, warnings={warn_count}")
    return 1 if error_count else 0


if __name__ == "__main__":
    sys.exit(main())
