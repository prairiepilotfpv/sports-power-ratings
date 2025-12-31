from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


def _ensure_src_on_path() -> None:
    src_dir = Path(__file__).resolve().parents[2] / "src"
    if importlib.util.find_spec("data") is None and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main() -> None:
    _ensure_src_on_path()
    from data.paths import db_dir, db_path_for
    from data.repository import save_games
    from ingest.registry import get_ingest_source
    from pipelines.ingest import ingest_games, resolve_input_path

    parser = argparse.ArgumentParser(
        description="Ingest Sports-Reference CSV data into the local database."
    )
    parser.add_argument("input", help="Path to Sports-Reference file (CSV or HTML).")
    parser.add_argument(
        "--db",
        default=None,
        help=f"SQLite DB path override (default: {db_dir()}/<sport>/<season>.db)",
    )
    parser.add_argument("--sport", help="Sport identifier (e.g., nba)")
    parser.add_argument("--season", help="Season identifier (e.g., 2023-24)")
    parser.add_argument(
        "--format",
        choices=["auto", "csv", "html"],
        default="auto",
        help="Input format override (default: auto-detect by extension)",
    )
    args = parser.parse_args()

    print(
        "Warning: src.cli.ingest is legacy. Prefer: python -m src.cli.pipeline import"
    )

    ingest_source = get_ingest_source("sports-reference")()
    in_path = resolve_input_path(args.input)
    format_hint = None if args.format == "auto" else args.format
    games = ingest_games(
        ingest_source,
        input_path=in_path,
        input_text=None,
        sport=args.sport or "unknown",
        season=args.season or "unknown",
        format_hint=format_hint,
    )

    db_path = (
        Path(args.db)
        if args.db
        else db_path_for(args.sport or "unknown", args.season or "unknown")
    )
    saved = save_games(db_path, games)
    print(f"Saved {saved} games to {db_path}")


if __name__ == "__main__":
    main()
