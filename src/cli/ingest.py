from __future__ import annotations

import argparse
from pathlib import Path

try:  # Allow execution from repository root or nested directories
    from bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - fallback when bootstrap isn't on sys.path
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from bootstrap import ensure_src_on_path

ensure_src_on_path()

from data.repository import save_games
from ingest.sports_reference import parse_sr_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Sports-Reference CSV data into the local database."
    )
    parser.add_argument("input", help="Path to Sports-Reference CSV file.")
    parser.add_argument(
        "--db",
        default="data/app.db",
        help="SQLite DB path (default: data/app.db)",
    )
    parser.add_argument("--sport", help="Sport identifier (e.g., nba)")
    parser.add_argument("--season", help="Season identifier (e.g., 2023-24)")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    games = parse_sr_csv(in_path, sport=args.sport, season=args.season)
    saved = save_games(args.db, games)
    print(f"Saved {saved} games to {args.db}")


if __name__ == "__main__":
    main()
