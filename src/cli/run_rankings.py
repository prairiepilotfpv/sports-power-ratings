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
    from data.paths import db_dir, db_path_for, processed_dir
    from pipelines.run_rankings import run_rankings

    parser = argparse.ArgumentParser(
        description="Run rankings for a given sport/season from the repository database."
    )
    parser.add_argument(
        "--db",
        default=None,
        help=f"SQLite DB path override (default: {db_dir()}/<sport>/<season>.db)",
    )
    parser.add_argument("--sport", required=True, help="Sport identifier (e.g., nba)")
    parser.add_argument(
        "--season", required=True, help="Season identifier (e.g., 2023-24)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ranking model to run (default: run all available models)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Optional output CSV path. Defaults to "
            f"{processed_dir()}/<sport>/<season>/rankings.csv"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output file.",
    )
    args = parser.parse_args()

    print(
        "Warning: src.cli.run_rankings is legacy. Prefer: python -m src.cli.pipeline rank"
    )

    output_path = Path(args.output) if args.output else None
    if output_path is not None and output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {output_path}. Use --overwrite to replace."
        )

    db_path = Path(args.db) if args.db else db_path_for(args.sport, args.season)

    result_path = run_rankings(
        db_path,
        sport=args.sport,
        season=args.season,
        model=args.model,
        output_path=output_path,
    )
    if isinstance(result_path, list):
        for path in result_path:
            print(f"Saved rankings -> {path}")
    else:
        print(f"Saved rankings -> {result_path}")


if __name__ == "__main__":
    main()
