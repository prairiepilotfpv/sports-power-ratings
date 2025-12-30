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

from data.paths import db_path_for
from pipelines.run_rankings import run_rankings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run rankings for a given sport/season from the repository database."
    )
    parser.add_argument(
        "--db",
        default=None,
        help="SQLite DB path override (default: data/db/<sport>/<season>.db)",
    )
    parser.add_argument("--sport", required=True, help="Sport identifier (e.g., nba)")
    parser.add_argument("--season", required=True, help="Season identifier (e.g., 2023-24)")
    parser.add_argument(
        "--model",
        default=None,
        help="Ranking model to run (default: run all available models)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional output CSV path. Defaults to data/processed/<sport>/<season>/rankings.csv",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output file.",
    )
    args = parser.parse_args()

    print("Warning: src.cli.run_rankings is legacy. Prefer: python -m src.cli.pipeline rank")

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
