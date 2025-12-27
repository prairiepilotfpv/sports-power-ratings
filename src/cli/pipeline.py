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
from data.repository import save_games
from ingest.normalize import normalize_games
from ingest.sports_reference import parse_sr_csv, parse_sr_csv_text, parse_sr_html
from pipelines.run_rankings import run_rankings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sports power ratings pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser(
        "import",
        aliases=["input"],
        help="Import game results into the per-sport/per-season database.",
    )
    import_parser.add_argument("--sport", required=True, help="Sport identifier (e.g., nba)")
    import_parser.add_argument("--season", required=True, help="Season identifier (e.g., 2024-25)")
    import_parser.add_argument(
        "--source",
        default="sports-reference",
        help="Input source type (default: sports-reference)",
    )
    import_parser.add_argument(
        "--input",
        help="Path to input file (CSV or HTML).",
    )
    import_parser.add_argument(
        "--input-text",
        help="Raw CSV text pasted from Sports-Reference.",
    )
    import_parser.add_argument(
        "--db",
        help="Optional SQLite DB path override (default: data/db/<sport>/<season>.db)",
    )

    rank_parser = subparsers.add_parser(
        "rank",
        aliases=["run_model", "run-model"],
        help="Generate rankings from the per-sport/per-season database.",
    )
    rank_parser.add_argument("--sport", required=True, help="Sport identifier (e.g., nba)")
    rank_parser.add_argument("--season", required=True, help="Season identifier (e.g., 2024-25)")
    rank_parser.add_argument(
        "--model",
        default="bradley-terry",
        help="Ranking model to run (default: bradley-terry)",
    )
    rank_parser.add_argument(
        "--output",
        help="Optional output CSV path. Defaults to data/processed/<sport>/<season>/rankings.csv",
    )
    rank_parser.add_argument(
        "--db",
        help="Optional SQLite DB path override (default: data/db/<sport>/<season>.db)",
    )
    rank_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output file.",
    )

    return parser.parse_args()


def _import_games(args: argparse.Namespace) -> None:
    if args.source != "sports-reference":
        raise ValueError(f"Unsupported source: {args.source}")

    if not args.input and not args.input_text:
        raise ValueError("Provide --input or --input-text for import.")
    if args.input and args.input_text:
        raise ValueError("Provide only one of --input or --input-text.")

    if args.input_text:
        games = parse_sr_csv_text(args.input_text, sport=args.sport, season=args.season)
    else:
        in_path = Path(args.input)
        if not in_path.exists():
            raise FileNotFoundError(f"Input not found: {in_path}")
        if in_path.suffix.lower() in {".html", ".htm"}:
            games = parse_sr_html(in_path, sport=args.sport, season=args.season)
        else:
            games = parse_sr_csv(in_path, sport=args.sport, season=args.season)

    games = normalize_games(games)
    db_path = Path(args.db) if args.db else db_path_for(args.sport, args.season)
    saved = save_games(db_path, games)
    print(f"Saved {saved} games to {db_path}")


def _run_rankings(args: argparse.Namespace) -> None:
    output_path = Path(args.output) if args.output else None
    if output_path is not None and output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Use --overwrite to replace.")

    db_path = Path(args.db) if args.db else db_path_for(args.sport, args.season)
    result_path = run_rankings(
        db_path,
        sport=args.sport,
        season=args.season,
        model=args.model,
        output_path=output_path,
    )
    print(f"Saved rankings -> {result_path}")


def main() -> None:
    args = _parse_args()
    if args.command == "import":
        _import_games(args)
    elif args.command == "rank":
        _run_rankings(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
