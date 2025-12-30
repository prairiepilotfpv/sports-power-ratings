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

from backtest.runner import load_games_df_from_db, run_backtest
from data.paths import db_path_for
from models.registry import get_backtest_model, list_backtest_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a model backtest on historical games.")
    parser.add_argument(
        "--model",
        default="bradley_terry_hfa",
        help=f"Model to backtest (choices: {', '.join(list_backtest_models())})",
    )
    parser.add_argument("--start", required=True, help="Backtest start date (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, help="Backtest end date (YYYY-MM-DD).")
    parser.add_argument(
        "--window",
        default="expanding",
        choices=["expanding", "rolling"],
        help="Training window type (default: expanding).",
    )
    parser.add_argument(
        "--rolling-days",
        type=int,
        help="Rolling window size in days (required for rolling).",
    )
    parser.add_argument(
        "--rolling-games",
        type=int,
        help="Rolling window size in games (optional alternative for rolling).",
    )
    parser.add_argument(
        "--db",
        help="Optional SQLite DB path override (default: data/db/<sport>/<season>.db).",
    )
    parser.add_argument("--sport", default="nba", help="Sport identifier (default: nba).")
    parser.add_argument("--season", default="2025-26", help="Season identifier (default: 2025-26).")
    parser.add_argument(
        "--output-dir",
        help="Optional output directory override.",
    )
    args = parser.parse_args()

    model_cls = get_backtest_model(args.model)
    db_path = Path(args.db) if args.db else db_path_for(args.sport, args.season)
    games_df = load_games_df_from_db(db_path, sport=args.sport, season=args.season)

    outputs = run_backtest(
        model_factory=model_cls,
        games_df=games_df,
        start_date=args.start,
        end_date=args.end,
        window=args.window,
        rolling_days=args.rolling_days,
        rolling_games=args.rolling_games,
        output_dir=args.output_dir,
        model_name=args.model,
    )

    print(f"Saved backtest outputs to {outputs.output_dir}")


if __name__ == "__main__":
    main()
