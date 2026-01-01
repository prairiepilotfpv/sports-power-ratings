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
    from backtest.runner import load_games_df_from_csv, run_backtest
    from models.registry import get_backtest_model, list_backtest_models

    parser = argparse.ArgumentParser(
        description="Run a model backtest on historical games."
    )
    parser.add_argument(
        "--model",
        default="bradley_terry_hfa",
        help=f"Model to backtest (choices: {', '.join(list_backtest_models())})",
    )
    parser.add_argument(
        "--start", required=True, help="Backtest start date (YYYY-MM-DD)."
    )
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
        "--csv",
        required=True,
        help="CSV path containing historical games for backtesting.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory override.",
    )
    args = parser.parse_args()

    model_cls = get_backtest_model(args.model)
    games_df = load_games_df_from_csv(Path(args.csv))

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
