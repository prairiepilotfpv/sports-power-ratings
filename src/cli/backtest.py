from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    src_str = str(src_dir)
    root_str = str(repo_root)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    if root_str not in sys.path:
        insert_at = sys.path.index(src_str) + 1 if src_str in sys.path else 0
        sys.path.insert(insert_at, root_str)


def main() -> None:
    _ensure_src_on_path()
    from backtest.runner import load_games_df_from_csv, run_backtest
    from data.paths import db_path_for
    from models.registry import get_backtest_model, list_backtest_models, normalize_model_name

    parser = argparse.ArgumentParser(
        description="Run a model backtest on historical games."
    )
    parser.add_argument(
        "--model",
        default="bradley-terry",
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
    parser.add_argument(
        "--sport",
        help="Optional sport identifier for persisting backtest calibration metrics.",
    )
    parser.add_argument(
        "--season",
        help="Optional season identifier for persisting backtest calibration metrics.",
    )
    parser.add_argument(
        "--db",
        help="Optional SQLite DB path to persist backtest calibration metrics.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Enable post-fit probability calibration (Platt/Isotonic) during backtest.",
    )
    parser.add_argument(
        "--calib-dir",
        help="Directory to persist fitted calibrators (default: outputs/calibrators/<model>).",
    )
    parser.add_argument(
        "--calibrator",
        choices=["auto", "platt", "isotonic"],
        default="auto",
        help="Override calibrator selection: 'platt', 'isotonic', or 'auto' (default).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise errors on invalid BT predictions instead of warnings.",
    )
    args = parser.parse_args()

    model_name = normalize_model_name(args.model)
    model_cls = get_backtest_model(model_name)
    games_df = load_games_df_from_csv(
        Path(args.csv),
        sport=args.sport,
        season=args.season,
    )
    db_path = None
    if args.db:
        db_path = Path(args.db)
    elif args.sport and args.season:
        db_path = db_path_for(args.sport, args.season)

    model_kwargs = {}
    if args.strict and model_name in {"bradley-terry"}:
        model_kwargs["strict"] = True

    outputs = run_backtest(
        model_factory=lambda: model_cls(**model_kwargs),
        games_df=games_df,
        start_date=args.start,
        end_date=args.end,
        window=args.window,
        rolling_days=args.rolling_days,
        rolling_games=args.rolling_games,
        output_dir=args.output_dir,
        calibrate=bool(getattr(args, "calibrate", False)),
        calib_dir=Path(args.calib_dir) if getattr(args, "calib_dir", None) else None,
        calibrator_override=(getattr(args, "calibrator", None) or None),
        model_name=args.model,
        db_path=db_path,
        sport=args.sport,
        season=args.season,
    )

    print(f"Saved backtest outputs to {outputs.output_dir}")


if __name__ == "__main__":
    main()
