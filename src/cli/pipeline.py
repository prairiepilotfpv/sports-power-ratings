"""Command-line pipeline for ingesting, ranking, and projecting sports results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import pandas as pd


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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the CLI argument parser with subcommands for each pipeline step."""
    _ensure_src_on_path()
    from data.paths import db_dir, processed_dir

    parser = argparse.ArgumentParser(description="Sports power ratings pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser(
        "import",
        aliases=["input"],
        help="Import game results into the per-sport/per-season database.",
    )
    import_parser.add_argument(
        "--sport", required=True, help="Sport identifier (e.g., nba)"
    )
    import_parser.add_argument(
        "--season", required=True, help="Season identifier (e.g., 2024-25)"
    )
    import_parser.add_argument(
        "--division",
        help="Optional division identifier (e.g., ncaa-d1) for grouping games.",
    )
    import_parser.add_argument(
        "--conference",
        help="Optional conference identifier (e.g., big-12) for grouping games.",
    )
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
        "--input-dir",
        help=(
            "Directory containing per-conference CSV/HTML files. "
            "Use --conference-from-filename to tag each file by its name."
        ),
    )
    import_parser.add_argument(
        "--input-text",
        help="Raw CSV text pasted from Sports-Reference.",
    )
    import_parser.add_argument(
        "--format",
        choices=["auto", "csv", "html"],
        default="auto",
        help="Input format override (default: auto-detect by extension).",
    )
    import_parser.add_argument(
        "--conference-from-filename",
        action="store_true",
        help="Tag each imported file with a conference name derived from its filename.",
    )
    import_parser.add_argument(
        "--db",
        help=f"Optional SQLite DB path override (default: {db_dir()}/<sport>/<season>.db)",
    )

    rank_parser = subparsers.add_parser(
        "rank",
        aliases=["run_model", "run-model"],
        help="Generate rankings from the per-sport/per-season database.",
    )
    rank_parser.add_argument(
        "--sport", required=True, help="Sport identifier (e.g., nba)"
    )
    rank_parser.add_argument(
        "--season", required=True, help="Season identifier (e.g., 2024-25)"
    )
    rank_parser.add_argument(
        "--division",
        help="Optional division identifier to filter games (e.g., ncaa-d1).",
    )
    rank_parser.add_argument(
        "--conference",
        help="Optional conference identifier to filter games (e.g., big-12).",
    )
    rank_parser.add_argument(
        "--model",
        default=None,
        help="Ranking model to run (default: run all available models)",
    )
    rank_parser.add_argument(
        "--model-params",
        help="JSON string of model parameters to override defaults.",
    )
    rank_parser.add_argument(
        "--model-params-file",
        help="JSON file containing model parameters or per-model overrides.",
    )
    rank_parser.add_argument(
        "--tuned-metric",
        help="Use tuned params for a specific metric instead of the active tuned params.",
    )
    rank_parser.add_argument(
        "-o",
        "--output",
        help=(
            "Optional output CSV path. Defaults to "
            f"{processed_dir()}/<sport>/<season>/rankings.csv"
        ),
    )
    rank_parser.add_argument(
        "--db",
        help=f"Optional SQLite DB path override (default: {db_dir()}/<sport>/<season>.db)",
    )
    rank_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output file.",
    )

    matchup_parser = subparsers.add_parser(
        "matchup",
        aliases=["predict", "predict_matchup"],
        help="Predict a matchup using stored rankings data.",
    )
    matchup_parser.add_argument(
        "--sport", required=True, help="Sport identifier (e.g., nba)"
    )
    matchup_parser.add_argument(
        "--season", required=True, help="Season identifier (e.g., 2024-25)"
    )
    matchup_parser.add_argument(
        "--division",
        help="Optional division identifier to filter games (e.g., ncaa-d1).",
    )
    matchup_parser.add_argument(
        "--conference",
        help="Optional conference identifier to filter games (e.g., big-12).",
    )
    matchup_parser.add_argument(
        "--matchup",
        help='Matchup string like "Eagles vs Cowboys".',
    )
    matchup_parser.add_argument("--home", help="Home team name.")
    matchup_parser.add_argument("--away", help="Away team name.")
    matchup_parser.add_argument(
        "--model",
        default=None,
        help="Ranking model to run (default: run all available models)",
    )
    matchup_parser.add_argument(
        "--model-params",
        help="JSON string of model parameters to override defaults.",
    )
    matchup_parser.add_argument(
        "--model-params-file",
        help="JSON file containing model parameters or per-model overrides.",
    )
    matchup_parser.add_argument(
        "--tuned-metric",
        help="Use tuned params for a specific metric instead of the active tuned params.",
    )
    matchup_parser.add_argument(
        "--db",
        help=f"Optional SQLite DB path override (default: {db_dir()}/<sport>/<season>.db)",
    )

    schedule_parser = subparsers.add_parser(
        "schedule",
        aliases=["calendar", "projections"],
        help=(
            "Export played/upcoming games with model projections for upcoming games "
            "(includes home_rating/away_rating power ratings from build_rankings)."
        ),
    )
    schedule_parser.add_argument(
        "--sport", required=True, help="Sport identifier (e.g., nba)"
    )
    schedule_parser.add_argument(
        "--season", required=True, help="Season identifier (e.g., 2024-25)"
    )
    schedule_parser.add_argument(
        "--division",
        help="Optional division identifier to filter games (e.g., ncaa-d1).",
    )
    schedule_parser.add_argument(
        "--conference",
        help="Optional conference identifier to filter games (e.g., big-12).",
    )
    schedule_parser.add_argument(
        "--model",
        default=None,
        help="Ranking model to use for projections (default: run all available models)",
    )
    schedule_parser.add_argument(
        "--model-params",
        help="JSON string of model parameters to override defaults.",
    )
    schedule_parser.add_argument(
        "--model-params-file",
        help="JSON file containing model parameters or per-model overrides.",
    )
    schedule_parser.add_argument(
        "--tuned-metric",
        help="Use tuned params for a specific metric instead of the active tuned params.",
    )
    schedule_parser.add_argument(
        "--output",
        help=(
            "Optional output path. Defaults to "
            f"{processed_dir()}/<sport>/<season>/schedule_with_projections.xlsx. "
            "Use a .csv extension to force CSV output."
        ),
    )
    schedule_parser.add_argument(
        "--upcoming-only",
        action="store_true",
        help="Only include games without scores (future games).",
    )
    schedule_parser.add_argument(
        "--as-of-date",
        dest="as_of_date",
        help="Optional target date for dashboard/BETS (YYYY-MM-DD or YYYYMMDD).",
    )
    schedule_parser.add_argument(
        "--bets-model",
        dest="bets_model",
        help="Model to populate BETS when multiple models are requested.",
    )
    schedule_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if active tuned params/weights are missing for market ensembles.",
    )
    schedule_parser.add_argument(
        "--market-csv",
        dest="market_csv",
        help="CSV file with market lines to import before building BETS sheet (columns: team_home_raw, team_away_raw, game_date, market_type, selection, line, odds).",
    )
    schedule_parser.add_argument(
        "--default-book",
        dest="default_book",
        help="Default book name for imported market lines (used with --market-csv).",
    )
    schedule_parser.add_argument(
        "--validate-ensemble-weights",
        action="store_true",
        help="Compare tuned ML ensemble weights vs equal weights on completed games; saves report to outputs/ensemble_validation/.",
    )
    schedule_parser.add_argument(
        "--db",
        help=f"Optional SQLite DB path override (default: {db_dir()}/<sport>/<season>.db)",
    )

    market_review_parser = subparsers.add_parser(
        "market-review",
        help="Review OCR staging rows and accept/reject matches.",
    )
    market_review_parser.add_argument("--sport", required=True, help="Sport identifier (e.g., nba)")
    market_review_parser.add_argument("--season", required=True, help="Season identifier (e.g., 2024-25)")
    market_review_parser.add_argument(
        "--status",
        default="unmatched,needs_review",
        help=(
            "Comma-separated match_status filters (default: unmatched,needs_review). "
            "Use 'all' for every status."
        ),
    )
    market_review_parser.add_argument(
        "--auto-match",
        action="store_true",
        help="Attempt to auto-match staging rows using team/date heuristics (skips accept/reject).",
    )
    market_review_parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of staging rows when listing.",
    )
    market_review_parser.add_argument(
        "--accept",
        dest="accept_id",
        type=int,
        help="Staging row id to mark as matched.",
    )
    market_review_parser.add_argument(
        "--game-id",
        dest="game_id",
        help="Game id to persist when accepting a staging row.",
    )
    market_review_parser.add_argument(
        "--match-confidence",
        dest="match_confidence",
        type=float,
        help="Optional manual match_confidence when accepting (default: 1.0).",
    )
    market_review_parser.add_argument(
        "--reject",
        dest="reject_id",
        type=int,
        help="Staging row id to mark as unmatched.",
    )
    market_review_parser.add_argument(
        "--db",
        help=f"Optional SQLite DB path override (default: {db_dir()}/<sport>/<season>.db)",
    )

    market_bets_parser = subparsers.add_parser(
        "market-bets",
        help="Pivot reviewed staging rows into bets with stake presets.",
    )
    market_bets_parser.add_argument("--sport", required=True, help="Sport identifier (e.g., nba)")
    market_bets_parser.add_argument("--season", required=True, help="Season identifier (e.g., 2024-25)")
    market_bets_parser.add_argument(
        "--status",
        default="matched",
        help="Comma-separated match_status filters (default: matched). Use 'all' for every status.",
    )
    market_bets_parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of staging rows when pivoting.",
    )
    market_bets_parser.add_argument(
        "--review-run-id",
        dest="review_run_id",
        help="Optional review_run_id to attach to bets (defaults to staging-<timestamp>).",
    )
    market_bets_parser.add_argument(
        "--stake-preset",
        choices=["half", "unit", "double"],
        default="unit",
        help="Stake preset multiplier applied to unit stake (default: unit=1x).",
    )
    market_bets_parser.add_argument(
        "--unit-stake",
        dest="unit_stake",
        type=float,
        default=1.0,
        help="Base unit stake applied with preset multiplier (default: 1.0).",
    )
    market_bets_parser.add_argument(
        "--default-book",
        dest="default_book",
        help="Fallback book name when staging rows have no book.",
    )
    market_bets_parser.add_argument(
        "--disable-auto-hold",
        action="store_true",
        help="Disable duplicate detection that auto-holds duplicate markets from the same image.",
    )
    market_bets_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and summarize without writing bets.",
    )
    market_bets_parser.add_argument(
        "--db",
        help=f"Optional SQLite DB path override (default: {db_dir()}/<sport>/<season>.db)",
    )

    review_generate_parser = subparsers.add_parser(
        "review-generate",
        help="Generate a review workbook for a given sport/season.",
    )
    review_generate_parser.add_argument("--sport", required=True)
    review_generate_parser.add_argument("--season", required=True)
    review_generate_parser.add_argument("--model", required=True)
    review_generate_parser.add_argument("--db", help="Optional DB path override")
    review_generate_parser.add_argument("--output-dir", help="Optional output directory")
    review_generate_parser.add_argument("--review-run-id", help="Optional existing review_run_id to use")
    review_generate_parser.add_argument("--snapshot-run-id", required=True, help="Market snapshot run id to evaluate")
    review_generate_parser.add_argument("--snapshot-date", help="Filter market snapshots by captured date (YYYY-MM-DD)")
    review_generate_parser.add_argument(
        "--formula-workbook",
        "--formula",
        dest="formula_workbook",
        action="store_true",
        help="Generate a formula-based review workbook (implied_prob/edge/ev formulas).",
    )
    review_generate_parser.set_defaults(include_ocr_raw=True)
    review_ocr_group = review_generate_parser.add_mutually_exclusive_group()
    review_ocr_group.add_argument(
        "--include-ocr-raw",
        dest="include_ocr_raw",
        action="store_true",
        help="Include OCR_RAW sheet with source OCR staging rows (default).",
    )
    review_ocr_group.add_argument(
        "--no-include-ocr-raw",
        dest="include_ocr_raw",
        action="store_false",
        help="Skip the OCR_RAW sheet in the review workbook.",
    )

    report_parser = subparsers.add_parser(
        "report",
        aliases=["excel", "excel_report"],
        help="Generate an Excel report with rankings per model.",
    )

    # Betting subcommands (delegated to src.cli.betting)
    try:
        from src.cli.betting import add_subparser as add_betting_subparser

        add_betting_subparser(subparsers)
    except Exception:
        # Best-effort: if betting module isn't available or errors, keep CLI usable
        pass

    report_parser.add_argument(
        "--sport", required=True, help="Sport identifier (e.g., nba)"
    )
    report_parser.add_argument(
        "--season", required=True, help="Season identifier (e.g., 2024-25)"
    )
    report_parser.add_argument(
        "--division",
        help="Optional division identifier to filter games (e.g., ncaa-d1).",
    )
    report_parser.add_argument(
        "--conference",
        help="Optional conference identifier to filter games (e.g., big-12).",
    )
    report_parser.add_argument(
        "--models",
        help="Comma-separated list of ranking models (default: all available models).",
    )
    report_parser.add_argument(
        "--model-params",
        help="JSON string of model parameters to override defaults.",
    )
    report_parser.add_argument(
        "--model-params-file",
        help="JSON file containing model parameters or per-model overrides.",
    )
    report_parser.add_argument(
        "--output",
        help=(
            "Optional output Excel path. Defaults to "
            f"{processed_dir()}/<sport>/<season>/report.xlsx"
        ),
    )
    report_parser.add_argument(
        "--db",
        help=f"Optional SQLite DB path override (default: {db_dir()}/<sport>/<season>.db)",
    )

    validation_report_parser = subparsers.add_parser(
        "validation-report",
        help="Generate a system validation report (tuning, ensembles, EV).",
    )
    validation_report_parser.add_argument("--sport", required=True, help="Sport identifier (e.g., nba)")
    validation_report_parser.add_argument("--season", required=True, help="Season identifier (e.g., 2024-25)")
    validation_report_parser.add_argument("--db", help="Optional SQLite DB path override")
    validation_report_parser.add_argument(
        "--output-dir",
        help="Optional output directory override (default: outputs/validation/<sport>/<season>).",
    )
    validation_report_parser.add_argument(
        "--days-back",
        type=int,
        default=7,
        help="Rolling window for ML ensemble weight validation (default: 7).",
    )
    validation_report_parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Top tuning runs per group to include in the report (default: 5).",
    )
    validation_report_parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip fresh backtest runs (use stored metrics only).",
    )
    validation_report_parser.add_argument(
        "--backtest-models",
        help="Comma-separated backtest model list (default: all backtest models).",
    )
    validation_report_parser.add_argument(
        "--backtest-window",
        choices=["expanding", "rolling"],
        default="expanding",
        help="Backtest window type (default: expanding).",
    )
    validation_report_parser.add_argument(
        "--backtest-start",
        help="Optional backtest start date (YYYY-MM-DD). Defaults to first scored game.",
    )
    validation_report_parser.add_argument(
        "--backtest-end",
        help="Optional backtest end date (YYYY-MM-DD). Defaults to last scored game.",
    )
    validation_report_parser.add_argument(
        "--backtest-rolling-days",
        type=int,
        help="Rolling window size in days (required if --backtest-window=rolling).",
    )
    validation_report_parser.add_argument(
        "--backtest-rolling-games",
        type=int,
        help="Rolling window size in games (optional alternative for rolling).",
    )
    validation_report_parser.add_argument(
        "--keep-backtest-artifacts",
        action="store_true",
        help="Keep per-run backtest artifacts under outputs/validation/.../backtests.",
    )
    validation_report_parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip running/checking standalone calibration during validation.",
    )
    validation_report_parser.add_argument(
        "--calibration-source-id",
        default="historical",
        help="Calibration source id (default: historical).",
    )
    validation_report_parser.add_argument(
        "--calibration-start",
        help="Optional calibration start date (YYYY-MM-DD).",
    )
    validation_report_parser.add_argument(
        "--calibration-end",
        help="Optional calibration end date (YYYY-MM-DD).",
    )

    backtest_parser = subparsers.add_parser(
        "backtest",
        aliases=["bt"],
        help="Run a model backtest on historical games.",
    )
    backtest_parser.add_argument(
        "--model",
        default="bradley-terry",
        help="Backtest model to run (default: bradley-terry).",
    )
    backtest_parser.add_argument(
        "--model-params",
        help="JSON string of model parameters to override defaults.",
    )
    backtest_parser.add_argument(
        "--model-params-file",
        help="JSON file containing model parameters or per-model overrides.",
    )
    backtest_parser.add_argument(
        "--start", required=True, help="Backtest start date (YYYY-MM-DD)."
    )
    backtest_parser.add_argument(
        "--end", required=True, help="Backtest end date (YYYY-MM-DD)."
    )
    backtest_parser.add_argument(
        "--window",
        default="expanding",
        choices=["expanding", "rolling"],
        help="Training window type (default: expanding).",
    )
    backtest_parser.add_argument(
        "--rolling-days",
        type=int,
        help="Rolling window size in days (required for rolling).",
    )
    backtest_parser.add_argument(
        "--rolling-games",
        type=int,
        help="Rolling window size in games (optional alternative for rolling).",
    )
    backtest_parser.add_argument(
        "--output-dir",
        help="Optional output directory override (default: outputs/backtests/<model>).",
    )
    backtest_parser.add_argument(
        "--csv",
        required=True,
        help="CSV path containing historical games for backtesting.",
    )
    backtest_parser.add_argument(
        "--sport",
        help="Optional sport identifier for persisting backtest calibration metrics.",
    )
    backtest_parser.add_argument(
        "--season",
        help="Optional season identifier for persisting backtest calibration metrics.",
    )
    backtest_parser.add_argument(
        "--db",
        help="Optional SQLite DB path to persist backtest calibration metrics.",
    )
    backtest_parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Enable post-fit probability calibration (Platt/Isotonic) during backtest.",
    )
    backtest_parser.add_argument(
        "--calib-dir",
        help="Directory to persist fitted calibrators (default: outputs/calibrators/<model>).",
    )
    backtest_parser.add_argument(
        "--calibrator",
        choices=["auto", "platt", "isotonic"],
        default="auto",
        help="Override calibrator selection: 'platt', 'isotonic', or 'auto' (default).",
    )

    tune_parser = subparsers.add_parser(
        "tune",
        aliases=["tuning", "backtest_tune"],
        help="Tune model hyperparameters via repeated backtests.",
    )
    tune_parser.add_argument(
        "--model",
        required=True,
        help="Backtest model to tune (e.g., elo, gssd, poisson, toor).",
    )
    tune_parser.add_argument(
        "--start", required=True, help="Backtest start date (YYYY-MM-DD)."
    )
    tune_parser.add_argument(
        "--end", required=True, help="Backtest end date (YYYY-MM-DD)."
    )
    tune_parser.add_argument(
        "--window",
        default="expanding",
        choices=["expanding", "rolling"],
        help="Training window type (default: expanding).",
    )
    tune_parser.add_argument(
        "--rolling-days",
        type=int,
        help="Rolling window size in days (required for rolling).",
    )
    tune_parser.add_argument(
        "--rolling-games",
        type=int,
        help="Rolling window size in games (optional alternative for rolling).",
    )
    tune_parser.add_argument(
        "--metric",
        default="log_loss",
        choices=["log_loss", "brier_score", "mae_margin", "mae_total", "all"],
        help="Metric to optimize (default: log_loss).",
    )
    tune_parser.add_argument(
        "--output-dir",
        help="Optional output directory override (default: outputs/tuning/<model>).",
    )
    tune_parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Number of parallel jobs to run (default: auto). Use 0 for all cores minus one.",
    )
    tune_parser.add_argument(
        "--csv",
        required=True,
        help="CSV path containing historical games for tuning.",
    )
    tune_parser.add_argument(
        "--grid-file",
        help="Optional JSON file defining parameter grids.",
    )
    tune_parser.add_argument(
        "--apply-best",
        action="store_true",
        help="Run a final backtest with best params and persist metrics to the DB.",
    )
    tune_parser.add_argument(
        "--apply-metric",
        default="log_loss",
        choices=["log_loss", "brier_score", "mae_margin", "mae_total"],
        help="Metric to activate when applying tuned params (default: log_loss).",
    )
    tune_parser.add_argument(
        "--allow-worse",
        action="store_true",
        help="Allow worse results than the default parameters (disables improvement guard).",
    )
    tune_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first tuning failure instead of continuing.",
    )
    tune_parser.add_argument(
        "--sport",
        help="Optional sport identifier for persisting best backtest metrics.",
    )
    tune_parser.add_argument(
        "--season",
        help="Optional season identifier for persisting best backtest metrics.",
    )
    tune_parser.add_argument(
        "--db",
        help="Optional SQLite DB path to persist best backtest metrics.",
    )

    tune_batch_parser = subparsers.add_parser(
        "tune-batch",
        help=(
            "Run tuning across many models/metrics for a sport/season and apply the best params."
        ),
    )
    tune_batch_parser.add_argument("--sport", required=True, help="Sport identifier (e.g., nba)")
    tune_batch_parser.add_argument("--season", required=True, help="Season identifier (e.g., 2024-25)")
    tune_batch_parser.add_argument("--start", required=True, help="Backtest start date (YYYY-MM-DD).")
    tune_batch_parser.add_argument("--end", required=True, help="Backtest end date (YYYY-MM-DD).")
    tune_batch_parser.add_argument(
        "--models",
        help=(
            "Optional comma-separated models. Default: union of ensemble_config models or all backtest models."
        ),
    )
    tune_batch_parser.add_argument(
        "--include-all-models",
        action="store_true",
        help="Force tuning of every backtest model (includes experimental variants).",
    )
    tune_batch_parser.add_argument(
        "--include-experimental",
        action="store_true",
        help="Allow experimental models (HFA variants) to be tuned when using defaults.",
    )
    tune_batch_parser.add_argument(
        "--metrics",
        help=(
            "Optional comma-separated metrics. Default: log_loss,mae_margin,mae_total"
        ),
    )
    tune_batch_parser.add_argument(
        "--window",
        default="expanding",
        choices=["expanding", "rolling"],
        help="Training window type (default: expanding).",
    )
    tune_batch_parser.add_argument(
        "--rolling-days",
        type=int,
        help="Rolling window size in days (required for rolling).",
    )
    tune_batch_parser.add_argument(
        "--rolling-games",
        type=int,
        help="Rolling window size in games (optional alternative for rolling).",
    )
    tune_batch_parser.add_argument(
        "--csv",
        required=True,
        help="CSV path containing historical games for tuning.",
    )
    tune_batch_parser.add_argument(
        "--db",
        help="Optional SQLite DB path to persist best backtest metrics.",
    )
    tune_batch_parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Number of parallel jobs to use for tuning runs (default: auto). Use 0 for all cores minus one.",
    )

    init_ensemble_parser = subparsers.add_parser(
        "init-ensemble-config",
        help="Create default ensemble config files for a sport/season (per market).",
    )
    init_ensemble_parser.add_argument("--sport", required=True, help="Sport identifier (e.g., nba)")
    init_ensemble_parser.add_argument("--season", required=True, help="Season identifier (e.g., 2025-26)")
    init_ensemble_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing default ensemble configs if present.",
    )

    tune_model_parser = subparsers.add_parser(
        "tune-model",
        aliases=["tune_model"],
        help="Tune model hyperparameters per sport/season/market and persist results.",
    )
    tune_model_parser.add_argument("--sport", required=True, help="Sport identifier (e.g., nba)")
    tune_model_parser.add_argument("--season", required=True, help="Season identifier (e.g., 2024-25)")
    tune_model_parser.add_argument("--model", required=True, help="Backtest model to tune (e.g., elo, gssd).")
    tune_model_parser.add_argument(
        "--market",
        help="Market identifier(s) (ML, SPREAD, TOTAL). Defaults to all three when omitted.",
    )
    tune_model_parser.add_argument("--start", required=True, help="Backtest start date (YYYY-MM-DD).")
    tune_model_parser.add_argument("--end", required=True, help="Backtest end date (YYYY-MM-DD).")
    tune_model_parser.add_argument(
        "--window",
        default="expanding",
        choices=["expanding", "rolling"],
        help="Training window type (default: expanding).",
    )
    tune_model_parser.add_argument(
        "--rolling-days",
        type=int,
        help="Rolling window size in days (required for rolling).",
    )
    tune_model_parser.add_argument(
        "--rolling-games",
        type=int,
        help="Rolling window size in games (optional alternative for rolling).",
    )
    tune_model_parser.add_argument(
        "--metric",
        help="Metric to optimize (overrides market default).",
    )
    tune_model_parser.add_argument(
        "--market-metrics",
        help="Optional JSON map of market->metric overrides (e.g., '{\"ML\":\"log_loss\"}').",
    )
    tune_model_parser.add_argument(
        "--output-dir",
        help="Optional output directory override (default: outputs/tuning/<sport>/<season>/<model>).",
    )
    tune_model_parser.add_argument(
        "--csv",
        required=True,
        help="CSV path containing historical games for tuning.",
    )
    tune_model_parser.add_argument(
        "--grid-file",
        help="Optional JSON file defining parameter grids.",
    )
    tune_model_parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Number of parallel jobs to run (default: auto). Use 0 for all cores minus one.",
    )
    tune_model_parser.add_argument(
        "--allow-worse",
        action="store_true",
        help="Allow worse results than the default parameters (disables improvement guard).",
    )
    tune_model_parser.add_argument(
        "--activate",
        action="store_true",
        help="Activate tuned params per market after tuning completes.",
    )
    tune_model_parser.add_argument(
        "--db",
        help="Optional SQLite DB path to persist tuning results.",
    )

    activate_parser = subparsers.add_parser(
        "activate-tuning",
        help="Mark a tuning run as active for a model+market (promote to active).",
    )
    activate_parser.add_argument("--sport", required=True, help="Sport identifier (e.g., nba)")
    activate_parser.add_argument("--season", required=True, help="Season identifier (e.g., 2024-25)")
    activate_parser.add_argument("--model", required=True, help="Model id (e.g., elo)")
    activate_parser.add_argument("--market", required=True, help="Market id (ML, SPREAD, TOTAL)")
    activate_parser.add_argument("--metric", help="Metric used by the tuning run (e.g., log_loss)")
    activate_parser.add_argument("--run-id", help="Optional run_id to activate (overrides metric lookup)")
    activate_parser.add_argument("--db", help="Optional DB path override")

    tuning_status_parser = subparsers.add_parser(
        "tuning-status",
        help="Show tuning/run activation status for models and ensembles (read-only).",
    )
    tuning_status_parser.add_argument("--sport", required=True, help="Sport identifier (e.g., nba)")
    tuning_status_parser.add_argument("--season", required=True, help="Season identifier (e.g., 2024-25)")
    tuning_status_parser.add_argument(
        "--models",
        help="Comma-separated list of models to report (default: all registered models).",
    )
    tuning_status_parser.add_argument(
        "--db",
        help="Optional SQLite DB path override (default: data/db/<sport>/<season>.db)",
    )

    show_active_params_parser = subparsers.add_parser(
        "show-active-params",
        help="Show active model params per market with provenance (read-only).",
    )
    show_active_params_parser.add_argument("--sport", required=True, help="Sport identifier (e.g., nba)")
    show_active_params_parser.add_argument("--season", required=True, help="Season identifier (e.g., 2024-25)")
    show_active_params_parser.add_argument(
        "--models",
        help="Comma-separated list of models to report (default: all registered models).",
    )
    show_active_params_parser.add_argument(
        "--db",
        help="Optional SQLite DB path override (default: data/db/<sport>/<season>.db)",
    )

    bootstrap_market_actives_parser = subparsers.add_parser(
        "bootstrap-market-actives",
        help="Populate missing active model market params from tuning runs or defaults.",
    )
    bootstrap_market_actives_parser.add_argument("--sport", required=True, help="Sport identifier (e.g., nba)")
    bootstrap_market_actives_parser.add_argument("--season", required=True, help="Season identifier (e.g., 2024-25)")
    bootstrap_market_actives_parser.add_argument(
        "--model",
        default="all",
        help="Model id to bootstrap (default: all registered models).",
    )
    bootstrap_market_actives_parser.add_argument(
        "--no-ml",
        action="store_true",
        help="Skip ML market defaults/bootstrapping.",
    )
    bootstrap_market_actives_parser.add_argument("--db", help="Optional DB path override")

    tune_ensemble_parser = subparsers.add_parser(
        "tune-ensemble",
        aliases=["tune_ensemble"],
        help="Tune ensemble weights for a market using backtest predictions.",
    )
    tune_ensemble_parser.add_argument("--sport", required=True, help="Sport identifier (e.g., nba)")
    tune_ensemble_parser.add_argument("--season", required=True, help="Season identifier (e.g., 2024-25)")
    tune_ensemble_parser.add_argument("--start-date", required=True, help="Training start date (YYYY-MM-DD).")
    tune_ensemble_parser.add_argument("--end-date", required=True, help="Training end date (YYYY-MM-DD).")
    tune_ensemble_parser.add_argument("--market", default="ML", help="Market identifier (default: ML).")
    tune_ensemble_parser.add_argument(
        "--ensemble",
        default="ensemble_ml_v1",
        help="Ensemble id to tune (default: ensemble_ml_v1).",
    )
    tune_ensemble_parser.add_argument(
        "--models",
        help="Comma-separated list of models to override the active selection (default: use active selection).",
    )
    tune_ensemble_parser.add_argument(
        "--selection-run-id",
        help="Optional selection run_id to force a specific member list instead of the active selection.",
    )
    tune_ensemble_parser.add_argument(
        "--as-of",
        dest="as_of",
        help="Optional cutoff date for training data (YYYY-MM-DD).",
    )
    tune_ensemble_parser.add_argument(
        "--csv",
        help="Optional CSV path of historical games (defaults to DB for sport/season).",
    )
    tune_ensemble_parser.add_argument(
        "--db",
        "--db-path",
        dest="db",
        help="Optional SQLite DB path override for historical games.",
    )
    tune_ensemble_parser.add_argument(
        "--no-activate",
        dest="activate",
        action="store_false",
        help="Skip activation of tuned weights even when they improve the baseline.",
    )
    tune_ensemble_parser.set_defaults(activate=True)

    select_parser = subparsers.add_parser(
        "select-ensemble",
        help="Select ensemble members for a market using completed games.",
    )
    select_parser.add_argument("--sport", required=True, help="Sport identifier (e.g., nba)")
    select_parser.add_argument("--season", required=True, help="Season identifier (e.g., 2024-25)")
    select_parser.add_argument("--market", default="ML", help="Market identifier (default: ML).")
    select_parser.add_argument(
        "--ensemble",
        default="ensemble_ml_v1",
        help="Ensemble id to select (default: ensemble_ml_v1).",
    )
    select_parser.add_argument(
        "--start",
        dest="start",
        help="Optional earliest game date to consider (YYYY-MM-DD).",
    )
    select_parser.add_argument(
        "--end",
        dest="end",
        help="Optional latest game date to consider (YYYY-MM-DD).",
    )
    select_parser.add_argument(
        "--as-of",
        dest="as_of",
        help="Optional cutoff date for dataset (YYYY-MM-DD).",
    )
    select_parser.add_argument(
        "--candidates",
        help="Comma-separated list of candidate models to consider (defaults to ensemble config).",
    )
    select_parser.add_argument(
        "--min-coverage",
        dest="min_coverage",
        type=float,
        default=0.95,
        help="Minimum coverage fraction per model (default: 0.95).",
    )
    select_parser.add_argument(
        "--max-members",
        dest="max_members",
        type=int,
        default=3,
        help="Maximum number of ensemble members to select (default: 3).",
    )
    select_parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-4,
        help="Minimum metric improvement required to add another member (default: 1e-4).",
    )
    select_parser.add_argument(
        "--notes",
        help="Optional notes to persist with the selection run.",
    )
    select_parser.add_argument(
        "--csv",
        help="Optional CSV path of historical games (defaults to DB for sport/season).",
    )
    select_parser.add_argument(
        "--db",
        "--db-path",
        dest="db",
        help="Optional SQLite DB path override for historical games.",
    )
    select_parser.add_argument(
        "--no-activate",
        dest="activate",
        action="store_false",
        help="Skip activating the selection even if it improves.",
    )
    select_parser.set_defaults(activate=True)

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Fit a probability calibrator for an ML source (e.g., ensemble_ml_v1).",
    )
    calibrate_parser.add_argument("--sport", required=True, help="Sport identifier (e.g., nba)")
    calibrate_parser.add_argument("--season", required=True, help="Season identifier (e.g., 2024-25)")
    calibrate_parser.add_argument("--start-date", required=True, help="Training start date (YYYY-MM-DD).")
    calibrate_parser.add_argument("--end-date", required=True, help="Training end date (YYYY-MM-DD).")
    calibrate_parser.add_argument("--market", default="ML", help="Market identifier (default: ML).")
    calibrate_parser.add_argument("--source", required=True, help="Prediction source id to calibrate.")
    calibrate_parser.add_argument(
        "--models",
        help="Comma-separated list of base models to include (default: ML-capable models).",
    )
    calibrate_parser.add_argument(
        "--method",
        choices=["auto", "platt", "isotonic"],
        default="auto",
        help="Calibrator method (default: auto).",
    )
    calibrate_parser.add_argument(
        "--csv",
        help="Optional CSV path of historical games (defaults to DB for sport/season).",
    )
    calibrate_parser.add_argument(
        "--db",
        help="Optional SQLite DB path override for historical games.",
    )

    return parser.parse_args(argv)


def _import_games(args: argparse.Namespace) -> None:
    """Load games from Sports-Reference sources, validate, and persist to SQLite."""
    _ensure_src_on_path()
    from data.paths import db_path_for
    from data.repository import save_games
    from ingest.registry import get_ingest_source
    from pipelines.ingest import ingest_games, resolve_input_path

    if args.source != "sports-reference":
        raise ValueError(f"Unsupported source: {args.source}")

    input_dir = getattr(args, "input_dir", None)
    input_sources = [bool(args.input), bool(args.input_text), bool(input_dir)]
    if sum(input_sources) == 0:
        raise ValueError(
            "Provide --input or --input-text (or --input-dir) for import."
        )
    if sum(input_sources) > 1:
        raise ValueError(
            "Provide only one of --input, --input-dir, or --input-text."
        )

    ingest_source = get_ingest_source(args.source)()
    division = getattr(args, "division", None)
    conference = getattr(args, "conference", None)
    raw_format = getattr(args, "format", "auto")
    format_hint = None if raw_format == "auto" else raw_format
    games = []

    if input_dir:
        input_dir = Path(input_dir)
        if not input_dir.exists() or not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        files = sorted(
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".csv", ".html", ".htm"}
        )
        if not files:
            raise ValueError(f"No CSV/HTML files found in {input_dir}")
        conference_from_filename = getattr(args, "conference_from_filename", False)
        for path in files:
            file_conference = conference
            if conference_from_filename and not file_conference:
                file_conference = _normalize_conference_name(path.stem)
            games.extend(
                ingest_games(
                    ingest_source,
                    input_path=path,
                    input_text=None,
                    sport=args.sport,
                    season=args.season,
                    division=division,
                    conference=file_conference,
                    format_hint=format_hint,
                )
            )
    else:
        input_path = resolve_input_path(args.input) if args.input else None
        games = ingest_games(
            ingest_source,
            input_path=input_path,
            input_text=args.input_text,
            sport=args.sport,
            season=args.season,
            division=division,
            conference=conference,
            format_hint=format_hint,
        )
    db_path = Path(args.db) if args.db else db_path_for(args.sport, args.season)
    _echo_db_path(db_path)
    saved = save_games(db_path, games)
    print(f"Saved {saved} games to {db_path}")


def _run_rankings(args: argparse.Namespace) -> None:
    """Generate rankings for a sport/season and persist the CSV."""
    _ensure_src_on_path()
    from data.paths import db_path_for
    from pipelines.run_rankings import run_rankings

    model_params = _parse_json_arg(args.model_params)
    output_path = Path(args.output) if args.output else None
    if output_path is not None and output_path.exists() and not args.overwrite:
        output_path = _next_available_path(output_path)
        print(
            f"Output exists: {args.output}. Writing to {output_path}. Use --overwrite to replace."
        )

    db_path = Path(args.db) if args.db else db_path_for(args.sport, args.season)
    _echo_db_path(db_path)
    result_path = run_rankings(
        db_path,
        sport=args.sport,
        season=args.season,
        division=args.division,
        conference=args.conference,
        model=args.model,
        output_path=output_path,
        model_params=model_params,
        model_params_file=args.model_params_file,
        tuned_metric=args.tuned_metric,
    )
    if isinstance(result_path, list):
        for path in result_path:
            print(f"Saved rankings -> {path}")
    else:
        print(f"Saved rankings -> {result_path}")


def _next_available_path(output_path: Path) -> Path:
    """Return the first unused filename by appending a numeric suffix."""
    if not output_path.exists():
        return output_path
    stem = output_path.stem
    suffix = output_path.suffix
    parent = output_path.parent
    for index in range(1, 1000):
        candidate = parent / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(
        f"Output exists: {output_path}. Tried 999 alternate names. Use --overwrite to replace."
    )


def _normalize_conference_name(value: str) -> str:
    """Normalize a filename stem into a conference identifier."""
    normalized = value.strip().lower()
    normalized = normalized.replace(" ", "-").replace("_", "-")
    normalized = "".join(ch for ch in normalized if ch.isalnum() or ch == "-")
    return normalized


def _parse_matchup_text(text: str) -> tuple[str, str]:
    """Normalize a human-friendly matchup string into home/away names."""
    cleaned = (
        text.replace("VS", "vs")
        .replace("Vs", "vs")
        .replace("v.", "vs")
        .replace(" v ", " vs ")
    )
    if "vs" not in cleaned:
        raise ValueError("Matchup must include 'vs' between team names.")
    parts = [part.strip() for part in cleaned.split("vs") if part.strip()]
    if len(parts) != 2:
        raise ValueError("Matchup must include exactly two teams.")
    return parts[0], parts[1]


def _run_matchup(args: argparse.Namespace) -> None:
    """Predict a single matchup using stored rankings data."""
    _ensure_src_on_path()
    from data.paths import db_path_for
    from models.registry import list_models
    from pipelines.matchups import format_matchup, predict_matchup

    home = args.home
    away = args.away
    if args.matchup:
        home, away = _parse_matchup_text(args.matchup)

    if not home or not away:
        raise ValueError("Provide --matchup or both --home and --away.")

    model_params = _parse_json_arg(args.model_params)
    db_path = Path(args.db) if args.db else db_path_for(args.sport, args.season)
    _echo_db_path(db_path)
    models = list_models() if args.model is None else [args.model]
    for model in models:
        prediction = predict_matchup(
            db_path,
            sport=args.sport,
            season=args.season,
            division=args.division,
            conference=args.conference,
            home_team=home,
            away_team=away,
            model=model,
            model_params=model_params,
            model_params_file=args.model_params_file,
            tuned_metric=args.tuned_metric,
        )
        line, metrics = format_matchup(prediction)
        print(f"[{model}] {line}")
        print(f"[{model}] Spread: {metrics['spread']:.2f}")
        print(f"[{model}] Total Points: {metrics['total_points']:.2f}")
        print(
            f"[{model}] Params source: {metrics.get('params_source')}, "
            f"tuned metric used: {metrics.get('tuned_metric_used')}"
        )


def _run_schedule(args: argparse.Namespace) -> None:
    """Export the schedule with projections for played and upcoming games."""
    _ensure_src_on_path()
    from data.paths import db_path_for
    from pipelines.schedule import (
        build_schedule_excel_report,
        build_schedule_with_projections,
    )

    model_params = _parse_json_arg(args.model_params)
    db_path = Path(args.db) if args.db else db_path_for(args.sport, args.season)

    # Import market CSV into market_lines if provided
    market_csv = getattr(args, "market_csv", None)
    if market_csv:
        from src.data.market_lines import import_market_csv

        csv_path = Path(market_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"Market CSV not found: {csv_path}")
        default_book = getattr(args, "default_book", None)
        as_of_date = getattr(args, "as_of_date", None)
        result = import_market_csv(
            db_path,
            csv_path=csv_path,
            sport=args.sport,
            season=args.season,
            default_book=default_book,
            date_filter=as_of_date,
        )
        print(
            f"market-csv import: rows={result.get('rows_loaded')} "
            f"inserted={result.get('inserted')} "
            f"unmatched={result.get('unmatched')} "
            f"filtered={result.get('date_filtered')}"
        )
        if result.get("unmatched", 0) > 0:
            reasons = result.get("unmatched_reasons", {})
            print(f"  unmatched reasons: {reasons}")
    _echo_db_path(db_path)
    output_path = Path(args.output) if args.output else None
    if getattr(args, "as_of_date", None):
        args.as_of_date = _require_iso_date(args.as_of_date, field="--as-of-date")
    if output_path is not None and output_path.suffix.lower() == ".csv":
        result_path = build_schedule_with_projections(
            db_path,
            sport=args.sport,
            season=args.season,
            division=args.division,
            conference=args.conference,
            model=args.model,
            output_path=output_path,
            upcoming_only=args.upcoming_only,
            model_params=model_params,
            model_params_file=args.model_params_file,
            tuned_metric=args.tuned_metric,
        )
        if isinstance(result_path, list):
            for path in result_path:
                print(f"Saved schedule with projections -> {path}")
        else:
            print(f"Saved schedule with projections -> {result_path}")
        return

    result_path = build_schedule_excel_report(
        db_path,
        sport=args.sport,
        season=args.season,
        division=args.division,
        conference=args.conference,
        model=args.model,
        output_path=output_path,
        upcoming_only=args.upcoming_only,
        model_params=model_params,
        model_params_file=args.model_params_file,
        tuned_metric=args.tuned_metric,
        as_of_date=getattr(args, "as_of_date", None),
        bets_model=getattr(args, "bets_model", None),
        strict=bool(getattr(args, "strict", False)),
    )
    print(f"Saved schedule workbook -> {result_path}")
    
    # Optional: save BETS predictions to DB and run validation
    if getattr(args, "validate_ensemble_weights", False):
        try:
            from data.bets_repository import save_bets_predictions
            from pipelines.ensemble_weight_validation import (
                validate_ensemble_ml_weights,
                save_validation_report,
            )
            
            # Load BETS sheet and save predictions to database
            bets_df = pd.read_excel(result_path, sheet_name="BETS")
            n_saved = save_bets_predictions(
                db_path,
                bets_df=bets_df,
                sport=args.sport,
                season=args.season,
            )
            print(f"  Saved {n_saved} BETS predictions to database")
            
            # Run validation using DB data (7-day rolling window by default)
            validation_result = validate_ensemble_ml_weights(
                db_path=db_path,
                sport=args.sport,
                season=args.season,
                market="ML",
                days_back=7,
            )
            
            if validation_result:
                report_path = save_validation_report(
                    validation_result=validation_result,
                    sport=args.sport,
                    season=args.season,
                )
                if report_path:
                    print(f"  Ensemble weight validation report -> {report_path}")
        except Exception as e:
            print(f"Warning: Ensemble weight validation failed (non-fatal): {e}")


def _run_market_review(args: argparse.Namespace) -> None:
    """List or update staging rows for manual review."""
    _ensure_src_on_path()
    from data.paths import db_path_for
    from pipelines import market_review as review_pipeline

    db_path = Path(args.db) if args.db else db_path_for(args.sport, args.season)
    _echo_db_path(db_path)
    raw_status = getattr(args, "status", None)
    statuses = None
    if raw_status and raw_status.lower() != "all":
        statuses = [s.strip() for s in raw_status.split(",") if s.strip()]

    limit = getattr(args, "limit", None)
    accept_id = getattr(args, "accept_id", None)
    reject_id = getattr(args, "reject_id", None)

    if accept_id and reject_id:
        raise ValueError("Provide only one of --accept or --reject.")

    if getattr(args, "auto_match", False):
        if accept_id or reject_id:
            raise ValueError("--auto-match cannot be combined with --accept/--reject.")
        summary = review_pipeline.auto_match_rows(
            db_path,
            sport=args.sport,
            season=args.season,
        )
        print(
            "market-review: auto-match "
            f"matched={summary.get('matched')} skipped={summary.get('skipped')} total={summary.get('total')}"
        )
        return

    if accept_id is not None:
        if not args.game_id:
            raise ValueError("--game-id is required when accepting a staging row.")
        updated = review_pipeline.accept_match(
            db_path,
            staging_id=accept_id,
            game_id=args.game_id,
            match_confidence=getattr(args, "match_confidence", None),
        )
        print(
            "Accepted staging "
            f"{updated.get('id')} -> game_id={updated.get('game_id')} "
            f"status={updated.get('match_status')} confidence={updated.get('match_confidence')}"
        )
        return

    if reject_id is not None:
        updated = review_pipeline.reject_match(db_path, staging_id=reject_id)
        print(
            "Rejected staging "
            f"{updated.get('id')} -> status={updated.get('match_status')}"
        )
        return

    rows = review_pipeline.list_staging_rows(
        db_path, match_statuses=statuses, limit=limit
    )
    if not rows:
        print("No staging rows found for given filters.")
        return

    for r in rows:
        teams = f"{r.get('team_home_raw') or '?'} vs {r.get('team_away_raw') or '?'}"
        captured = r.get("captured_at") or "-"
        print(
            f"#{r.get('id')} [{r.get('match_status')}] "
            f"conf={r.get('match_confidence')} game={r.get('game_id') or '-'} "
            f"market={r.get('market_type') or '-'} sel={r.get('selection') or '-'} "
            f"line={r.get('line')} odds={r.get('odds')} teams={teams} "
            f"captured={captured} book={r.get('book') or '-'} "
            f"hold={r.get('hold_reason') or '-'}"
        )


def _run_market_bets(args: argparse.Namespace) -> None:
    """Convert reviewed staging rows into bet entries."""
    _ensure_src_on_path()
    from data.paths import db_path_for
    from pipelines import staging_bets as staging_bets_pipeline

    db_path = Path(args.db) if args.db else db_path_for(args.sport, args.season)
    _echo_db_path(db_path)
    raw_status = getattr(args, "status", None)
    statuses = None
    if raw_status and raw_status.lower() != "all":
        statuses = [s.strip() for s in raw_status.split(",") if s.strip()]

    result = staging_bets_pipeline.pivot_staging_to_bets(
        db_path,
        review_run_id=getattr(args, "review_run_id", None),
        match_statuses=statuses,
        limit=getattr(args, "limit", None),
        stake_preset=getattr(args, "stake_preset", "unit"),
        unit_stake=getattr(args, "unit_stake", 1.0),
        default_book=getattr(args, "default_book", None),
        auto_hold_duplicates=not getattr(args, "disable_auto_hold", False),
        dry_run=bool(getattr(args, "dry_run", False)),
    )
    print(
        "market-bets: "
        f"review_run_id={result.get('review_run_id')} "
        f"stake={result.get('stake')} "
        f"inserted={result.get('inserted')} held={result.get('held')} skipped={result.get('skipped')}"
    )


def _run_report(args: argparse.Namespace) -> None:
    """Build an Excel report with one sheet per model."""
    _ensure_src_on_path()
    from data.paths import db_path_for
    from pipelines.excel_report import build_excel_report

    model_params = _parse_json_arg(args.model_params)
    db_path = Path(args.db) if args.db else db_path_for(args.sport, args.season)
    _echo_db_path(db_path)
    output_path = Path(args.output) if args.output else None
    models = None
    if args.models:
        models = [model.strip() for model in args.models.split(",") if model.strip()]
    result_path = build_excel_report(
        db_path,
        sport=args.sport,
        season=args.season,
        division=args.division,
        conference=args.conference,
        models=models,
        output_path=output_path,
        model_params=model_params,
        model_params_file=args.model_params_file,
    )
    if isinstance(result_path, list):
        for path in result_path:
            print(f"Saved Excel report -> {path}")
    else:
        print(f"Saved Excel report -> {result_path}")


def _run_validation_report(args: argparse.Namespace) -> None:
    """Generate a system validation report for tuning/ensembles/EV."""
    _ensure_src_on_path()
    from data.paths import db_path_for
    from pipelines.validation_report import run_validation_report

    db_path = Path(args.db) if args.db else db_path_for(args.sport, args.season)
    _echo_db_path(db_path)
    output_dir = Path(args.output_dir) if getattr(args, "output_dir", None) else None
    backtest_models = None
    if getattr(args, "backtest_models", None):
        backtest_models = [m.strip() for m in args.backtest_models.split(",") if m.strip()]
    backtest_window = getattr(args, "backtest_window", "expanding")
    if (
        backtest_window == "rolling"
        and getattr(args, "backtest_rolling_days", None) is None
        and getattr(args, "backtest_rolling_games", None) is None
    ):
        raise ValueError("Rolling backtests require --backtest-rolling-days or --backtest-rolling-games.")
    outputs = run_validation_report(
        db_path=db_path,
        sport=args.sport,
        season=args.season,
        output_dir=output_dir,
        days_back=int(getattr(args, "days_back", 7)),
        top_n=int(getattr(args, "top_n", 5)),
        run_backtests=not bool(getattr(args, "skip_backtest", False)),
        backtest_models=backtest_models,
        backtest_window=backtest_window,
        backtest_start=getattr(args, "backtest_start", None),
        backtest_end=getattr(args, "backtest_end", None),
        backtest_rolling_days=getattr(args, "backtest_rolling_days", None),
        backtest_rolling_games=getattr(args, "backtest_rolling_games", None),
        keep_backtest_artifacts=bool(getattr(args, "keep_backtest_artifacts", False)),
        run_calibration=not bool(getattr(args, "skip_calibration", False)),
        calibration_source_id=getattr(args, "calibration_source_id", "historical"),
        calibration_start=getattr(args, "calibration_start", None),
        calibration_end=getattr(args, "calibration_end", None),
    )
    print(f"Validation report -> {outputs.report_path}")
    print(f"Validation workbook -> {outputs.workbook_path}")
    if outputs.summary_path:
        print(f"Validation summary -> {outputs.summary_path}")
    for name, path in outputs.frame_paths.items():
        print(f"Validation data ({name}) -> {path}")


def _run_backtest(args: argparse.Namespace) -> None:
    """Run a backtest pipeline for a single model."""
    _ensure_src_on_path()
    from data.paths import db_path_for
    from pipelines.backtest import run_backtest_pipeline

    model_params = _parse_json_arg(args.model_params)
    output_dir = Path(args.output_dir) if args.output_dir else None
    db_path = None
    if args.db:
        db_path = Path(args.db)
    elif args.sport and args.season:
        db_path = db_path_for(args.sport, args.season)
    if db_path is not None:
        _echo_db_path(db_path)
    outputs = run_backtest_pipeline(
        csv_path=Path(args.csv),
        model=args.model,
        start_date=args.start,
        end_date=args.end,
        window=args.window,
        rolling_days=args.rolling_days,
        rolling_games=args.rolling_games,
        output_dir=output_dir,
        model_params=model_params,
        db_path=db_path,
        sport=args.sport,
        season=args.season,
        calibrate=bool(getattr(args, "calibrate", False)),
        calib_dir=Path(args.calib_dir) if getattr(args, "calib_dir", None) else None,
        calibrator_override=(getattr(args, "calibrator", None) or None),
    )
    print(f"Saved backtest outputs to {outputs.output_dir}")


def _run_tuning(args: argparse.Namespace) -> None:
    """Run hyperparameter tuning via backtest grid search."""
    _ensure_src_on_path()
    from data.paths import db_path_for
    from models.registry import list_backtest_models, normalize_model_name
    from pipelines.tuning import list_metrics, run_tuning_pipeline
    from pipelines.tuning_policy import default_active_metric_for_model

    output_dir = Path(args.output_dir) if args.output_dir else None
    grid_override = None
    if args.grid_file:
        with Path(args.grid_file).open("r", encoding="utf-8") as handle:
            grid_override = json.load(handle)
    db_path = None
    if args.db:
        db_path = Path(args.db)
    elif args.sport and args.season:
        db_path = db_path_for(args.sport, args.season)
    if db_path is not None:
        _echo_db_path(db_path)
    all_forecast_models = ["bradley-terry", "toor", "gssd", "elo", "poisson"]
    models_to_run = (
        all_forecast_models if args.model == "all" else [args.model]
    )
    metrics_to_run = list_metrics() if args.metric == "all" else [args.metric]
    apply_metric = args.apply_metric.strip().lower()

    summary_rows: list[dict[str, object]] = []
    errors: list[str] = []

    def _resolve_output_dir(model_name: str, metric: str) -> Path:
        if output_dir is not None:
            return output_dir / model_name / metric
        if args.sport and args.season:
            return Path("outputs/tuning") / args.sport / args.season / model_name / metric
        return Path("outputs/tuning") / model_name / metric

    for model in models_to_run:
        model_name = normalize_model_name(model)
        active_metric = None
        if args.metric == "all" and args.apply_best:
            active_metric = default_active_metric_for_model(model_name)
        for metric in metrics_to_run:
            run_output_dir = _resolve_output_dir(model_name, metric)
            apply_best = args.apply_best
            if args.metric == "all" and args.apply_best:
                apply_best = metric == active_metric
            try:
                outputs = run_tuning_pipeline(
                    csv_path=Path(args.csv),
                    model=model_name,
                    start_date=args.start,
                    end_date=args.end,
                    window=args.window,
                    rolling_days=args.rolling_days,
                    rolling_games=args.rolling_games,
                    metric=metric,
                    output_dir=run_output_dir,
                    grid_override=grid_override,
                    apply_best=apply_best,
                    require_improvement=not args.allow_worse,
                    db_path=db_path,
                    jobs=getattr(args, "jobs", 0),
                    sport=args.sport,
                    season=args.season,
                )
                run_id = (
                    str(outputs.results.iloc[0]["run_id"])
                    if not outputs.results.empty
                    else ""
                )
                summary_rows.append(
                    {
                        "model": model_name,
                        "metric": metric,
                        "run_id": run_id,
                        "baseline_score": outputs.baseline_score,
                        "best_score": outputs.best_score,
                        "improved": outputs.improved,
                        "applied": outputs.applied,
                        "output_dir": str(outputs.output_dir),
                        "error": None,
                    }
                )
                if outputs.improved:
                    print(
                        "Best params -> "
                        f"{outputs.best_params} (score={outputs.best_score:.4f}) "
                        f"saved in {outputs.output_dir}"
                    )
                else:
                    print(
                        "No improvement over baseline. "
                        f"Baseline score={outputs.baseline_score:.4f}; "
                        "best candidate was rejected."
                    )
            except Exception as exc:
                summary_rows.append(
                    {
                        "model": model_name,
                        "metric": metric,
                        "run_id": None,
                        "baseline_score": None,
                        "best_score": None,
                        "improved": False,
                        "applied": False,
                        "output_dir": str(run_output_dir),
                        "error": str(exc),
                    }
                )
                errors.append(f"{model_name}/{metric}: {exc}")
                if args.fail_fast:
                    raise

    if args.model == "all" or args.metric == "all":
        if summary_rows:
            from datetime import datetime, timezone
            import pandas as pd

            summary_dir = Path("outputs/tuning")
            if args.sport and args.season:
                summary_dir = summary_dir / args.sport / args.season / "_all"
            else:
                summary_dir = summary_dir / "_all"
            summary_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            summary_df = pd.DataFrame(summary_rows)
            csv_path = summary_dir / f"tune_summary_{timestamp}.csv"
            json_path = summary_dir / f"tune_summary_{timestamp}.json"
            summary_df.to_csv(csv_path, index=False)
            json_path.write_text(
                summary_df.to_json(orient="records", indent=2),
                encoding="utf-8",
            )
            print(f"Saved tuning summary -> {csv_path}")
            print(f"Saved tuning summary -> {json_path}")

    if errors:
        print("Tuning completed with errors:")
        for error in errors:
            print(f"- {error}")


def _run_tune_batch(args: argparse.Namespace) -> None:
    """Wrapper to run tuning across many models/metrics for a season."""
    _ensure_src_on_path()
    from data.paths import db_path_for
    from pipelines.tune_batch import run_tune_batch, summarize_tune_batch

    start_date = _require_iso_date(args.start, field="--start")
    end_date = _require_iso_date(args.end, field="--end")
    db_path = Path(args.db) if args.db else db_path_for(args.sport, args.season)
    _echo_db_path(db_path)

    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    metrics = None
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    results = run_tune_batch(
        sport=args.sport,
        season=args.season,
        start_date=start_date,
        end_date=end_date,
        csv_path=Path(args.csv),
        db_path=db_path,
        models=models,
        metrics=metrics,
        window=args.window,
        rolling_days=args.rolling_days,
        rolling_games=args.rolling_games,
        include_all_models=args.include_all_models,
        include_experimental=args.include_experimental,
        jobs=getattr(args, "jobs", 0),
    )

    leaderboard = summarize_tune_batch(results)
    print("TUNE-BATCH LEADERBOARD")
    for metric, rows in leaderboard.items():
        print(f"metric={metric}")
        for row in rows:
            score = row.get("best_score")
            rid = row.get("run_id")
            model = row.get("model")
            print(f"  {model}: score={score} run_id={rid}")


def _run_init_ensemble_config(args: argparse.Namespace) -> None:
    _ensure_src_on_path()
    from pipelines.ensemble_config_init import init_default_ensemble_configs

    results = init_default_ensemble_configs(
        sport=args.sport,
        season=args.season,
        overwrite=args.overwrite,
    )
    for result in results:
        if result.created:
            print(f"Created {result.market} default config -> {result.path}")
        elif result.skipped:
            print(f"Skipped {result.market} (exists) -> {result.path}")


def _run_tune_model(args: argparse.Namespace) -> None:
    _ensure_src_on_path()
    from pipelines.market_tuning import SUPPORTED_MARKETS, run_model_markets_tuning
    from data.paths import db_path_for

    start_date = _require_iso_date(args.start, field="--start")
    end_date = _require_iso_date(args.end, field="--end")
    grid_override = None
    if args.grid_file:
        grid_override = _load_grid_override(args.grid_file)
    db_path = args.db or db_path_for(args.sport, args.season)
    _echo_db_path(Path(db_path))
    markets = _parse_markets_arg(args.market)
    metric_overrides = _parse_market_metrics_arg(args.market_metrics)
    target_markets = markets if markets is not None else list(SUPPORTED_MARKETS)
    combined_metrics: dict[str, str] = {}
    if args.metric:
        metric_value = args.metric.strip()
        if metric_value:
            for market in target_markets:
                combined_metrics[market] = metric_value
    if metric_overrides:
        combined_metrics.update(metric_overrides)
    if not combined_metrics:
        metric_map = None
    else:
        metric_map = combined_metrics
    outcomes = run_model_markets_tuning(
        sport=args.sport,
        season=args.season,
        model=args.model,
        markets=markets,
        start_date=start_date,
        end_date=end_date,
        window=args.window,
        rolling_days=args.rolling_days,
        rolling_games=args.rolling_games,
        csv_path=args.csv,
        output_dir=args.output_dir,
        grid_override=grid_override,
        db_path=db_path,
        metric_overrides=metric_map,
        allow_worse=args.allow_worse,
        jobs=args.jobs,
        activate_best=bool(getattr(args, "activate", False)),
    )
    succeeded: list[str] = []
    failed: list[str] = []
    for outcome in outcomes:
        if outcome.result:
            result = outcome.result
            activated_flag = "yes" if result.activated else "no"
            print(
                f"{result.market}: metric={result.metric_optimized} "
                f"best_score={result.best_score} params_source={result.params_source} "
                f"activated={activated_flag}"
            )
            succeeded.append(result.market)
        else:
            print(f"{outcome.market}: failed -> {outcome.error}")
            failed.append(outcome.market)
    if not failed:
        overall = "success"
    elif succeeded:
        overall = "partial success"
    else:
        overall = "failure"
    print(
        f"Model tuning overall: {overall} "
        f"({len(succeeded)} succeeded, {len(failed)} failed)"
    )


def _run_tune_ensemble(args: argparse.Namespace) -> None:
    _ensure_src_on_path()
    from pipelines.market_tuning import run_ensemble_market_tuning
    from data.paths import db_path_for

    start_date = _require_iso_date(args.start_date, field="--start-date")
    end_date = _require_iso_date(args.end_date, field="--end-date")
    selection_models = None
    if args.models:
        selection_models = [m.strip() for m in args.models.split(",") if m.strip()]
    selection_run_id = getattr(args, "selection_run_id", None)
    as_of = _optional_iso_date(getattr(args, "as_of", None), field="--as-of")
    db_path = _resolve_db_path(args)
    if db_path is None:
        raise ValueError("DB path could not be resolved; pass --db or --sport/--season")
    _echo_db_path(Path(db_path))
    result = run_ensemble_market_tuning(
        sport=args.sport,
        season=args.season,
        market=args.market,
        start_date=start_date,
        end_date=end_date,
        as_of_date=as_of,
        ensemble_id=args.ensemble,
        selection_models=selection_models,
        selection_run_id=selection_run_id,
        csv_path=args.csv,
        db_path=db_path,
        activate=bool(getattr(args, "activate", True)),
    )
    print(f"Ensemble tuned on {result.games} games for {result.selection_models}.")
    print(f"Best score ({result.metric_optimized}) = {result.best_score}")
    if result.baseline_score is not None and result.baseline_score != float("inf"):
        print(f"Baseline (equal weights) = {result.baseline_score}")
    print(f"Activated tuned weights: {'yes' if result.activated else 'no'}")
    print(f"Saved weights -> {result.artifact_path}")
    print(f"Stored tuning run -> {result.run_id}")
    if result.selection_run_id:
        print(f"Selection run -> {result.selection_run_id}")


def _run_select_ensemble(args: argparse.Namespace) -> None:
    _ensure_src_on_path()
    from pipelines.ensemble_tuning import run_market_ensemble_selection

    start_date = _optional_iso_date(args.start, field="--start")
    end_date = _optional_iso_date(args.end, field="--end")
    as_of_date = _optional_iso_date(args.as_of, field="--as-of")
    candidate_models = None
    if args.candidates:
        candidate_models = [m.strip() for m in args.candidates.split(",") if m.strip()]
    db_path = _resolve_db_path(args)
    if db_path is None:
        raise ValueError("DB path could not be resolved; pass --db or --sport/--season")
    _echo_db_path(Path(db_path))
    selection, activated = run_market_ensemble_selection(
        sport=args.sport,
        season=args.season,
        market=args.market,
        ensemble_id=args.ensemble,
        start_date=start_date,
        end_date=end_date,
        as_of_date=as_of_date,
        candidates=candidate_models,
        min_coverage=args.min_coverage,
        max_members=args.max_members,
        epsilon=args.epsilon,
        activate=bool(getattr(args, "activate", True)),
        notes=args.notes,
        csv_path=args.csv,
        db_path=db_path,
    )
    print(f"Selection run -> run_id={selection.run_id}")
    print(f"Selected models: {', '.join(selection.selected_models)}")
    print(
        f"Score ({selection.objective_metric}) = {selection.score:.6f} "
        f"on {selection.games} games (as-of {selection.metadata.asof})"
    )
    print(f"Activated selection: {'yes' if activated else 'no'}")


def _run_calibrate_ensemble(args: argparse.Namespace) -> None:
    _ensure_src_on_path()
    from pipelines.ensemble_tuning import calibrate_ml_ensemble

    if args.market.strip().upper() != "ML":
        raise ValueError("Only ML calibration is supported for now.")
    start_date = _require_iso_date(args.start_date, field="--start-date")
    end_date = _require_iso_date(args.end_date, field="--end-date")
    model_list = None
    if args.models:
        model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    out_path = calibrate_ml_ensemble(
        sport=args.sport,
        season=args.season,
        start_date=start_date,
        end_date=end_date,
        ensemble_id=args.source,
        models=model_list,
        csv_path=args.csv,
        db_path=args.db,
        method=args.method,
    )
    print(f"Saved calibrator -> {out_path}")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: route subcommands to their handlers."""
    import logging
    
    # Configure logging to file for debugging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('tmp_pipeline_debug.log'),
        ]
    )
    
    args = _parse_args(argv)
    if hasattr(args, "season") and args.season:
        args.season = _require_season_format(args.season)
    if args.command == "import":
        _import_games(args)
    elif args.command == "rank":
        _run_rankings(args)
    elif args.command == "matchup":
        _run_matchup(args)
    elif args.command == "schedule":
        _run_schedule(args)
    elif args.command == "market-review":
        raise ValueError(
            "market-review has been retired. Use `betting market-csv` to load CSV market lines "
            "into `market_lines` and rely on the diagnostics/emitted errors tracked in "
            "`market_line_import_errors`. The schedule/daily workbook pipelines already read "
            "from `market_lines`, so no review command is required."
        )
    elif args.command == "market-bets":
        _run_market_bets(args)
    elif args.command == "report":
        _run_report(args)
    elif args.command == "validation-report":
        _run_validation_report(args)
    elif args.command == "backtest":
        _run_backtest(args)
    elif args.command == "tune":
        _run_tuning(args)
    elif args.command == "tune-batch":
        _run_tune_batch(args)
    elif args.command == "tune-model":
        _run_tune_model(args)
    elif args.command == "tune-ensemble":
        _run_tune_ensemble(args)
    elif args.command == "select-ensemble":
        _run_select_ensemble(args)
    elif args.command == "init-ensemble-config":
        _run_init_ensemble_config(args)
    elif args.command == "calibrate":
        _run_calibrate_ensemble(args)
    elif args.command == "activate-tuning":
        _run_activate_tuning(args)
    elif args.command == "tuning-status":
        _run_tuning_status(args)
    elif args.command == "show-active-params":
        _run_show_active_params(args)
    elif args.command == "bootstrap-market-actives":
        _run_bootstrap_market_actives(args)
    elif args.command == "betting":
        _run_betting(args)
    elif args.command == "review-generate":
        _run_review_generate(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


def _resolve_db_path(args: argparse.Namespace) -> str | None:
    from data.paths import db_path_for

    if hasattr(args, "db") and args.db:
        return args.db
    if hasattr(args, "sport") and hasattr(args, "season"):
        if args.season:
            args.season = _require_season_format(args.season)
        return db_path_for(args.sport, args.season)
    return None


def _run_review_generate(args: argparse.Namespace) -> None:
    from src.data import betting_repository as br
    from src.pipelines import review_runs as rr
    from src.pipelines import opportunities as opportunities_pipeline

    db_path = _resolve_db_path(args)
    if db_path is None:
        raise ValueError("DB path could not be resolved; pass --db or --sport/--season")
    _echo_db_path(Path(db_path))

    sport = args.sport
    season = args.season
    model = args.model
    review_id = getattr(args, "review_run_id", None)
    snapshot_run_id = getattr(args, "snapshot_run_id", None)
    snapshot_date = getattr(args, "snapshot_date", None)
    output_dir = getattr(args, "output_dir", None)
    use_formulas = bool(getattr(args, "formula_workbook", False))
    include_ocr_raw = bool(getattr(args, "include_ocr_raw", True))
    if snapshot_run_id is None:
        raise ValueError("review-generate requires --snapshot-run-id")
    _require_snapshot_run_id_format(snapshot_run_id)
    print(f"snapshot_run_id: {snapshot_run_id}")
    if snapshot_date:
        snapshot_date = _require_iso_date(snapshot_date, field="--snapshot-date")
    review_run_id = br.create_review_run(
        db_path,
        sport=sport,
        season=season,
        model=model,
        notes=None,
        id=review_id,
    )
    summary = opportunities_pipeline.build_opportunities(
        db_path,
        review_run_id=review_run_id,
        sport=sport,
        season=season,
        model=model,
        snapshot_run_id=snapshot_run_id,
        snapshot_date=snapshot_date,
    )
    if use_formulas:
        path = rr.build_review_workbook_with_formulas(
            db_path,
            review_run_id=review_run_id,
            sport=sport,
            season=season,
            output_path=output_dir,
            include_ocr_raw=include_ocr_raw,
        )
    else:
        path = rr.build_review_workbook(
            db_path,
            review_run_id=review_run_id,
            sport=sport,
            season=season,
            output_path=output_dir,
            include_ocr_raw=include_ocr_raw,
        )
    print(
        "Review workbook -> "
        f"{path} (review_run_id={review_run_id}, opportunities={summary.get('inserted')})"
    )


def _run_betting(args: argparse.Namespace) -> None:
    from src.cli import betting as betting_cli
    from src.pipelines import market_ocr as market_ocr_pipeline
    from src.pipelines import daily_workbook as daily_workbook_pipeline
    from src.data import betting_repository as br
    from src.pipelines import review_runs as rr
    from src.pipelines import bets as bets_pipeline
    from src.pipelines import opportunities as opportunities_pipeline
    import sqlite3

    cmd = getattr(args, "betting_cmd", None)
    
    # Some commands (like parse-export) don't need to resolve db_path upfront
    if cmd == "parse-export":
        # Handle parse-export without early db_path resolution
        from src.parsers import betting_app
        
        csv_path = getattr(args, "csv_path", None)
        if not csv_path:
            raise ValueError("--csv is required for parse-export")
        
        sport = getattr(args, "sport", None)
        season = getattr(args, "season", None)
        output = getattr(args, "output", None)
        
        # Handle DB path
        db_override = getattr(args, "db", None)
        if db_override:
            db_base = db_override
        elif season:
            db_base = f"data/db/<sport>/{season}.db"
        else:
            db_base = "data/db/<sport>"
        
        print(f"Parsing betting app export: {csv_path}")
        
        if sport:
            # Single sport mode
            matched, total, unmatched = betting_app.parse_betting_app_export(
                csv_path=csv_path,
                db_path=db_base,
                output_path=output,
                sport=sport,
                season=season,
            )
            print(f"[OK] Matched {matched}/{total} bets")
            if unmatched:
                print(f"[WARNING] {len(unmatched)} unmatched games:")
                for game in unmatched[:10]:
                    print(f"    {game}")
                if len(unmatched) > 10:
                    print(f"    ... and {len(unmatched) - 10} more")
        else:
            # Auto-detect sports
            results = betting_app.parse_betting_app_exports_by_sport(
                csv_path=csv_path,
                db_path=db_base,
                output_dir=output,
                season=season,
            )
            print("\nResults by sport:")
            for sport_code, stats in results.items():
                matched = stats['matched']
                total = stats['total']
                unmatched_count = len(stats['unmatched'])
                print(f"  {sport_code.upper()}: {matched}/{total} matched")
                if stats['unmatched']:
                    print(f"    [WARNING] {unmatched_count} unmatched")
        return

    # Resolve DB for other commands (skip for import-csv which infers from game_id)
    if cmd != "import-csv":
        db_path = _resolve_db_path(args)
    else:
        db_path = None

    if cmd == "market-ocr":
        images = args.images
        book = getattr(args, "book", None)
        captured_at = getattr(args, "captured_at", None)
        json_out = getattr(args, "json_output", None)
        if db_path is None and not json_out:
            raise ValueError("DB path could not be resolved; pass --db or --sport/--season, or use --json-output")
        # Interactive confirmation: if we're going to write to a DB (non-JSON mode),
        # ask the user to confirm before performing writes.
        if not json_out and db_path is not None:
            _echo_db_path(Path(db_path))
            try:
                images_list = market_ocr_pipeline._gather_images([images])
                img_count = len(images_list)
            except Exception:
                img_count = 1
            resp = input(f"About to ingest {img_count} image(s) into DB {db_path}. Proceed? [y/N]: ").strip().lower()
            if resp != "y":
                print("market-ocr: aborted by user (no changes made)")
                return

        created = market_ocr_pipeline.ingest_screenshots([images], db_path=db_path, sport=args.sport, season=args.season, source=book or "screenshot", book=book, captured_at=captured_at, json_output=json_out)
        print(f"market-ocr: ingested {created} staging rows")

    elif cmd == "market-commit":
        raise ValueError(
            "market-commit is no longer supported. Market ingestion now writes directly into `market_lines` "
            "via `betting market-csv`; the import command handles unmatched diagnostics and the schedule "
            "workbooks read the latest odds from `market_lines`."
        )

    elif cmd == "market-csv":
        if db_path is None:
            raise ValueError("DB path could not be resolved; pass --db or --sport/--season")
        _echo_db_path(Path(db_path))
        result = br.import_market_csv(
            db_path,
            csv_path=args.csv_path,
            sport=args.sport,
            season=args.season,
            default_book=getattr(args, "default_book", None),
            date_filter=getattr(args, "date_filter", None),
        )
        rows_loaded = result.get("rows_loaded", 0)
        inserted = result.get("inserted", 0)
        unmatched = result.get("unmatched", 0)
        filtered = result.get("date_filtered", 0)
        print(
            "market-csv: "
            f"rows_loaded={rows_loaded} "
            f"inserted={inserted} "
            f"unmatched={unmatched} "
            f"date_filtered={filtered}"
        )
        if unmatched:
            unmatched_reasons = result.get("unmatched_reasons", {})
            if unmatched_reasons:
                print("market-csv: unmatched reasons:")
                for reason, count in sorted(unmatched_reasons.items()):
                    print(f"  {reason}={count}")
            examples = result.get("unmatched_examples", [])
            if examples:
                print("market-csv: unmatched examples (up to 10):")
                for ex in examples:
                    print(
                        "  "
                        f"row={ex.get('row_index')} "
                        f"reason={ex.get('reason')} "
                        f"teams={ex.get('team_home_raw')} vs {ex.get('team_away_raw')} "
                        f"date={ex.get('game_date')} "
                        f"market={ex.get('market_type')} "
                        f"selection={ex.get('selection')} "
                        f"line={ex.get('line')} "
                        f"odds={ex.get('odds')} "
                        f"details={ex.get('details')}"
                    )
    elif cmd == "clv-csv":
        if db_path is None:
            raise ValueError("DB path could not be resolved; pass --db or --sport/--season")
        _echo_db_path(Path(db_path))
        result = br.import_clv_csv(
            db_path,
            csv_path=args.csv_path,
            sport=args.sport,
            season=args.season,
            default_market_type=getattr(args, "default_market_type", None),
            default_captured_at=getattr(args, "captured_at", None),
            update_bets=not getattr(args, "no_update_bets", False),
        )
        print(
            "clv-csv: "
            f"snapshots={result.get('snapshots')} "
            f"bets_updated={result.get('bets_updated')} "
            f"rejected={result.get('rejected')}"
        )

    elif cmd == "import-csv":
        csv_path = getattr(args, "csv_path", None)
        if not csv_path:
            raise ValueError("--csv is required for import-csv")
        
        # Try to infer sport and season from game_id column if not provided
        sport = getattr(args, "sport", None)
        season = getattr(args, "season", None)
        
        if not sport or not season:
            # Read CSV to infer sport/season from game_id
            import pandas as pd
            csv_file = Path(csv_path)
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                if 'game_id' in df.columns and len(df) > 0:
                    first_game_id = df['game_id'].dropna().iloc[0] if not df['game_id'].dropna().empty else None
                    if first_game_id:
                        # game_id format: sport:season:date:hash
                        parts = str(first_game_id).split(':')
                        if len(parts) >= 2:
                            if not sport:
                                sport = parts[0]
                            if not season:
                                season = parts[1]
        
        if not sport:
            raise ValueError("--sport is required (or CSV must have game_id column to infer sport)")
        if not season:
            raise ValueError("--season is required (or CSV must have game_id column to infer season)")
        
        # Now resolve db_path with known sport/season
        db_override = getattr(args, "db", None)
        if db_override:
            db_path = db_override
        else:
            db_path = f"data/db/{sport}/{season}.db"
        
        if db_path is None:
            raise ValueError("DB path could not be resolved; pass --db or --sport/--season")
        _echo_db_path(Path(db_path))
        dry_run = bool(getattr(args, "dry_run", False))
        review_run_id = getattr(args, "review_run_id", None)
        
        result = br.import_bets_csv(
            db_path,
            csv_path=csv_path,
            sport=sport,
            season=season,
            review_run_id=review_run_id,
            dry_run=dry_run,
        )
        mode = "dry-run" if dry_run else "import"
        print(
            f"import-csv ({mode}): "
            f"inserted={result.get('inserted')} "
            f"updated={result.get('updated')} "
            f"rejected={result.get('rejected')} "
            f"skipped={result.get('skipped')} "
            f"review_run_id={result.get('review_run_id')}"
        )
        if result.get("errors"):
            print("\nFirst few errors:")
            for err in result.get("errors", []):
                print(f"  {err}")

    elif cmd == "action-import":
        csv_path = getattr(args, "csv", None)
        if not csv_path:
            raise ValueError("--csv is required for action-import")
        if db_path is None:
            raise ValueError("DB path could not be resolved; pass --db or --sport/--season")
        _echo_db_path(Path(db_path))
        from src.pipelines.action_import import import_from_csv

        result = import_from_csv(
            csv_path,
            db_path,
            sport=args.sport,
            season=args.season,
            snapshot_run_id=getattr(args, "snapshot_run_id", None),
            book=getattr(args, "book", None),
        )
        print(f"action-import: inserted={result.get('inserted')} staged={result.get('staged')} rejected={result.get('rejected')}")

    elif cmd == "review-generate":
        _run_review_generate(args)

    elif cmd == "daily-workbook":
        if db_path is None:
            raise ValueError("DB path could not be resolved; pass --db or --sport/--season")
        _echo_db_path(Path(db_path))
        args.date = _require_iso_date(args.date, field="--date")
        snapshot_run_id = getattr(args, "snapshot_run_id", None)
        if snapshot_run_id:
            _require_snapshot_run_id_format(snapshot_run_id)
            print(f"snapshot_run_id: {snapshot_run_id}")
        out = daily_workbook_pipeline.build_daily_workbook(
            db_path,
            sport=args.sport,
            season=args.season,
            date=args.date,
            model=getattr(args, "model", None),
            review_run_id=getattr(args, "review_run_id", None),
            snapshot_run_id=snapshot_run_id,
            output_path=getattr(args, "output", None),
        )
        print(f"Daily workbook -> {out}")

    elif cmd == "log-bets":
        workbook = args.workbook
        dry_run = bool(getattr(args, "dry_run", False))
        writeback = bool(getattr(args, "writeback", False))
        if db_path is None:
            raise ValueError("DB path could not be resolved; pass --db or --sport/--season")
        _echo_db_path(Path(db_path))
        if dry_run:
            print("log_bets dry-run: parsing workbook (no writes)")
            updated = bets_pipeline.log_bets(workbook, review_run_id=None, db_path=db_path, dry_run=True, writeback=False)
            print(f"log_bets dry-run parsed {updated} bet rows")
        else:
            updated = bets_pipeline.log_bets(workbook, review_run_id=None, db_path=db_path, dry_run=False, writeback=writeback)
            print(f"log_bets processed {updated} rows")

    elif cmd == "settle-bets":
        sport = args.sport
        season = args.season
        if db_path is not None:
            _echo_db_path(Path(db_path))
        settled = bets_pipeline.settle_bets(sport=sport, season=season)
        print(f"Settled {settled} bets")

    elif cmd == "report":
        # Generate aggregated betting reports (daily/weekly/monthly)
        rpt_type = getattr(args, "report_type", "daily")
        from src.data import reporting as rpt

        if db_path is None:
            raise ValueError("DB path could not be resolved; pass --db or --sport/--season")
        _echo_db_path(Path(db_path))
        output = getattr(args, "output", None)
        start = getattr(args, "start", None)
        end = getattr(args, "end", None)
        fmt = getattr(args, "format", None)

        if rpt_type == "daily":
            rows = rpt.daily_report(db_path, sport=args.sport, season=args.season)
        elif rpt_type == "weekly":
            rows = rpt.weekly_report(db_path, sport=args.sport, season=args.season, start=start, end=end)
        elif rpt_type == "monthly":
            rows = rpt.monthly_report(db_path, sport=args.sport, season=args.season, start=start, end=end)
        else:
            raise ValueError(f"Unsupported report type: {rpt_type}")

        if output:
            outp = Path(output)
            use_format = fmt or (outp.suffix.lstrip('.').lower() if outp.suffix else 'csv')
            if use_format == "csv":
                rpt.write_report_csv(rows, output)
                print(f"Report written -> {output}")
            elif use_format == "xlsx":
                # write composite workbook with main sheet + edge_buckets + clv
                rpt.write_full_report_xlsx(
                    db_path,
                    sport=args.sport,
                    season=args.season,
                    rpt_type=rpt_type,
                    rows=rows,
                    output_path=output,
                )
                print(f"Report written -> {output}")
            else:
                rpt.write_report_csv(rows, output)
                print(f"Report written -> {output}")
        else:
            for r in rows:
                print(r)

    elif cmd == "validate":
        from src.pipelines import betting_validation as betting_validation_pipeline

        if db_path is None:
            raise ValueError("DB path could not be resolved; pass --db or --sport/--season")
        _echo_db_path(Path(db_path))
        args.date = _require_iso_date(args.date, field="--date")
        snapshot_run_id = getattr(args, "snapshot_run_id", None)
        snapshot_date = getattr(args, "snapshot_date", None)
        if snapshot_date:
            snapshot_date = _require_iso_date(snapshot_date, field="--snapshot-date")
        if snapshot_run_id:
            _require_snapshot_run_id_format(snapshot_run_id)
            print(f"snapshot_run_id: {snapshot_run_id}")
        if snapshot_run_id is None and snapshot_date is None:
            raise ValueError("validate requires --snapshot-run-id or --snapshot-date")

        results = betting_validation_pipeline.run_preflight_validation(
            db_path,
            sport=args.sport,
            season=args.season,
            model=args.model,
            date=args.date,
            snapshot_run_id=snapshot_run_id,
            snapshot_date=snapshot_date,
            min_snapshots=int(getattr(args, "min_snapshots", 1)),
        )

        integrity = results.get("integrity", {})
        predictions = results.get("predictions", {})
        snapshots = results.get("snapshots", {})
        games = results.get("games", {})
        print("Pre-flight validation:")
        print(f"- integrity: {'ok' if integrity.get('ok') else 'fail'} ({integrity.get('detail')})")
        print(f"- games: {games.get('count')} on {args.date}")
        print(f"- predictions: {'ok' if predictions.get('ok') else 'fail'} (count={predictions.get('count')})")
        print(
            "- market_snapshots: "
            f"{'ok' if snapshots.get('ok') else 'fail'} "
            f"(count={snapshots.get('count')}, min_required={snapshots.get('min_required')})"
        )

        failures: list[str] = []
        if not integrity.get("ok"):
            failures.append(f"DB integrity_check failed: {integrity.get('detail')}")
        if not predictions.get("ok"):
            failures.append("No predictions found for the requested date/model")
        if not snapshots.get("ok"):
            failures.append("Insufficient market snapshots for the requested filter")
        if failures:
            raise ValueError("; ".join(failures))

    else:
        raise ValueError(f"Unknown betting command: {cmd}")


def _require_iso_date(value: str, *, field: str) -> str:
    from datetime import date as dt_date

    normalized = _normalize_date_str(value, field=field)
    try:
        dt_date.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field} must be YYYY-MM-DD: {value}") from exc
    if normalized != value:
        print(f"Normalized {field} to {normalized}")
    return normalized


def _optional_iso_date(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    return _require_iso_date(value, field=field)


def _run_activate_tuning(args: argparse.Namespace) -> None:
    from data.repository import (
        load_model_market_tuning_run_by_run_id,
        load_best_model_market_tuning_params_by_optimized_metric,
    )
    from pipelines.model_params import activate_best_params
    from pipelines.market_tuning import _resolve_market_metric

    db_path = _resolve_db_path(args)
    if db_path is None:
        raise ValueError("DB path could not be resolved; pass --db or --sport/--season")
    _echo_db_path(Path(db_path))

    sport = args.sport
    season = args.season
    model = args.model
    market = args.market.strip().upper()
    run_id = getattr(args, "run_id", None)
    metric = getattr(args, "metric", None)

    params = None
    source_run_id = None
    metric_optimized = None
    best_score = None
    if run_id:
        params, source_run_id, metric_optimized, best_score = load_model_market_tuning_run_by_run_id(
            db_path, run_id=run_id
        )
        if params is None:
            raise ValueError(f"No tuning run found with run_id={run_id}")
    else:
        # resolve optimized metric label and get best run
        metric_name, metric_optimized = _resolve_market_metric(market, metric)
        params, source_run_id = load_best_model_market_tuning_params_by_optimized_metric(
            db_path, sport=sport, season=season, model=model, market=market, metric_optimized=metric_optimized
        )
        if params is None:
            raise ValueError(f"No tuning runs found for model={model} market={market} metric={metric_name}")
        _, _, metric_from_run, best_score = load_model_market_tuning_run_by_run_id(db_path, run_id=source_run_id)
        metric_optimized = metric_optimized or metric_from_run

    activate_best_params(
        db_path=db_path,
        sport=sport,
        season=season,
        model=model,
        market=market,
        run_id=source_run_id,
        best_params=params,
        best_score=best_score,
        metric_optimized=metric_optimized,
    )
    print(
        f"Activated tuning run -> model={model} market={market} run_id={source_run_id} metric={metric_optimized}"
    )


def _run_tuning_status(args: argparse.Namespace) -> None:
    """Read-only status of tuned/active params and ensemble weights.

    Prints per-market per-model whether active params exist, were auto-selected
    from tuning runs, or defaults will be used. Also prints ensemble weight status.
    """
    _ensure_src_on_path()
    from data.paths import db_path_for
    from pipelines.model_params import (
        resolve_active_model_market_params,
        resolve_active_ensemble_weights,
    )
    from models.registry import list_models
    from ensemble.config import load_ensemble_config
    from markets.base import Market

    db_path = Path(args.db) if getattr(args, "db", None) else db_path_for(args.sport, args.season)
    _echo_db_path(db_path)

    requested_models = None
    if getattr(args, "models", None):
        requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    models = requested_models if requested_models else list_models()

    print("Model parameter sources by market:")
    for market in (Market.ML, Market.SPREAD, Market.TOTAL):
        print(f"  Market: {market.name}")
        for model in models:
            resolved = resolve_active_model_market_params(
                db_path=db_path, sport=args.sport, season=args.season, model=model, market=market.name
            )
            status = "DEFAULT"
            details = ""
            if resolved.params is not None:
                    label = resolved.params_source
                    if label in {"tuned_active", "db_market_active"}:
                        status = "ACTIVE"
                        details = f"(source_run_id={resolved.source_run_id})"
                    elif label == "legacy_active":
                        status = "LEGACY_ACTIVE"
                        details = f"(source_run_id={resolved.source_run_id})"
                    elif label == "default_active":
                        status = "DEFAULT_ACTIVE"
                        details = f"(source_run_id={resolved.source_run_id})"
                    elif label == "missing_active":
                        status = "MISSING"
                    else:
                        status = label.upper()
            print(f"    {model}: {status} {details}")

    # Ensembles
    try:
        ensemble_cfg = load_ensemble_config(args.sport, args.season)
    except Exception:
        ensemble_cfg = {}
    if ensemble_cfg:
        print("\nEnsemble weights status:")
        for market in (Market.ML, Market.SPREAD, Market.TOTAL):
            eid = ensemble_cfg.get(market.name, {}).get("ensemble_id") if isinstance(ensemble_cfg.get(market.name), dict) else None
            if not eid:
                # try simple mapping
                eid = ensemble_cfg.get(market.name)
            if not eid:
                continue
            resolved = resolve_active_ensemble_weights(db_path=db_path, sport=args.sport, season=args.season, market=market.name, ensemble_id=eid)
            status = "DEFAULT"
            details = ""
            if resolved.params is not None:
                if resolved.params_source == "db_ensemble_active":
                    status = "ACTIVE"
                    details = f"(source_run_id={resolved.source_run_id})"
                elif resolved.params_source == "db_ensemble_best_run":
                    status = "AUTO-SELECT"
                    details = f"(metric={resolved.tuned_metric_used} run_id={resolved.source_run_id})"
                else:
                    status = resolved.params_source.upper()
            print(f"  {market.name} -> ensemble={eid}: {status} {details}")
    else:
        print("\nNo ensemble config found for this sport/season.")


def _run_show_active_params(args: argparse.Namespace) -> None:
    _ensure_src_on_path()
    from data.paths import db_path_for
    from markets.base import Market
    from models.registry import list_models
    from pipelines.model_params import resolve_effective_params

    db_path = Path(args.db) if getattr(args, "db", None) else db_path_for(args.sport, args.season)
    _echo_db_path(db_path)

    requested_models = None
    if getattr(args, "models", None):
        requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    models = requested_models if requested_models else list_models()

    header = f"{'model':<16} {'market':<6} {'source_label':<18} {'run_id':<28} {'metric':<12} {'best_score':<12} {'fingerprint':<12} {'keys':>4}"
    print(header)
    print("-" * len(header))
    for model in models:
        for market in (Market.ML, Market.SPREAD, Market.TOTAL):
            resolved = resolve_effective_params(
                db_path=db_path,
                sport=args.sport,
                season=args.season,
                model=model,
                market=market.name,
            )
            metric_display = resolved.metric_optimized or ""
            if metric_display.startswith("backtest_"):
                metric_display = metric_display.replace("backtest_", "", 1)
            best_score_display = "" if resolved.best_score is None else f"{resolved.best_score:.6g}"
            fingerprint = resolved.params_fingerprint or ""
            run_id = resolved.source_run_id or ""
            print(
                f"{model:<16} {market.name:<6} {resolved.params_source_label:<18} {run_id:<28} {metric_display:<12} "
                f"{best_score_display:<12} {fingerprint[:12]:<12} {len(resolved.params or {}):>4}"
            )


def _run_bootstrap_market_actives(args: argparse.Namespace) -> None:
    from pipelines.model_params import bootstrap_market_active_params
    from models.registry import list_models, normalize_model_name

    db_path = _resolve_db_path(args)
    if db_path is None:
        raise ValueError("DB path could not be resolved; pass --db or --sport/--season")
    _echo_db_path(Path(db_path))

    model_arg = str(args.model).strip()
    if model_arg and model_arg.lower() not in {"all", "*"}:
        models = [normalize_model_name(model_arg)]
    else:
        models = list_models()

    summary = bootstrap_market_active_params(
        db_path=db_path,
        sport=args.sport,
        season=args.season,
        models=models,
        include_ml=not bool(getattr(args, "no_ml", False)),
    )
    counts = summary.get("counts", {})
    print("Bootstrap market actives summary:")
    print(f"- created from best runs: {counts.get('created_from_best_run', 0)}")
    print(f"- created from model metrics: {counts.get('created_from_model_metric', 0)}")
    print(f"- created defaults: {counts.get('created_default', 0)}")
    print(f"- skipped existing: {counts.get('skipped_existing', 0)}")

    for label, pairs in (
        ("created from best runs", summary.get("created_from_best_run", [])),
        ("created from model metrics", summary.get("created_from_model_metric", [])),
        ("created defaults", summary.get("created_default", [])),
    ):
        if not pairs:
            continue
        formatted = []
        for model_name, market_name, source in pairs:
            suffix = f" ({source})" if source else ""
            formatted.append(f"{model_name}/{market_name}{suffix}")
        print(f"{label}: {', '.join(formatted)}")


def _normalize_date_str(value: str, *, field: str) -> str:
    if re.match(r"^\d{8}$", value):
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        return value
    raise ValueError(f"{field} must be YYYY-MM-DD or YYYYMMDD: {value}")


def _require_season_format(season: str) -> str:
    if not re.match(r"^\d{4}-\d{2}$", season):
        raise ValueError(
            f"--season must be in YYYY-YY format (e.g., 2025-26): {season}"
        )
    start_year = int(season[:4])
    end_year = int(season[-2:])
    expected_end = (start_year + 1) % 100
    if end_year != expected_end:
        raise ValueError(
            f"--season end year must be {expected_end:02d} for {start_year}: {season}"
        )
    return season


def _require_snapshot_run_id_format(value: str) -> str:
    if not (value.startswith("snap-") or value.startswith("schedule-")):
        raise ValueError(
            "snapshot_run_id must start with 'snap-' or 'schedule-' "
            f"(got {value!r})"
        )
    return value


def _echo_db_path(db_path: Path) -> None:
    print(f"DB: {db_path}")


def _parse_json_arg(raw: str | None) -> dict | None:
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for --model-params: {raw}") from exc
    if not isinstance(data, dict):
        raise ValueError("--model-params must be a JSON object.")
    return data


def _parse_markets_arg(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    markets = [item.strip() for item in raw.split(",") if item.strip()]
    if not markets:
        raise ValueError("Provide at least one market identifier via --market.")
    return markets


def _parse_market_metrics_arg(raw: str | None) -> dict[str, str] | None:
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for --market-metrics: {raw}") from exc
    if not isinstance(payload, dict):
        raise ValueError("--market-metrics must be a JSON object.")
    cleaned: dict[str, str] = {}
    for market_key, metric_value in payload.items():
        if not isinstance(market_key, str):
            raise ValueError("Market keys in --market-metrics must be strings.")
        if not isinstance(metric_value, str):
            raise ValueError("Metric values in --market-metrics must be strings.")
        key = market_key.strip()
        metric = metric_value.strip()
        if not key:
            raise ValueError("Market keys in --market-metrics cannot be blank.")
        if not metric:
            raise ValueError("Metric values in --market-metrics cannot be blank.")
        cleaned[key] = metric
    return cleaned


def _load_grid_override(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Grid file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("Grid file must contain a JSON object.")
    return data


if __name__ == "__main__":
    main()
