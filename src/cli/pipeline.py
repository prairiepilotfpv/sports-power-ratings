"""Command-line pipeline for ingesting, ranking, and projecting sports results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _ensure_src_on_path() -> None:
    src_dir = Path(__file__).resolve().parents[2] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _parse_args() -> argparse.Namespace:
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
        "--db",
        help=f"Optional SQLite DB path override (default: {db_dir()}/<sport>/<season>.db)",
    )

    report_parser = subparsers.add_parser(
        "report",
        aliases=["excel", "excel_report"],
        help="Generate an Excel report with rankings per model.",
    )
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

    backtest_parser = subparsers.add_parser(
        "backtest",
        aliases=["bt"],
        help="Run a model backtest on historical games.",
    )
    backtest_parser.add_argument(
        "--model",
        default="bradley_terry_hfa",
        help="Backtest model to run (default: bradley_terry_hfa).",
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
        choices=["log_loss", "brier_score", "mae_margin"],
        help="Metric to optimize (default: log_loss).",
    )
    tune_parser.add_argument(
        "--output-dir",
        help="Optional output directory override (default: outputs/tuning/<model>).",
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
        "--allow-worse",
        action="store_true",
        help="Allow worse results than the default parameters (disables improvement guard).",
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

    return parser.parse_args()


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
        )
    db_path = Path(args.db) if args.db else db_path_for(args.sport, args.season)
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
        )
        line, metrics = format_matchup(prediction)
        print(f"[{model}] {line}")
        print(f"[{model}] Spread: {metrics['spread']:.2f}")
        print(f"[{model}] Total Points: {metrics['total_points']:.2f}")


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
    output_path = Path(args.output) if args.output else None
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
    )
    print(f"Saved schedule workbook -> {result_path}")


def _run_report(args: argparse.Namespace) -> None:
    """Build an Excel report with one sheet per model."""
    _ensure_src_on_path()
    from data.paths import db_path_for
    from pipelines.excel_report import build_excel_report

    model_params = _parse_json_arg(args.model_params)
    db_path = Path(args.db) if args.db else db_path_for(args.sport, args.season)
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
    )
    print(f"Saved backtest outputs to {outputs.output_dir}")


def _run_tuning(args: argparse.Namespace) -> None:
    """Run hyperparameter tuning via backtest grid search."""
    _ensure_src_on_path()
    from pipelines.tuning import run_tuning_pipeline
    from data.paths import db_path_for

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
    outputs = run_tuning_pipeline(
        csv_path=Path(args.csv),
        model=args.model,
        start_date=args.start,
        end_date=args.end,
        window=args.window,
        rolling_days=args.rolling_days,
        rolling_games=args.rolling_games,
        metric=args.metric,
        output_dir=output_dir,
        grid_override=grid_override,
        apply_best=args.apply_best,
        require_improvement=not args.allow_worse,
        db_path=db_path,
        sport=args.sport,
        season=args.season,
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


def main() -> None:
    """CLI entry point: route subcommands to their handlers."""
    args = _parse_args()
    if args.command == "import":
        _import_games(args)
    elif args.command == "rank":
        _run_rankings(args)
    elif args.command == "matchup":
        _run_matchup(args)
    elif args.command == "schedule":
        _run_schedule(args)
    elif args.command == "report":
        _run_report(args)
    elif args.command == "backtest":
        _run_backtest(args)
    elif args.command == "tune":
        _run_tuning(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


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


if __name__ == "__main__":
    main()
