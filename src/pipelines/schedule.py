from __future__ import annotations

"""Schedule export pipeline with projection fields."""

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

try:  # Allow execution from repository root or nested directories
    from bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - fallback when bootstrap isn't on sys.path
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from bootstrap import ensure_src_on_path

ensure_src_on_path()

import pandas as pd

from data.repository import load_games, load_model_metrics
from pipelines.common import normalize_games
from pipelines.matchups import team_home_advantages
from config import DEFAULT_WIN_PROB_K
from pipelines.projections import (
    average_total_points,
    fit_total_model,
    matchup_total_from_averages,
    project_game,
    total_from_ratings,
    team_scoring_averages,
)
from models.registry import (
    get_model,
    get_model_abbreviation,
    list_models,
    normalize_model_name,
)
from pipelines.run_rankings import build_rankings
from models.base import resolve_model_identity
from pipelines.metadata import prediction_hash


SCHEDULE_EXPORT_COLUMNS: List[str] = [
    "date",
    "game_id",
    "status",
    "home_team",
    "away_team",
    "neutral",
    "overtime",
    "home_score",
    "away_score",
    "result_margin",
    "result_total",
    "home_rating",
    "away_rating",
    "home_advantage",
    "projected_winner",
    "projected_spread",
    "projected_home_spread",
    "projected_win_prob",
    "projected_home_score",
    "projected_away_score",
    "projected_total",
]

DASHBOARD_COLUMNS: List[str] = [
    "model",
    "date",
    "game",
    "projected_home_score",
    "projected_away_score",
    "projected_total",
    "projected_winner",
    "projected_spread",
]

MODEL_METADATA_DATA_START_ROW = 10


def _completed_games(df: pd.DataFrame) -> pd.DataFrame:
    """Return only games with recorded scores."""
    if df.empty:
        return df
    mask = df["home_score"].notna() & df["away_score"].notna()
    return df[mask]


def _upcoming_games(df: pd.DataFrame) -> pd.DataFrame:
    """Return only games without scores (future or incomplete)."""
    if df.empty:
        return df
    mask = df["home_score"].isna() | df["away_score"].isna()
    return df[mask]


def _rating_lookup(rankings: pd.DataFrame) -> Dict[str, float]:
    """Build a lookup for team name -> point-scale rating."""
    return {str(row["team"]).strip(): float(row["points"]) for _, row in rankings.iterrows()}


def _safe_date(value: Any) -> str:
    """Convert date-like values to an ISO date string."""
    if value is None:
        return ""
    try:
        return pd.to_datetime(value).date().isoformat()
    except Exception:
        return str(value)


def _base_schedule_row(row: pd.Series) -> Dict[str, Any]:
    """Normalize the base schedule fields shared by played and upcoming games."""
    neutral_raw = row.get("neutral", False)
    overtime_raw = row.get("overtime", False)
    neutral = False if pd.isna(neutral_raw) else bool(neutral_raw)
    overtime = False if pd.isna(overtime_raw) else bool(overtime_raw)

    return {
        "date": _safe_date(row.get("date")),
        "home_team": str(row.get("home_team", "")).strip(),
        "away_team": str(row.get("away_team", "")).strip(),
        "home_score": row.get("home_score"),
        "away_score": row.get("away_score"),
        "neutral": neutral,
        "overtime": overtime,
        "game_id": row.get("game_id"),
    }


def _project_row(
    row: pd.Series,
    *,
    ratings: Dict[str, float],
    base_total: float,
    scoring_averages: Dict[str, tuple[float, float]],
    status: str,
    home_advantage: float,
    win_prob_k: float,
    total_intercept: float,
    total_slope: float,
) -> Dict[str, Any]:
    """Create a schedule export row with projections when ratings are available."""
    base = _base_schedule_row(row)
    home = base["home_team"]
    away = base["away_team"]
    applied_home_advantage = 0.0 if base["neutral"] else home_advantage
    home_rating = ratings.get(home)
    away_rating = ratings.get(away)

    projected_winner = None
    projected_spread = None
    projected_home_spread = None
    projected_win_prob = None
    projected_home_score = None
    projected_away_score = None
    projected_total = None

    if home_rating is not None and away_rating is not None:
        # Build projected spreads/totals when both team ratings are available.
        matchup_total = matchup_total_from_averages(home, away, scoring_averages)
        model_total = total_from_ratings(
            home,
            away,
            ratings,
            intercept=total_intercept,
            slope=total_slope,
        )
        applied_total = model_total or matchup_total or (base_total if base_total > 0 else None)
        projection = project_game(
            home_rating,
            away_rating,
            home_advantage=applied_home_advantage,
            neutral=base["neutral"],
            k=win_prob_k,
            base_total=applied_total,
            home_team=home,
            away_team=away,
        )
        projected_winner = projection.projected_winner
        projected_spread = projection.projected_spread
        projected_home_spread = projection.projected_home_spread
        projected_win_prob = projection.projected_win_prob
        projected_home_score = projection.projected_home_score
        projected_away_score = projection.projected_away_score
        projected_total = projection.projected_total

    result_margin = None
    result_total = None
    if base["home_score"] is not None and base["away_score"] is not None:
        try:
            result_margin = int(base["home_score"]) - int(base["away_score"])
            result_total = int(base["home_score"]) + int(base["away_score"])
        except Exception:
            result_margin = None
            result_total = None

    base.update(
        {
            "status": status,
            "home_rating": home_rating,
            "away_rating": away_rating,
            "projected_winner": projected_winner,
            "projected_spread": projected_spread,
            "projected_home_spread": projected_home_spread,
            "projected_win_prob": projected_win_prob,
            "home_advantage": applied_home_advantage,
            "projected_home_score": projected_home_score,
            "projected_away_score": projected_away_score,
            "projected_total": projected_total,
            "result_margin": result_margin,
            "result_total": result_total,
        }
    )
    return base


def _order_schedule_export(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a consistent column order for downstream reporting."""
    expected_set = set(SCHEDULE_EXPORT_COLUMNS)
    missing = [col for col in SCHEDULE_EXPORT_COLUMNS if col not in schedule_df.columns]
    extra = [col for col in schedule_df.columns if col not in expected_set]
    if missing or extra:
        raise ValueError(f"Schedule export column mismatch. Missing: {missing or 'none'}, extra: {extra or 'none'}")
    # report-only ordering; do not modify upstream calculations.
    return schedule_df[SCHEDULE_EXPORT_COLUMNS]


def _resolve_models(model: str | None) -> list[str]:
    if model is None:
        return list_models()
    return [normalize_model_name(model)]


def _resolve_output_path(
    output_path: str | Path | None,
    *,
    sport: str,
    season: str,
    default_name: str,
    model: str,
    add_prefix: bool,
) -> Path:
    if output_path is None:
        resolved = Path("data/processed") / sport / season / default_name
    else:
        resolved = Path(output_path)
        if resolved.is_dir() or resolved.suffix == "":
            resolved = resolved / default_name
    if add_prefix:
        abbrev = get_model_abbreviation(model)
        resolved = resolved.with_name(f"{abbrev}_{resolved.name}")
    return resolved


def _resolve_workbook_path(
    output_path: str | Path | None,
    *,
    sport: str,
    season: str,
    default_name: str,
) -> Path:
    if output_path is None:
        resolved = Path("data/processed") / sport / season / default_name
    else:
        resolved = Path(output_path)
        if resolved.is_dir() or resolved.suffix == "":
            resolved = resolved / default_name
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _build_schedule_dataframe(
    df: pd.DataFrame,
    *,
    db_path: str | Path,
    sport: str,
    season: str,
    model: str,
    upcoming_only: bool,
) -> pd.DataFrame:
    played = _completed_games(df)
    upcoming = _upcoming_games(df)

    rankings = build_rankings(played, model=model, require_scores=False)
    ratings = _rating_lookup(rankings)
    fallback_total = average_total_points(played.to_dict(orient="records"))
    metrics = load_model_metrics(db_path, sport=sport, season=season, model=model) or {}
    # Team-specific home advantages (per-home team residuals) are the chosen H option.
    home_advantages = team_home_advantages(played, ratings)
    fallback_home_advantage = float(metrics.get("home_advantage", 0.0))
    win_prob_k = float(metrics.get("win_prob_k", DEFAULT_WIN_PROB_K))
    base_total = float(metrics.get("base_total", 0.0)) or fallback_total
    played_records = played.to_dict(orient="records")
    total_intercept, total_slope = fit_total_model(played_records, ratings)
    scoring_averages = team_scoring_averages(played_records)

    schedule_rows: List[Dict[str, Any]] = []
    if not upcoming_only:
        for _, row in played.iterrows():
            home = str(row.get("home_team", "")).strip()
            schedule_rows.append(
                _project_row(
                    row,
                    ratings=ratings,
                    base_total=base_total,
                    scoring_averages=scoring_averages,
                    status="final",
                    home_advantage=home_advantages.get(home, fallback_home_advantage),
                    win_prob_k=win_prob_k,
                    total_intercept=total_intercept,
                    total_slope=total_slope,
                )
            )

    for _, row in upcoming.iterrows():
        home = str(row.get("home_team", "")).strip()
        schedule_rows.append(
            _project_row(
                row,
                ratings=ratings,
                base_total=base_total,
                scoring_averages=scoring_averages,
                status="scheduled",
                home_advantage=home_advantages.get(home, fallback_home_advantage),
                win_prob_k=win_prob_k,
                total_intercept=total_intercept,
                total_slope=total_slope,
            )
        )

    schedule_df = pd.DataFrame(schedule_rows)
    if not schedule_df.empty and "date" in schedule_df.columns:
        schedule_df = schedule_df.assign(_dt=pd.to_datetime(schedule_df["date"], errors="coerce")).sort_values(
            ["_dt", "game_id", "away_team", "home_team"]
        ).drop(columns=["_dt"], errors="ignore")

    schedule_df = _order_schedule_export(schedule_df)
    return schedule_df


def _format_game_name(away_team: Any, home_team: Any) -> str:
    """Render a simple matchup label."""
    away = str(away_team or "").strip()
    home = str(home_team or "").strip()
    if away and home:
        return f"{away} @ {home}"
    return away or home


def _dashboard_rows_for_today(schedule_df: pd.DataFrame, model_name: str, as_of_date: date | None = None) -> list[Dict[str, Any]]:
    """Collect scheduled games for the current day for the dashboard sheet."""
    if schedule_df.empty:
        return []

    today = as_of_date or pd.Timestamp.today().date()
    df = schedule_df.assign(_date=pd.to_datetime(schedule_df["date"], errors="coerce").dt.date)
    df = df[(df["status"] == "scheduled") & (df["_date"] == today)]
    if df.empty:
        return []

    rows: list[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rows.append(
            {
                "model": model_name,
                "date": row.get("date"),
                "game": _format_game_name(row.get("away_team"), row.get("home_team")),
                "projected_home_score": row.get("projected_home_score"),
                "projected_away_score": row.get("projected_away_score"),
                "projected_total": row.get("projected_total"),
                "projected_winner": row.get("projected_winner"),
                "projected_spread": row.get("projected_spread"),
            }
        )
    return rows


def _training_date_range(df: pd.DataFrame) -> str:
    if df.empty or "date" not in df.columns:
        return ""
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        return ""
    return f"{dates.min().date().isoformat()} to {dates.max().date().isoformat()}"


def _serialize_params(params: Any) -> str:
    try:
        return json.dumps(params, sort_keys=True, default=str)
    except TypeError:
        return str(params)


def _build_model_metadata(
    *,
    model_name: str,
    played: pd.DataFrame,
    schedule_df: pd.DataFrame,
) -> dict[str, Any]:
    model_instance = get_model(model_name)()
    identity = resolve_model_identity(model_instance)
    metadata = {
        "model_id": identity["model_id"],
        "model_version": identity["model_version"],
        "params": _serialize_params(identity["params"]),
        "trained_on_date_range": _training_date_range(played),
        "n_games_train": int(len(played)),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_hash": prediction_hash(schedule_df, SCHEDULE_EXPORT_COLUMNS),
    }
    return metadata


def _write_metadata_section(
    writer: pd.ExcelWriter,
    sheet_name: str,
    metadata: dict[str, Any],
) -> int:
    if sheet_name in writer.sheets:
        ws = writer.sheets[sheet_name]
    else:
        ws = writer.book.create_sheet(sheet_name)
        writer.sheets[sheet_name] = ws

    ws.cell(row=1, column=1, value="metadata_key")
    ws.cell(row=1, column=2, value="metadata_value")

    row = 2
    for key, value in metadata.items():
        ws.cell(row=row, column=1, value=key)
        ws.cell(row=row, column=2, value=value)
        row += 1

    return MODEL_METADATA_DATA_START_ROW - 1


def _build_schedule_for_model(
    df: pd.DataFrame,
    *,
    db_path: str | Path,
    sport: str,
    season: str,
    model: str,
    output_path: str | Path | None,
    upcoming_only: bool,
    add_prefix: bool,
) -> Path:
    schedule_df = _build_schedule_dataframe(
        df,
        db_path=db_path,
        sport=sport,
        season=season,
        model=model,
        upcoming_only=upcoming_only,
    )
    resolved_output = _resolve_output_path(
        output_path,
        sport=sport,
        season=season,
        default_name="schedule_with_projections.csv",
        model=model,
        add_prefix=add_prefix,
    )
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    schedule_df.to_csv(resolved_output, index=False)
    return resolved_output


def build_schedule_with_projections(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str | None = None,
    output_path: str | Path | None = None,
    upcoming_only: bool = False,
) -> Path | list[Path]:
    """Build a schedule export containing projections for upcoming games."""
    rows = load_games(db_path, sport=sport, season=season)
    df = normalize_games(rows)
    if df.empty:
        raise ValueError(f"No games found for sport={sport!r}, season={season!r}")

    models = _resolve_models(model)
    multiple = len(models) > 1
    results = [
        _build_schedule_for_model(
            df.copy(deep=True),
            db_path=db_path,
            sport=sport,
            season=season,
            model=model_name,
            output_path=output_path,
            upcoming_only=upcoming_only,
            add_prefix=multiple,
        )
        for model_name in models
    ]
    return results[0] if len(results) == 1 else results


def build_schedule_excel_report(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str | None = None,
    output_path: str | Path | None = None,
    upcoming_only: bool = False,
) -> Path:
    """Build an Excel workbook with schedule projections (one sheet per model)."""
    rows = load_games(db_path, sport=sport, season=season)
    df = normalize_games(rows)
    if df.empty:
        raise ValueError(f"No games found for sport={sport!r}, season={season!r}")

    models = _resolve_models(model)
    played = _completed_games(df)
    report_path = _resolve_workbook_path(
        output_path,
        sport=sport,
        season=season,
        default_name="schedule_with_projections.xlsx",
    )
    dashboard_rows: list[Dict[str, Any]] = []
    with pd.ExcelWriter(report_path) as writer:
        for model_name in models:
            model_df = df.copy(deep=True)
            schedule_df = _build_schedule_dataframe(
                model_df,
                db_path=db_path,
                sport=sport,
                season=season,
                model=model_name,
                upcoming_only=upcoming_only,
            )
            metadata = _build_model_metadata(
                model_name=model_name,
                played=_completed_games(model_df),
                schedule_df=schedule_df,
            )
            start_row = _write_metadata_section(writer, model_name, metadata)
            schedule_df.to_excel(
                writer,
                sheet_name=model_name,
                index=False,
                startrow=start_row,
            )
            dashboard_rows.extend(_dashboard_rows_for_today(schedule_df, model_name))

        dashboard_df = pd.DataFrame(dashboard_rows, columns=DASHBOARD_COLUMNS)
        dashboard_df.to_excel(writer, sheet_name="dashboard", index=False)
    return report_path
