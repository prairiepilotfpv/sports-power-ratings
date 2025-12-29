from __future__ import annotations

from pathlib import Path
import math
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
from pipelines.matchups import average_total_points, projected_total_points, team_total_averages
from pipelines.run_rankings import build_rankings


def _completed_games(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df["home_score"].notna() & df["away_score"].notna()
    return df[mask]


def _upcoming_games(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df["home_score"].isna() | df["away_score"].isna()
    return df[mask]


def _rating_lookup(rankings: pd.DataFrame) -> Dict[str, float]:
    return {str(row["team"]).strip(): float(row["points"]) for _, row in rankings.iterrows()}


def _safe_date(value: Any) -> str:
    if value is None:
        return ""
    try:
        return pd.to_datetime(value).date().isoformat()
    except Exception:
        return str(value)


def _base_schedule_row(row: pd.Series) -> Dict[str, Any]:
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
        "notes": row.get("notes"),
    }


def _project_row(
    row: pd.Series,
    *,
    ratings: Dict[str, float],
    team_totals: Dict[str, float],
    fallback_total: float,
    model: str,
    status: str,
    home_advantage: float,
    model_error: float,
) -> Dict[str, Any]:
    base = _base_schedule_row(row)
    home = base["home_team"]
    away = base["away_team"]
    home_rating = ratings.get(home)
    away_rating = ratings.get(away)

    projected_spread = None
    projected_winner = None
    projected_win_prob = None
    if home_rating is not None and away_rating is not None:
        projected_spread = home_rating - away_rating + home_advantage
        projected_winner = home if projected_spread >= 0 else away
        if model_error > 0:
            projected_win_prob = 1.0 / (1.0 + math.exp(-projected_spread / model_error))

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
            "model": model,
            "home_rating": home_rating,
            "away_rating": away_rating,
            "projected_winner": projected_winner,
            "projected_spread": projected_spread,
            "projected_win_prob": projected_win_prob,
            "home_advantage": home_advantage,
            "model_error": model_error,
            "projected_total": projected_total_points(
                team_totals,
                home_team=home,
                away_team=away,
                fallback=fallback_total,
            )
            if fallback_total > 0
            else None,
            "result_margin": result_margin,
            "result_total": result_total,
        }
    )
    return base


def build_schedule_with_projections(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str = "bradley-terry",
    output_path: str | Path | None = None,
    upcoming_only: bool = False,
) -> Path:
    rows = load_games(db_path, sport=sport, season=season)
    df = normalize_games(rows)
    if df.empty:
        raise ValueError(f"No games found for sport={sport!r}, season={season!r}")

    played = _completed_games(df)
    upcoming = _upcoming_games(df)

    rankings = build_rankings(played, model=model, require_scores=False)
    ratings = _rating_lookup(rankings)
    fallback_total = average_total_points(played)
    team_totals = team_total_averages(played)
    metrics = load_model_metrics(db_path, sport=sport, season=season, model=model) or {}
    home_advantage = float(metrics.get("home_advantage", 0.0))
    model_error = float(metrics.get("model_error", 1.0))

    schedule_rows: List[Dict[str, Any]] = []
    if not upcoming_only:
        for _, row in played.iterrows():
            schedule_rows.append(
                _project_row(
                    row,
                    ratings=ratings,
                    team_totals=team_totals,
                    fallback_total=fallback_total,
                    model=model,
                    status="final",
                    home_advantage=home_advantage,
                    model_error=model_error,
                )
            )

    for _, row in upcoming.iterrows():
        schedule_rows.append(
            _project_row(
                row,
                ratings=ratings,
                team_totals=team_totals,
                fallback_total=fallback_total,
                model=model,
                status="scheduled",
                home_advantage=home_advantage,
                model_error=model_error,
            )
        )

    schedule_df = pd.DataFrame(schedule_rows)
    if not schedule_df.empty and "date" in schedule_df.columns:
        schedule_df = schedule_df.assign(_dt=pd.to_datetime(schedule_df["date"], errors="coerce")).sort_values(
            ["_dt", "game_id", "away_team", "home_team"]
        ).drop(columns=["_dt"], errors="ignore")

    if output_path is None:
        output_path = Path("data/processed") / sport / season / "schedule_with_projections.csv"
    else:
        output_path = Path(output_path)
        if output_path.is_dir() or output_path.suffix == "":
            output_path = output_path / "schedule_with_projections.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    schedule_df.to_csv(output_path, index=False)
    return output_path
