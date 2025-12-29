from __future__ import annotations

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
from pipelines.projections import DEFAULT_LOGISTIC_SCALE, average_total_points, project_game
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
    }


def _project_row(
    row: pd.Series,
    *,
    ratings: Dict[str, float],
    base_total: float,
    status: str,
    home_advantage: float,
    win_prob_k: float,
) -> Dict[str, Any]:
    base = _base_schedule_row(row)
    home = base["home_team"]
    away = base["away_team"]
    applied_home_advantage = 0.0 if base["neutral"] else home_advantage
    home_rating = ratings.get(home)
    away_rating = ratings.get(away)

    projected_winner = None
    projected_spread = None
    projected_win_prob = None
    projected_home_score = None
    projected_away_score = None
    projected_total = None

    if home_rating is not None and away_rating is not None:
        projection = project_game(
            home_rating,
            away_rating,
            home_advantage=applied_home_advantage,
            neutral=base["neutral"],
            k=win_prob_k,
            base_total=base_total if base_total > 0 else None,
            home_team=home,
            away_team=away,
        )
        projected_winner = projection.projected_winner
        projected_spread = projection.projected_spread
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
    fallback_total = average_total_points(played.to_dict(orient="records"))
    metrics = load_model_metrics(db_path, sport=sport, season=season, model=model) or {}
    home_advantages = team_home_advantages(played, ratings)
    fallback_home_advantage = float(metrics.get("home_advantage", 0.0))
    win_prob_k = float(metrics.get("win_prob_k", DEFAULT_LOGISTIC_SCALE))
    base_total = float(metrics.get("base_total", 0.0)) or fallback_total

    schedule_rows: List[Dict[str, Any]] = []
    if not upcoming_only:
        for _, row in played.iterrows():
            home = str(row.get("home_team", "")).strip()
            schedule_rows.append(
                _project_row(
                    row,
                    ratings=ratings,
                    base_total=base_total,
                    status="final",
                    home_advantage=home_advantages.get(home, fallback_home_advantage),
                    win_prob_k=win_prob_k,
                )
            )

    for _, row in upcoming.iterrows():
        home = str(row.get("home_team", "")).strip()
        schedule_rows.append(
            _project_row(
                row,
                ratings=ratings,
                base_total=base_total,
                status="scheduled",
                home_advantage=home_advantages.get(home, fallback_home_advantage),
                win_prob_k=win_prob_k,
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
