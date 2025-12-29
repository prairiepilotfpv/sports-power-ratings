from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

try:  # Allow execution from repository root or nested directories
    from bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - fallback when bootstrap isn't on sys.path
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from bootstrap import ensure_src_on_path

ensure_src_on_path()

import pandas as pd

from data.repository import load_games, save_model_metrics
from models.registry import get_model
from pipelines.common import normalize_games


def _completed_games(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df["home_score"].notna() & df["away_score"].notna()
    return df[mask]


def _empty_rankings() -> pd.DataFrame:
    return pd.DataFrame(columns=["team", "rating", "points", "games"])


def build_rankings(
    df: pd.DataFrame,
    model: str = "bradley-terry",
    *,
    require_scores: bool = True,
) -> pd.DataFrame:
    played = _completed_games(df)
    if played.empty:
        if require_scores:
            raise ValueError("No completed games available to build rankings.")
        return _empty_rankings()

    model_cls = get_model(model)
    model_instance = model_cls()
    model_instance.fit(played.to_dict(orient="records"))

    games_played: Dict[str, int] = {}
    for _, row in played.iterrows():
        home = str(row.get("home_team", "")).strip()
        away = str(row.get("away_team", "")).strip()
        if not home or not away:
            continue
        games_played[home] = games_played.get(home, 0) + 1
        games_played[away] = games_played.get(away, 0) + 1

    rating_map = dict(model_instance.rankings())

    point_scale = _estimate_point_scale(played, rating_map)

    items = []
    for team, rating in rating_map.items():
        points_rating = math.log(rating) * point_scale if rating > 0 else 0.0
        items.append(
            {
                "team": team,
                "rating": rating,
                "points": points_rating,
                "games": games_played.get(team, 0),
            }
        )
    return pd.DataFrame(items).sort_values("rating", ascending=False)


def _estimate_point_scale(df: pd.DataFrame, ratings: Dict[str, float]) -> float:
    if df.empty or not ratings:
        return 0.0

    numerator = 0.0
    denominator = 0.0
    for _, row in df.iterrows():
        home = str(row.get("home_team", "")).strip()
        away = str(row.get("away_team", "")).strip()
        if not home or not away:
            continue
        rating_home = ratings.get(home)
        rating_away = ratings.get(away)
        if not rating_home or not rating_away:
            continue
        try:
            margin = int(row.get("home_score")) - int(row.get("away_score"))
        except Exception:
            continue
        log_diff = math.log(rating_home) - math.log(rating_away)
        if log_diff == 0.0:
            continue
        numerator += log_diff * margin
        denominator += log_diff * log_diff

    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def run_rankings(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str = "bradley-terry",
    output_path: str | Path | None = None,
) -> Path:
    rows = load_games(db_path, sport=sport, season=season)
    df = normalize_games(rows)
    if df.empty:
        raise ValueError(f"No games found for sport={sport!r}, season={season!r}")
    try:
        rankings = build_rankings(df, model=model)
    except ValueError as exc:
        if "No completed games" in str(exc):
            raise ValueError(f"No completed games found for sport={sport!r}, season={season!r}") from exc
        raise
    if output_path is None:
        output_path = Path("data/processed") / sport / season / "rankings.csv"
    else:
        output_path = Path(output_path)
        if output_path.is_dir() or output_path.suffix == "":
            output_path = output_path / "rankings.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(output_path, index=False)
    _store_model_metrics(db_path, df, rankings, sport=sport, season=season, model=model)
    return output_path


def _store_model_metrics(
    db_path: str | Path,
    df: pd.DataFrame,
    rankings: pd.DataFrame,
    *,
    sport: str,
    season: str,
    model: str,
) -> None:
    if df.empty or rankings.empty:
        return
    played = _completed_games(df)
    if played.empty:
        return

    ratings = {str(row["team"]).strip(): float(row["points"]) for _, row in rankings.iterrows()}
    neutral_mask = played.get("neutral")
    if neutral_mask is not None:
        non_neutral = played[~neutral_mask.astype(bool)]
        sample = non_neutral if not non_neutral.empty else played
    else:
        sample = played

    margins = sample["home_score"].astype(float) - sample["away_score"].astype(float)
    home_advantage = float(margins.mean()) if not margins.empty else 0.0

    errors = []
    for _, row in played.iterrows():
        home = str(row.get("home_team", "")).strip()
        away = str(row.get("away_team", "")).strip()
        if not home or not away:
            continue
        if home not in ratings or away not in ratings:
            continue
        try:
            margin = float(row.get("home_score")) - float(row.get("away_score"))
        except Exception:
            continue
        predicted = (ratings[home] - ratings[away]) + home_advantage
        errors.append(predicted - margin)

    model_error = math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else 0.0
    if model_error == 0.0:
        model_error = 1.0

    save_model_metrics(
        db_path,
        sport=sport,
        season=season,
        model=model,
        home_advantage=home_advantage,
        model_error=model_error,
    )
