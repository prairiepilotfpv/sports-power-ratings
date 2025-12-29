from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from data.repository import load_games, load_model_metrics
from pipelines.common import normalize_games
from config import DEFAULT_WIN_PROB_K
from pipelines.projections import (
    average_total_points,
    matchup_total_from_averages,
    project_game,
    team_scoring_averages,
)
from pipelines.run_rankings import build_rankings


@dataclass(frozen=True)
class MatchupPrediction:
    home_team: str
    away_team: str
    winner: str
    loser: str
    spread: float
    total_points: float
    win_prob: float | None


def _completed_games(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df["home_score"].notna() & df["away_score"].notna()
    return df[mask]


def team_home_advantages(df: pd.DataFrame, ratings: Dict[str, float]) -> Dict[str, float]:
    """Estimate team-specific home advantages from residual margins."""
    if df.empty or not ratings:
        return {}

    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for _, row in df.iterrows():
        neutral_raw = row.get("neutral", False)
        neutral = False if pd.isna(neutral_raw) else bool(neutral_raw)
        if neutral:
            continue
        home = str(row.get("home_team", "")).strip()
        away = str(row.get("away_team", "")).strip()
        if not home or not away:
            continue
        home_rating = ratings.get(home)
        away_rating = ratings.get(away)
        if home_rating is None or away_rating is None:
            continue
        try:
            margin = float(row.get("home_score")) - float(row.get("away_score"))
        except Exception:
            continue
        residual = margin - (home_rating - away_rating)
        sums[home] = sums.get(home, 0.0) + residual
        counts[home] = counts.get(home, 0) + 1

    return {team: sums[team] / counts[team] for team in sums if counts.get(team)}


def _rating_lookup(rankings: pd.DataFrame) -> Dict[str, float]:
    return {str(row["team"]).strip(): float(row["points"]) for _, row in rankings.iterrows()}


def predict_matchup(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    home_team: str,
    away_team: str,
    model: str = "bradley-terry",
) -> MatchupPrediction:
    rows = load_games(db_path, sport=sport, season=season)
    df = normalize_games(rows)
    if df.empty:
        raise ValueError(f"No games found for sport={sport!r}, season={season!r}")

    rankings = build_rankings(df, model=model)
    ratings = _rating_lookup(rankings)
    played = _completed_games(df)
    home_advantages = team_home_advantages(played, ratings)
    metrics = load_model_metrics(db_path, sport=sport, season=season, model=model) or {}
    fallback_home_advantage = float(metrics.get("home_advantage", 0.0))
    win_prob_k = float(metrics.get("win_prob_k", DEFAULT_WIN_PROB_K))
    base_total = float(metrics.get("base_total", 0.0))
    scoring_averages = team_scoring_averages(played.to_dict(orient="records"))

    home_key = home_team.strip()
    away_key = away_team.strip()
    if home_key not in ratings:
        raise ValueError(f"Unknown team: {home_team}")
    if away_key not in ratings:
        raise ValueError(f"Unknown team: {away_team}")

    home_advantage = home_advantages.get(home_key, fallback_home_advantage)
    fallback_total = average_total_points(played.to_dict(orient="records"))
    applied_total = base_total if base_total > 0 else fallback_total
    matchup_total = matchup_total_from_averages(home_key, away_key, scoring_averages)
    projection = project_game(
        ratings[home_key],
        ratings[away_key],
        home_advantage=home_advantage,
        neutral=False,
        k=win_prob_k,
        base_total=matchup_total or (applied_total if applied_total > 0 else None),
        home_team=home_key,
        away_team=away_key,
    )
    if projection.margin >= 0:
        winner, loser = home_key, away_key
    else:
        winner, loser = away_key, home_key

    return MatchupPrediction(
        home_team=home_key,
        away_team=away_key,
        winner=winner,
        loser=loser,
        spread=projection.projected_spread,
        total_points=projection.projected_total or 0.0,
        win_prob=projection.projected_win_prob,
    )


def format_matchup(prediction: MatchupPrediction) -> Tuple[str, Dict[str, float]]:
    spread_points = abs(prediction.spread)
    prob_suffix = ""
    if prediction.win_prob is not None:
        prob_suffix = f" Win prob: {prediction.win_prob:.1%}."
    line = (
        f"{prediction.winner} over {prediction.loser} by {spread_points:.1f} points. "
        f"Projected total: {prediction.total_points:.1f}.{prob_suffix}"
    )
    metrics = {
        "spread": prediction.spread,
        "total_points": prediction.total_points,
    }
    if prediction.win_prob is not None:
        metrics["win_prob"] = prediction.win_prob
    return line, metrics
