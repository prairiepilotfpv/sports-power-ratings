from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from data.repository import load_games
from pipelines.common import normalize_games
from pipelines.run_rankings import build_rankings


@dataclass(frozen=True)
class MatchupPrediction:
    home_team: str
    away_team: str
    winner: str
    loser: str
    spread: float
    total_points: float


def average_total_points(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    totals = []
    for _, row in df.iterrows():
        try:
            total = float(row.get("home_score")) + float(row.get("away_score"))
        except Exception:
            continue
        totals.append(total)
    if not totals:
        return 0.0
    return sum(totals) / len(totals)


def team_total_averages(df: pd.DataFrame) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    if df.empty:
        return totals

    for _, row in df.iterrows():
        try:
            total = float(row.get("home_score")) + float(row.get("away_score"))
        except Exception:
            continue
        home = str(row.get("home_team", "")).strip()
        away = str(row.get("away_team", "")).strip()
        if home:
            totals[home] = totals.get(home, 0.0) + total
            counts[home] = counts.get(home, 0) + 1
        if away:
            totals[away] = totals.get(away, 0.0) + total
            counts[away] = counts.get(away, 0) + 1

    return {team: totals[team] / counts[team] for team in totals if counts.get(team)}


def projected_total_points(
    team_totals: Dict[str, float],
    *,
    home_team: str,
    away_team: str,
    fallback: float,
) -> float:
    home_total = team_totals.get(home_team)
    away_total = team_totals.get(away_team)
    if home_total is not None and away_total is not None:
        return (home_total + away_total) / 2.0
    if home_total is not None:
        return home_total
    if away_total is not None:
        return away_total
    return fallback


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

    home_key = home_team.strip()
    away_key = away_team.strip()
    if home_key not in ratings:
        raise ValueError(f"Unknown team: {home_team}")
    if away_key not in ratings:
        raise ValueError(f"Unknown team: {away_team}")

    spread = ratings[home_key] - ratings[away_key]
    if spread >= 0:
        winner, loser = home_key, away_key
    else:
        winner, loser = away_key, home_key

    overall_total = average_total_points(df)
    total_points = projected_total_points(
        team_total_averages(df),
        home_team=home_key,
        away_team=away_key,
        fallback=overall_total,
    )

    return MatchupPrediction(
        home_team=home_key,
        away_team=away_key,
        winner=winner,
        loser=loser,
        spread=spread,
        total_points=total_points,
    )


def format_matchup(prediction: MatchupPrediction) -> Tuple[str, Dict[str, float]]:
    spread_points = abs(prediction.spread)
    line = (
        f"{prediction.winner} over {prediction.loser} by {spread_points:.1f} points. "
        f"Projected total: {prediction.total_points:.1f}."
    )
    metrics = {
        "spread": prediction.spread,
        "total_points": prediction.total_points,
    }
    return line, metrics
