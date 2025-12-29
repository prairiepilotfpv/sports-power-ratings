from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from data.repository import load_games, load_model_metrics
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
    win_prob: float | None


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


def _completed_games(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df["home_score"].notna() & df["away_score"].notna()
    return df[mask]


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


def team_home_advantages(df: pd.DataFrame, ratings: Dict[str, float]) -> Dict[str, float]:
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
    played = _completed_games(df)
    home_advantages = team_home_advantages(played, ratings)
    metrics = load_model_metrics(db_path, sport=sport, season=season, model=model) or {}
    fallback_home_advantage = float(metrics.get("home_advantage", 0.0))
    model_error = float(metrics.get("model_error", 1.0))

    home_key = home_team.strip()
    away_key = away_team.strip()
    if home_key not in ratings:
        raise ValueError(f"Unknown team: {home_team}")
    if away_key not in ratings:
        raise ValueError(f"Unknown team: {away_team}")

    home_advantage = home_advantages.get(home_key, fallback_home_advantage)
    spread = ratings[home_key] - ratings[away_key] + home_advantage
    win_prob = None
    if model_error > 0:
        win_prob = 1.0 / (1.0 + math.exp(-spread / model_error))
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
        win_prob=win_prob,
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
