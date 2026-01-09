"""Matchup prediction pipeline built on power ratings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from data.repository import load_games, load_model_metrics
from pipelines.common import normalize_games
from pipelines.model_params import resolve_model_params_with_metadata
from config import (
    DEFAULT_WIN_PROB_K,
)
from pipelines.projections import (
    average_total_points,
    fit_total_model,
    team_scoring_averages,
    win_prob_distribution,
)
from pipelines.projection_engines import get_projection_engine
from pipelines.run_rankings import build_rankings
from models.registry import normalize_model_name


@dataclass(frozen=True)
class MatchupPrediction:
    """Structured response from a matchup projection."""

    home_team: str
    away_team: str
    winner: str
    loser: str
    spread: float
    total_points: float
    win_prob: float | None
    win_prob_samples: list[dict[str, float]] | None
    model_win_prob: float | None
    logistic_home_win_prob: float | None
    margin_mean: float
    margin_std: float | None
    total_mean: float
    total_std: float | None
    params_source: str
    tuned_metric_used: str | None


def _completed_games(df: pd.DataFrame) -> pd.DataFrame:
    """Return only games with final scores."""
    if df.empty:
        return df
    mask = df["home_score"].notna() & df["away_score"].notna()
    return df[mask]


def team_home_advantages(
    df: pd.DataFrame, ratings: Dict[str, float]
) -> Dict[str, float]:
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
    """Map team name to point-scale rating."""
    return {
        str(row["team"]).strip(): float(row["points"]) for _, row in rankings.iterrows()
    }


def predict_matchup(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    division: str | None = None,
    conference: str | None = None,
    home_team: str,
    away_team: str,
    model: str = "bradley-terry",
    model_params: dict[str, float] | None = None,
    model_params_file: str | Path | None = None,
    tuned_metric: str | None = None,
) -> MatchupPrediction:
    """Predict a single matchup using stored games and rankings."""
    model = normalize_model_name(model)
    rows = load_games(
        db_path,
        sport=sport,
        season=season,
        division=division,
        conference=conference,
    )
    df = normalize_games(rows)
    if df.empty:
        raise ValueError(f"No games found for sport={sport!r}, season={season!r}")

    resolution = resolve_model_params_with_metadata(
        model,
        params=model_params,
        params_file=model_params_file,
        db_path=db_path,
        sport=sport,
        season=season,
        tuned_metric=tuned_metric,
    )
    resolved_params = resolution.params
    rankings, model_instance = build_rankings(
        df, model=model, model_params=resolved_params, return_model=True
    )
    ratings = _rating_lookup(rankings)
    projection_engine = get_projection_engine(model_instance)
    played = _completed_games(df)
    home_advantages = team_home_advantages(played, ratings)
    metrics = load_model_metrics(db_path, sport=sport, season=season, model=model) or {}
    fallback_home_advantage = float(metrics.get("home_advantage", 0.0))
    backtest_win_prob_k = metrics.get("backtest_win_prob_k")
    win_prob_k = float(
        backtest_win_prob_k
        if backtest_win_prob_k is not None
        else metrics.get("win_prob_k", DEFAULT_WIN_PROB_K)
    )
    if win_prob_k <= 0:
        win_prob_k = DEFAULT_WIN_PROB_K
    base_total = float(metrics.get("base_total", 0.0))
    conditional_sd_intercept = metrics.get("conditional_sd_intercept")
    conditional_sd_slope = metrics.get("conditional_sd_slope")
    played_records = played.to_dict(orient="records")
    total_intercept, total_slope = fit_total_model(played_records, ratings)
    scoring_averages = team_scoring_averages(played_records)

    home_key = home_team.strip()
    away_key = away_team.strip()
    if home_key not in ratings:
        raise ValueError(f"Unknown team: {home_team}")
    if away_key not in ratings:
        raise ValueError(f"Unknown team: {away_team}")

    home_advantage = home_advantages.get(home_key, fallback_home_advantage)
    fallback_total = average_total_points(played.to_dict(orient="records"))
    applied_total = base_total if base_total > 0 else fallback_total
    projection = projection_engine(
        home_key,
        away_key,
        model_instance,
        {
            "ratings": ratings,
            "base_total": applied_total,
            "scoring_averages": scoring_averages,
            "total_intercept": total_intercept,
            "total_slope": total_slope,
            "margin_std": metrics.get("margin_std"),
            "total_std": metrics.get("total_std"),
            "conditional_sd_intercept": conditional_sd_intercept,
            "conditional_sd_slope": conditional_sd_slope,
            "win_prob_k": win_prob_k,
            "home_advantage": home_advantage,
            "neutral": False,
        },
    )

    margin_mean = projection.get("margin_mean")
    margin_sd_value = projection.get("margin_sd")
    total_mean = projection.get("total_mean") or 0.0
    total_sd_value = projection.get("total_sd")
    win_prob_value = projection.get("model_p_home_win")
    if win_prob_value is None:
        win_prob_value = projection.get("projected_win_prob")
    logistic_home_win_prob = projection.get("logistic_home_win_prob")

    if margin_mean is None:
        raise ValueError("Projection engine did not return margin_mean.")

    # Hard-exclude models that report implausible margin SDs which would
    # otherwise produce near-certain covers or meaningless signals. Exclude
    # predictions with margin_sd < 5 or margin_sd > 30 by returning a
    # prediction with no win-prob or samples so eval/ensemble logic ignores it.
    # Only apply the hard-exclude rule for NBA — do not exclude low-scoring
    # sports like NHL where small SDs are expected.
    if (
        margin_sd_value is not None
        and (margin_sd_value < 5.0 or margin_sd_value > 30.0)
        and str(sport).lower() == "nba"
    ):
        return MatchupPrediction(
            home_team=home_key,
            away_team=away_key,
            winner=winner,
            loser=loser,
            spread=-margin_mean,
            total_points=projection.get("projected_total") or 0.0,
            win_prob=None,
            win_prob_samples=None,
            model_win_prob=None,
            logistic_home_win_prob=logistic_home_win_prob,
            margin_mean=margin_mean,
            margin_std=None,
            total_mean=total_mean,
            total_std=None,
            params_source=resolution.params_source,
            tuned_metric_used=resolution.tuned_metric_used,
        )

    if margin_mean >= 0:
        winner, loser = home_key, away_key
    else:
        winner, loser = away_key, home_key

    win_prob_samples = None
    if win_prob_value is not None and margin_sd_value is not None:
        if margin_mean > 0 and win_prob_value <= 0.5 - 1e-3:
            raise ValueError(
                "Inconsistent win probability: margin favors home but p_home_win <= 0.5"
            )
        if margin_mean < 0 and win_prob_value >= 0.5 + 1e-3:
            raise ValueError(
                "Inconsistent win probability: margin favors away but p_home_win >= 0.5"
            )
        win_prob_samples = win_prob_distribution(
            win_prob_value,
            win_prob_k=win_prob_k,
            margin_std=margin_sd_value,
        )

    return MatchupPrediction(
        home_team=home_key,
        away_team=away_key,
        winner=winner,
        loser=loser,
        spread=-margin_mean,
        total_points=projection.get("projected_total") or 0.0,
        win_prob=win_prob_value,
        win_prob_samples=win_prob_samples,
        model_win_prob=win_prob_value,
        logistic_home_win_prob=logistic_home_win_prob,
        margin_mean=margin_mean,
        margin_std=margin_sd_value if margin_mean is not None else None,
        total_mean=total_mean,
        total_std=total_sd_value if total_mean is not None else None,
        params_source=resolution.params_source,
        tuned_metric_used=resolution.tuned_metric_used,
    )


def format_matchup(prediction: MatchupPrediction) -> Tuple[str, Dict[str, float]]:
    """Format a matchup prediction for CLI output."""
    spread_points = abs(prediction.spread)
    prob_suffix = ""
    if prediction.win_prob is not None:
        prob_suffix = f" Win prob: {prediction.win_prob:.1%}."
    margin_std_suffix = ""
    if prediction.margin_std is not None:
        margin_std_suffix = f" (std {prediction.margin_std:.1f})"
    total_std_suffix = ""
    if prediction.total_std is not None:
        total_std_suffix = f" (std {prediction.total_std:.1f})"
    line = (
        f"{prediction.winner} over {prediction.loser} by {spread_points:.1f} points. "
        f"Margin mean: {prediction.margin_mean:.1f}{margin_std_suffix}. "
        f"Projected total: {prediction.total_points:.1f}{total_std_suffix}.{prob_suffix}"
    )
    metrics = {
        "spread": prediction.spread,
        "total_points": prediction.total_points,
        "margin_mean": prediction.margin_mean,
        "total_mean": prediction.total_mean,
        "params_source": prediction.params_source,
        "tuned_metric_used": prediction.tuned_metric_used,
    }
    if prediction.margin_std is not None:
        metrics["margin_std"] = prediction.margin_std
    if prediction.total_std is not None:
        metrics["total_std"] = prediction.total_std
    if prediction.win_prob is not None:
        metrics["win_prob"] = prediction.win_prob
    if prediction.win_prob_samples is not None:
        metrics["win_prob_samples"] = prediction.win_prob_samples
        metrics["win_prob_dist"] = prediction.win_prob_samples
    if prediction.model_win_prob is not None:
        metrics["model_win_prob"] = prediction.model_win_prob
    if prediction.logistic_home_win_prob is not None:
        metrics["logistic_home_win_prob"] = prediction.logistic_home_win_prob
    return line, metrics
