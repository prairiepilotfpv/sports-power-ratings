"""Ranking pipeline for fitting power ratings and storing calibration metrics."""

from __future__ import annotations

import statistics
import math
from pathlib import Path
from typing import Dict

import pandas as pd

from data.paths import processed_path_for
from data.repository import load_games, save_model_metrics
from models.registry import get_model, list_models, normalize_model_name
from pipelines.common import normalize_games, resolve_output_path
from pipelines.model_params import resolve_model_params
from config import (
    CALIBRATION_RESIDUAL_GAMES,
    DEFAULT_MARGIN_SD_FALLBACK,
    DEFAULT_TOTAL_SD_FALLBACK,
    DEFAULT_WIN_PROB_K,
    MIN_CALIBRATION_SAMPLES,
)
from pipelines.projections import (
    average_total_points,
    fit_total_model,
    fit_win_prob_scale,
    total_from_ratings,
)


def _completed_games(df: pd.DataFrame) -> pd.DataFrame:
    """Return only games with both home/away scores."""
    if df.empty:
        return df
    mask = df["home_score"].notna() & df["away_score"].notna()
    return df[mask]


def _empty_rankings() -> pd.DataFrame:
    """Return an empty rankings DataFrame with the expected columns."""
    return pd.DataFrame(columns=["team", "rating", "points", "games"])


def build_rankings(
    df: pd.DataFrame,
    model: str = "bradley-terry",
    *,
    require_scores: bool = True,
    model_params: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Fit a ranking model and return a DataFrame of ratings and point values."""
    working_df = df.copy(deep=True)
    played = _completed_games(working_df)
    if played.empty:
        if require_scores:
            raise ValueError("No completed games available to build rankings.")
        return _empty_rankings()

    model_cls = get_model(model)
    try:
        model_instance = model_cls(**(model_params or {}))
    except TypeError as exc:
        raise ValueError(
            f"Invalid parameters for model {model!r}: {model_params}"
        ) from exc
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
    if not rating_map:
        return _empty_rankings()

    # Convert log rating differences into point-spread units.
    point_scale = _estimate_point_scale(played, rating_map)

    points_ratings: Dict[str, float] = {}
    for team, rating in rating_map.items():
        points_rating = math.log(rating) * point_scale if rating > 0 else 0.0
        points_ratings[team] = points_rating

    centered_points = _center_ratings(points_ratings)

    items = []
    for team, rating in rating_map.items():
        items.append(
            {
                "team": team,
                "rating": rating,
                "points": centered_points.get(team, 0.0),
                "games": games_played.get(team, 0),
            }
        )
    return pd.DataFrame(items).sort_values("rating", ascending=False)


def _estimate_point_scale(df: pd.DataFrame, ratings: Dict[str, float]) -> float:
    """Estimate the point spread scale from rating differences vs actual margins."""
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


def _center_ratings(ratings: Dict[str, float]) -> Dict[str, float]:
    """Center point-scale ratings around zero."""
    if not ratings:
        return {}
    mean_rating = sum(ratings.values()) / len(ratings)
    return {team: value - mean_rating for team, value in ratings.items()}


def _resolve_models(model: str | None) -> list[str]:
    if model is None:
        return list_models()
    return [normalize_model_name(model)]


def run_rankings(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str | None = None,
    output_path: str | Path | None = None,
    model_params: dict[str, float] | None = None,
    model_params_file: str | Path | None = None,
) -> Path | list[Path]:
    """Load games from SQLite, generate rankings, and write them to CSV."""
    rows = load_games(db_path, sport=sport, season=season)
    df = normalize_games(rows)
    if df.empty:
        raise ValueError(f"No games found for sport={sport!r}, season={season!r}")
    models = _resolve_models(model)
    multiple = len(models) > 1
    results: list[Path] = []
    for model_name in models:
        try:
            model_df = df.copy(deep=True)
            resolved_params = resolve_model_params(
                model_name, params=model_params, params_file=model_params_file
            )
            rankings = build_rankings(
                model_df, model=model_name, model_params=resolved_params
            )
        except ValueError as exc:
            if "No completed games" in str(exc):
                raise ValueError(
                    f"No completed games found for sport={sport!r}, season={season!r}"
                ) from exc
            raise
        default_path = processed_path_for(sport, season, "rankings.csv")
        resolved_output = resolve_output_path(
            output_path,
            default_path=default_path,
            model=model_name,
            add_prefix=multiple,
        )
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        rankings.to_csv(resolved_output, index=False)
        _store_model_metrics(
            db_path, model_df, rankings, sport=sport, season=season, model=model_name
        )
        results.append(resolved_output)
    return results[0] if len(results) == 1 else results


def _store_model_metrics(
    db_path: str | Path,
    df: pd.DataFrame,
    rankings: pd.DataFrame,
    *,
    sport: str,
    season: str,
    model: str,
) -> None:
    """Compute and store calibration metrics for projections and win probabilities."""
    if df.empty or rankings.empty:
        return
    played = _completed_games(df)
    if played.empty:
        return
    played = played.sort_values("date")
    calibration_games = played.tail(CALIBRATION_RESIDUAL_GAMES)

    ratings = {
        str(row["team"]).strip(): float(row["points"]) for _, row in rankings.iterrows()
    }
    neutral_mask = played.get("neutral")
    if neutral_mask is not None:
        non_neutral = played[~neutral_mask.astype(bool)]
        sample = non_neutral if not non_neutral.empty else played
    else:
        sample = played

    margins = sample["home_score"].astype(float) - sample["away_score"].astype(float)
    home_advantage = float(margins.mean()) if not margins.empty else 0.0

    base_total = average_total_points(played.to_dict(orient="records"))
    errors = []
    predicted_margins: list[float] = []
    margin_residuals: list[float] = []
    predicted_totals: list[float] = []
    total_residuals: list[float] = []
    total_intercept, total_slope = fit_total_model(
        played.to_dict(orient="records"), ratings
    )
    win_prob_samples: list[tuple[float, int]] = []
    for _, row in calibration_games.iterrows():
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
        neutral_raw = row.get("neutral", False)
        neutral = False if pd.isna(neutral_raw) else bool(neutral_raw)
        predicted_margin = (ratings[home] - ratings[away]) + (
            0.0 if neutral else home_advantage
        )
        errors.append(predicted_margin - margin)
        predicted_margins.append(predicted_margin)
        margin_residuals.append(margin - predicted_margin)
        predicted_total = total_from_ratings(
            home,
            away,
            ratings,
            intercept=total_intercept,
            slope=total_slope,
        )
        if predicted_total is None:
            predicted_total = base_total if base_total > 0 else None
        if predicted_total is not None:
            actual_total = float(row.get("home_score")) + float(row.get("away_score"))
            predicted_totals.append(predicted_total)
            total_residuals.append(actual_total - predicted_total)
        if margin == 0:
            continue
        projected_spread = -predicted_margin
        win_prob_samples.append((projected_spread, 1 if margin > 0 else 0))

    model_error = math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else 0.0
    if model_error == 0.0:
        model_error = 1.0

    win_prob_k = fit_win_prob_scale(win_prob_samples, default_k=DEFAULT_WIN_PROB_K)
    margin_std = (
        statistics.pstdev(margin_residuals) if len(margin_residuals) >= 2 else None
    )
    total_std = (
        statistics.pstdev(total_residuals) if len(total_residuals) >= 2 else None
    )
    if margin_std is None or len(margin_residuals) < MIN_CALIBRATION_SAMPLES:
        margin_std = DEFAULT_MARGIN_SD_FALLBACK
    if total_std is None or len(total_residuals) < MIN_CALIBRATION_SAMPLES:
        total_std = DEFAULT_TOTAL_SD_FALLBACK
    margin_mean = statistics.fmean(predicted_margins) if predicted_margins else None
    total_mean = statistics.fmean(predicted_totals) if predicted_totals else None

    save_model_metrics(
        db_path,
        sport=sport,
        season=season,
        model=model,
        home_advantage=home_advantage,
        model_error=model_error,
        win_prob_k=win_prob_k,
        base_total=base_total,
        margin_std=margin_std,
        total_std=total_std,
        margin_mean=margin_mean,
        total_mean=total_mean,
    )
