"""Ranking pipeline for fitting power ratings and storing calibration metrics."""

from __future__ import annotations

import statistics
import math
import inspect
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import numpy as np

from data.paths import processed_path_for
from data.repository import load_games, save_model_metrics
from markets.base import Market
from models.registry import get_model, list_models, normalize_model_name
from models.calibration import fit_conditional_sd, resolve_fit_end_date
from pipelines.common import normalize_games, resolve_output_path
from pipelines.model_params import resolve_model_market_params_with_metadata
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
    return pd.DataFrame(columns=["team", "rating", "games"])


def _filter_kwargs(params: dict[str, Any], func) -> dict[str, Any]:
    """Return a dict containing only kwargs accepted by `func`.

    If `func` accepts **kwargs (VAR_KEYWORD), return `params` as-is.
    """
    if not params:
        return {}
    sig = inspect.signature(func)
    allowed: set[str] = set()
    for name, param in sig.parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            # func accepts arbitrary kwargs; nothing to filter
            return dict(params)
        if name in ("self", "cls"):
            continue
        if param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            allowed.add(name)
    return {k: v for k, v in params.items() if k in allowed}


def build_rankings(
    df: pd.DataFrame,
    model: str = "bradley-terry",
    *,
    require_scores: bool = True,
    model_params: dict[str, float] | None = None,
    include_implied_points: bool = False,
    return_model: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, Any]:
    """Fit a ranking model and return a DataFrame of ratings and point values."""
    working_df = df.copy(deep=True)
    played = _completed_games(working_df)

    model_cls = get_model(model)

    # Only pass kwargs to the model __init__ that the class actually accepts.
    init_kwargs = _filter_kwargs(model_params or {}, model_cls)
    try:
        model_instance = model_cls(**init_kwargs)
    except TypeError as exc:
        raise ValueError(
            f"Invalid parameters for model {model!r}: {model_params}"
        ) from exc
    if played.empty and require_scores:
        raise ValueError("No completed games available to build rankings.")

    # Filter kwargs for the model's fit method (e.g., recency_lambda for GSSD).
    fit_kwargs = _filter_kwargs(model_params or {}, model_instance.fit)
    # Resolve a single as-of date for recency semantics and pass explicitly when supported.
    fit_end_date = resolve_fit_end_date(played)
    try:
        model_instance.fit(played.to_dict(orient="records"), fit_end_date=fit_end_date, **fit_kwargs)
    except TypeError:
        # Backward compatibility: models that do not accept fit_end_date
        model_instance.fit(played.to_dict(orient="records"), **fit_kwargs)

    if played.empty:
        empty = _empty_rankings()
        if include_implied_points:
            # Ensure callers requesting implied points receive the expected
            # empty schema so downstream code can safely reference the
            # `implied_points` column even when no teams are present.
            empty = empty.copy()
            empty["implied_points"] = pd.Series(dtype=float)
        return (empty, model_instance) if return_model else empty

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
        empty = _empty_rankings()
        if include_implied_points:
            empty = empty.copy()
            empty["implied_points"] = pd.Series(dtype=float)
        return (empty, model_instance) if return_model else empty

    # Convert rating differences into point-spread units only when requested.
    items = []
    implied_points_map: dict[str, float] | None = None
    if include_implied_points:
        model_id = getattr(model_instance, "model_id", None)
        use_log_scale = all(rating > 0 for rating in rating_map.values()) and (
            model_id != "bradley-terry"
        )
        point_scale = _estimate_point_scale(
            played, rating_map, use_log_scale=use_log_scale
        )

        points_ratings: Dict[str, float] = {}
        for team, rating in rating_map.items():
            if use_log_scale:
                points_rating = math.log(rating) * point_scale if rating > 0 else 0.0
            else:
                points_rating = rating * point_scale
            points_ratings[team] = points_rating

        centered_points = _center_ratings(points_ratings)
        implied_points_map = centered_points

    for team, rating in rating_map.items():
        row = {
            "team": team,
            "rating": rating,
            "games": games_played.get(team, 0),
        }
        if include_implied_points and implied_points_map is not None:
            row.update(
                {
                    "implied_points": implied_points_map.get(team, 0.0),
                    "point_scale": float(point_scale),
                    "use_log_scale": bool(use_log_scale),
                }
            )
        items.append(row)
    rankings_df = pd.DataFrame(items).sort_values("rating", ascending=False)
    return (rankings_df, model_instance) if return_model else rankings_df


def _estimate_point_scale(
    df: pd.DataFrame, ratings: Dict[str, float], *, use_log_scale: bool
) -> float:
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
        if rating_home is None or rating_away is None:
            continue
        try:
            margin = int(row.get("home_score")) - int(row.get("away_score"))
        except Exception:
            continue
        if use_log_scale:
            if rating_home <= 0 or rating_away <= 0:
                continue
            rating_diff = math.log(rating_home) - math.log(rating_away)
        else:
            rating_diff = rating_home - rating_away
        if rating_diff == 0.0:
            continue
        numerator += rating_diff * margin
        denominator += rating_diff * rating_diff

    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _center_ratings(ratings: Dict[str, float]) -> Dict[str, float]:
    """Center point-scale ratings around zero."""
    if not ratings:
        return {}
    mean_rating = sum(ratings.values()) / len(ratings)
    return {team: value - mean_rating for team, value in ratings.items()}


def _safe_pstdev(values: list[float]) -> float | None:
    """Compute population standard deviation using native Python floats.

    This avoids mixing types (numpy floats, Fractions) which can trigger
    unexpected code paths inside `statistics.pstdev` on some Python
    versions (see RuntimeWarning / AttributeError from Fraction code).
    Returns None when there are fewer than 2 samples to match previous
    behaviour in this module.
    """
    if not values or len(values) < 2:
        return None
    vals = [float(v) for v in values]
    mean = statistics.fmean(vals)
    return math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))


def _resolve_models(model: str | None) -> list[str]:
    if model is None:
        return list_models()
    normalized = normalize_model_name(model)
    if normalized in {"all", "*"}:
        return list_models()
    return [normalized]


def run_rankings(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    division: str | None = None,
    conference: str | None = None,
    model: str | None = None,
    output_path: str | Path | None = None,
    model_params: dict[str, float] | None = None,
    model_params_file: str | Path | None = None,
    tuned_metric: str | None = None,
) -> Path | list[Path]:
    """Load games from SQLite, generate rankings, and write them to CSV."""
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
    models = _resolve_models(model)
    multiple = len(models) > 1
    results: list[Path] = []
    for model_name in models:
        try:
            model_df = df.copy(deep=True)
            resolution = resolve_model_market_params_with_metadata(
                model_name,
                params=model_params,
                params_file=model_params_file,
                db_path=db_path,
                sport=sport,
                season=season,
                tuned_metric=tuned_metric,
                market=Market.ML,
            )
            resolved_params = resolution.params
            rankings = build_rankings(
                model_df, model=model_name, model_params=resolved_params
            )
            rankings = rankings.assign(
                params_source=resolution.params_source,
                tuned_metric_used=resolution.tuned_metric_used,
                params_run_id=resolution.source_run_id,
            )
            rankings = rankings.drop(columns=["params_run_id"], errors="ignore")
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

    # Ensure we have spread-unit ratings available. If the provided `rankings`
    # DataFrame does not include implied points, explicitly re-run build_rankings
    # requesting `include_implied_points=True` so downstream calibration uses
    # the intended spread units.
    if "implied_points" in rankings.columns:
        ratings = {
            str(row["team"]).strip(): float(row["implied_points"]) for _, row in rankings.iterrows()
        }
    else:
        # Recompute rankings with implied points to derive spread-unit ratings.
        fallback_rankings = build_rankings(df, model=model, include_implied_points=True)
        ratings = {
            str(row["team"]).strip(): float(row["implied_points"]) for _, row in fallback_rankings.iterrows()
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
    margin_std = _safe_pstdev(margin_residuals)
    total_std = _safe_pstdev(total_residuals)
    conditional_sd_model = None
    if len(margin_residuals) >= MIN_CALIBRATION_SAMPLES:
        conditional_sd_model = fit_conditional_sd(
            np.asarray(predicted_margins, dtype=float),
            np.asarray(margin_residuals, dtype=float),
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
        conditional_sd_intercept=conditional_sd_model.intercept
        if conditional_sd_model
        else None,
        conditional_sd_slope=conditional_sd_model.slope if conditional_sd_model else None,
        margin_mean=margin_mean,
        total_mean=total_mean,
    )
