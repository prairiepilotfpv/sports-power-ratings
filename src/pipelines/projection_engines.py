"""Projection engine registry for schedule/matchup projections."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from config import DEFAULT_MARGIN_SD_FALLBACK, DEFAULT_TOTAL_SD_FALLBACK, DEFAULT_WIN_PROB_K
from models.calibration import ConditionalSDModel
from models.base import _home_win_prob_from_margin
from pipelines.projections import (
    matchup_total_from_averages,
    project_game,
    total_from_ratings,
)

ProjectionOutput = dict[str, float | None | str]
ProjectionContext = dict[str, Any]
ProjectionEngine = Callable[[str, str, Any, ProjectionContext], ProjectionOutput]

_ENGINES: dict[str, ProjectionEngine] = {}
_DEFAULT_ENGINE_KEY = "default"


def register_projection_engine(model_id: str, engine: ProjectionEngine) -> None:
    """Register a projection engine for a model id."""
    _ENGINES[model_id] = engine


def get_projection_engine(model: Any) -> ProjectionEngine:
    """Resolve the projection engine for a model instance."""
    model_id = None
    if hasattr(model, "metadata") and callable(model.metadata):
        meta = model.metadata()
        model_id = getattr(meta, "model_id", None)
    if model_id is None:
        model_id = getattr(model, "model_id", None)
    if callable(model_id):
        model_id = model_id()
    model_key = str(model_id) if model_id is not None else _DEFAULT_ENGINE_KEY
    return _ENGINES.get(model_key, _ENGINES[_DEFAULT_ENGINE_KEY])


def _rating_projection_engine(
    home_team: str,
    away_team: str,
    model: Any,
    context: ProjectionContext,
) -> ProjectionOutput:
    model_id = None
    if hasattr(model, "metadata") and callable(model.metadata):
        meta = model.metadata()
        model_id = getattr(meta, "model_id", None)
    if model_id is None:
        model_id = getattr(model, "model_id", None)
    if callable(model_id):
        model_id = model_id()
    ratings = context.get("ratings", {})
    home_rating = ratings.get(home_team)
    away_rating = ratings.get(away_team)
    if home_rating is None or away_rating is None:
        return {
            "projected_home_score": None,
            "projected_away_score": None,
            "projected_total": None,
            "projected_win_prob": None,
            "margin_mean": None,
            "margin_sd": None,
            "total_mean": None,
            "total_sd": None,
            "logistic_home_win_prob": None,
        }

    base_total = float(context.get("base_total", 0.0))
    scoring_averages = context.get("scoring_averages", {})
    total_intercept = context.get("total_intercept")
    total_slope = context.get("total_slope")
    neutral = bool(context.get("neutral", False))
    home_advantage = float(context.get("home_advantage", 0.0))
    win_prob_k = float(context.get("win_prob_k", DEFAULT_WIN_PROB_K))

    matchup_total = matchup_total_from_averages(home_team, away_team, scoring_averages)
    model_total = None
    if total_intercept is not None and total_slope is not None:
        model_total = total_from_ratings(
            home_team,
            away_team,
            ratings,
            intercept=float(total_intercept),
            slope=float(total_slope),
        )
    applied_total = model_total or matchup_total or (base_total if base_total > 0 else None)

    projection = project_game(
        home_rating,
        away_rating,
        home_advantage=home_advantage,
        neutral=neutral,
        k=win_prob_k,
        base_total=applied_total,
        home_team=home_team,
        away_team=away_team,
    )

    margin_sd = context.get("margin_std")
    if margin_sd is None or margin_sd <= 0:
        margin_sd = DEFAULT_MARGIN_SD_FALLBACK
    conditional_sd_intercept = context.get("conditional_sd_intercept")
    conditional_sd_slope = context.get("conditional_sd_slope")
    if projection.margin is not None and conditional_sd_intercept is not None:
        if conditional_sd_slope is not None:
            margin_sd = ConditionalSDModel(
                intercept=float(conditional_sd_intercept),
                slope=float(conditional_sd_slope),
            ).predict(projection.margin)

    total_sd = context.get("total_std")
    if total_sd is None or total_sd <= 0:
        total_sd = DEFAULT_TOTAL_SD_FALLBACK

    normal_p_home_win = _home_win_prob_from_margin(
        projection.margin, margin_sd
    ) if projection.margin is not None else None
    projected_win_prob = normal_p_home_win
    model_p_home_win = projection.projected_win_prob
    win_prob_source = "logistic"
    margin_dist_assumption = "normal_approx"
    if model_id == "bradley-terry":
        win_prob_source = "direct"
        margin_dist_assumption = "none"
        if hasattr(model, "predict_probability"):
            venue = "neutral" if neutral else "home"
            try:
                model_p_home_win = float(
                    model.predict_probability(
                        home_team,
                        away_team,
                        venue=venue,
                    )
                )
            except Exception:
                pass
        normal_p_home_win = None
        projected_win_prob = None
    logistic_home_win_prob = projection.projected_win_prob
    if win_prob_source == "direct" and model_p_home_win is not None:
        logistic_home_win_prob = model_p_home_win
    return {
        "projected_home_score": projection.projected_home_score,
        "projected_away_score": projection.projected_away_score,
        "projected_total": projection.projected_total,
        "projected_win_prob": projected_win_prob,
        "model_p_home_win": model_p_home_win,
        "normal_p_home_win": normal_p_home_win,
        "win_prob_source": win_prob_source,
        "margin_dist_assumption": margin_dist_assumption,
        "margin_mean": projection.margin,
        "margin_sd": margin_sd if projection.margin is not None else None,
        "total_mean": projection.projected_total,
        "total_sd": total_sd if projection.projected_total is not None else None,
        "logistic_home_win_prob": logistic_home_win_prob,
    }


def _bt_projection_engine(
    home_team: str,
    away_team: str,
    model: Any,
    context: ProjectionContext,
) -> ProjectionOutput:
    if not hasattr(model, "project_matchup"):
        return _rating_projection_engine(home_team, away_team, model, context)
    neutral = bool(context.get("neutral", False))
    projection = model.project_matchup(home_team, away_team, neutral=neutral)
    return {
        "projected_home_score": projection["projected_home_score"],
        "projected_away_score": projection["projected_away_score"],
        "projected_total": projection["total_mean"],
        "projected_win_prob": projection["p_home_win"],
        "margin_mean": projection["margin_mean"],
        "margin_sd": projection["margin_sd"],
        "total_mean": projection["total_mean"],
        "total_sd": projection["total_sd"],
        "logistic_home_win_prob": None,
    }


def _poisson_projection_engine(
    home_team: str,
    away_team: str,
    model: Any,
    context: ProjectionContext,
) -> ProjectionOutput:
    if not hasattr(model, "simulate_matchup"):
        raise ValueError("Poisson projection engine requires simulate_matchup on model.")
    neutral = bool(context.get("neutral", False))
    n_simulations = context.get("n_simulations")

    samples = model.simulate_matchup(
        home_team,
        away_team,
        neutral=neutral,
        n_simulations=n_simulations,
    )
    if samples is None:
        return {
            "projected_home_score": None,
            "projected_away_score": None,
            "projected_total": None,
            "projected_win_prob": None,
            "margin_mean": None,
            "margin_sd": None,
            "total_mean": None,
            "total_sd": None,
            "logistic_home_win_prob": None,
        }

    home_samples, away_samples = samples
    total_samples = home_samples + away_samples
    margin_samples = home_samples - away_samples

    projected_home_score = float(np.mean(home_samples))
    projected_away_score = float(np.mean(away_samples))
    projected_total = float(np.mean(total_samples))
    margin_mean = float(np.mean(margin_samples))
    margin_sd = float(np.std(margin_samples))
    total_mean = projected_total
    total_sd = float(np.std(total_samples))
    win_prob = float(
        np.mean(margin_samples > 0)
        + 0.5 * np.mean(margin_samples == 0)
    )

    return {
        "projected_home_score": projected_home_score,
        "projected_away_score": projected_away_score,
        "projected_total": projected_total,
        "projected_win_prob": None,
        "model_p_home_win": win_prob,
        "normal_p_home_win": None,
        "win_prob_source": "sample",
        "margin_dist_assumption": "empirical",
        "margin_mean": margin_mean,
        "margin_sd": margin_sd,
        "total_mean": total_mean,
        "total_sd": total_sd,
        "logistic_home_win_prob": None,
    }


register_projection_engine(_DEFAULT_ENGINE_KEY, _rating_projection_engine)
register_projection_engine("bradley-terry", _bt_projection_engine)
register_projection_engine("poisson", _poisson_projection_engine)
