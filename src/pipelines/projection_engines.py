"""Projection engine registry for schedule/matchup projections."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

import numpy as np

from config import (
    DEFAULT_MARGIN_SD_FALLBACK,
    DEFAULT_TOTAL_SD_FALLBACK,
    DEFAULT_WIN_PROB_K,
    LEAGUE_MARGIN_SD_DEFAULT,
    MARGIN_SD_GUARDRAIL_MAX,
    MARGIN_SD_GUARDRAIL_MIN,
)
from models.calibration import ConditionalSDModel, guardrail_margin_sd
from eval.validation import get_validation_config
from models.base import _home_win_prob_from_margin
from pipelines.projections import (
    matchup_total_from_averages,
    project_game,
    total_from_ratings,
)

_LOGGER = logging.getLogger(__name__)
_DEBUG_ASSERT = os.getenv("TOOR_MARGIN_SD_ASSERT", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

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
    # Enforce explicit rating units: projection logic expects ratings in
    # spread/points units (i.e. implied points). Callers must set
    # `projection_context["rating_units"] = "points"` or obtain ratings
    # via `build_rankings(..., include_implied_points=True)`.
    rating_units = context.get("rating_units")
    if rating_units != "points":
        raise ValueError(
            "Projection engine requires ratings expressed in points units. "
            "Call build_rankings(..., include_implied_points=True) or set "
            "projection_context['rating_units'] = 'points'."
        )
    home_rating = ratings.get(home_team)
    away_rating = ratings.get(away_team)
    sport = context.get("sport")
    cfg = get_validation_config(sport)

    guardrail_context = {
        "model_id": model_id,
        "game_id": context.get("game_id"),
        "date": context.get("game_date"),
        "home_team": home_team,
        "away_team": away_team,
        "sd_sample_size": context.get("sd_sample_size"),
        "residual_min": context.get("sd_residual_min"),
        "residual_max": context.get("sd_residual_max"),
    }
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

    raw_margin_sd = context.get("margin_std")
    margin_sd_reason = None
    if raw_margin_sd is None or raw_margin_sd <= 0:
        raw_margin_sd = DEFAULT_MARGIN_SD_FALLBACK
        margin_sd_reason = "missing_margin_std"
    conditional_sd_intercept = context.get("conditional_sd_intercept")
    conditional_sd_slope = context.get("conditional_sd_slope")
    if (
        projection.margin is not None
        and conditional_sd_intercept is not None
        and conditional_sd_slope is not None
    ):
        margin_sd = ConditionalSDModel(
            intercept=float(conditional_sd_intercept),
            slope=float(conditional_sd_slope),
        ).predict(
            projection.margin,
            guardrail_min=cfg.margin_sd_min,
            guardrail_max=cfg.margin_sd_max,
            fallback_sd=LEAGUE_MARGIN_SD_DEFAULT,
            logger_override=_LOGGER,
            log_context=guardrail_context,
            debug_assert=_DEBUG_ASSERT,
        )
    else:
        margin_sd, reason = guardrail_margin_sd(
            raw_margin_sd,
            fallback_sd=LEAGUE_MARGIN_SD_DEFAULT,
            guardrail_min=cfg.margin_sd_min,
            guardrail_max=cfg.margin_sd_max,
        )
        if reason is None:
            reason = margin_sd_reason
        if reason and _LOGGER.isEnabledFor(logging.DEBUG):
            parts = [
                f"reason={reason}",
                f"raw_sd={raw_margin_sd}",
                f"applied_sd={margin_sd}",
                f"date={guardrail_context.get('date')}",
                f"game_id={guardrail_context.get('game_id')}",
                f"home={home_team}",
                f"away={away_team}",
                f"n={guardrail_context.get('sd_sample_size')}",
                f"resid_min={guardrail_context.get('residual_min')}",
                f"resid_max={guardrail_context.get('residual_max')}",
            ]
            message = "margin_sd guardrail applied; " + ", ".join(
                str(part) for part in parts if part is not None
            )
            _LOGGER.debug(
                message,
                extra={
                    **guardrail_context,
                    "reason": reason,
                    "raw_margin_sd": raw_margin_sd,
                    "applied_margin_sd": margin_sd,
                },
            )
        if _DEBUG_ASSERT and margin_sd < cfg.margin_sd_min:
            raise AssertionError(
                f"margin_sd guardrail violated: applied={margin_sd} raw={raw_margin_sd}"
            )

    total_sd = context.get("total_std")
    if total_sd is None or total_sd <= 0:
        total_sd = DEFAULT_TOTAL_SD_FALLBACK

    normal_p_home_win = _home_win_prob_from_margin(
        projection.margin, margin_sd
    ) if projection.margin is not None else None
    projected_win_prob = normal_p_home_win
    # By default, expose the margin-derived probability as the legacy
    # `model_p_home_win` so downstream consumers relying on the old
    # semantics continue to function. The direct Bradley-Terry logistic
    # probability (when available) is exposed as `logistic_home_win_prob`.
    model_p_home_win = normal_p_home_win
    win_prob_source = "margin_normal"
    margin_dist_assumption = "normal_approx"
    if model_id == "bradley-terry":
        # For raw bradley-terry models, attempt to fetch direct logistic
        # probability but keep the legacy margin-based field authoritative.
        if hasattr(model, "predict_probability"):
            venue = "neutral" if neutral else "home"
            try:
                logistic_prob = float(
                    model.predict_probability(
                        home_team,
                        away_team,
                        venue=venue,
                    )
                )
            except Exception:
                logistic_prob = None
        else:
            logistic_prob = projection.projected_win_prob
        # expose logistic prob separately
        logistic_home_win_prob = logistic_prob
    else:
        logistic_home_win_prob = projection.projected_win_prob
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

def _elo_projection_engine(
    home_team: str,
    away_team: str,
    model: Any,
    context: ProjectionContext,
) -> ProjectionOutput:
    projection = _rating_projection_engine(home_team, away_team, model, context)
    logistic_p = projection.get("logistic_home_win_prob")
    if logistic_p is None:
        return projection
    projection.update(
        {
            "projected_win_prob": logistic_p,
            "model_p_home_win": logistic_p,
            "win_prob_source": "logistic",
            "logistic_home_win_prob": logistic_p,
            "margin_dist_assumption": "normal_approx",
        }
    )
    return projection


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
    # For pipeline projections (model-based scheduling, reporting), the
    # projection engine should expose the model's direct logistic win
    # probability as authoritative and suppress the margin-derived
    # probability to avoid confusing consumers that expect a direct
    # win-prob for ranking models.
    model_p = projection.get("model_p_home_win", projection.get("p_home_win"))
    # For raw Bradley-Terry models, we prefer the logistic probability and
    # do not surface the margin-normal derived probability in the pipeline
    # projection output (keep it None) to match historical pipeline behavior.
    normal_p = None if model_p is not None else projection.get("normal_p_home_win")
    return {
        "projected_home_score": projection.get("projected_home_score"),
        "projected_away_score": projection.get("projected_away_score"),
        "projected_total": projection.get("total_mean"),
        "projected_win_prob": model_p,
        "model_p_home_win": model_p,
        "normal_p_home_win": normal_p,
        "win_prob_source": "direct" if model_p is not None else "bt_margin_normal",
        "margin_mean": projection.get("margin_mean"),
        "margin_sd": projection.get("margin_sd"),
        "total_mean": projection.get("total_mean"),
        "total_sd": projection.get("total_sd"),
        "margin_dist_assumption": projection.get("margin_dist_assumption", "none"),
        "logistic_home_win_prob": model_p,
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
register_projection_engine("elo", _elo_projection_engine)
register_projection_engine("bradley-terry", _bt_projection_engine)
register_projection_engine("poisson", _poisson_projection_engine)
