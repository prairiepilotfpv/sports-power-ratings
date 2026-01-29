"""Projection heads for Elo model."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from config import (
    DEFAULT_MARGIN_SD_FALLBACK,
    DEFAULT_TOTAL_SD_FALLBACK,
    LEAGUE_MARGIN_SD_DEFAULT,
    MARGIN_SD_GUARDRAIL_MAX,
    MARGIN_SD_GUARDRAIL_MIN,
)
from forecasting.heads.base import Head, HeadSequence
from forecasting.heads.registry import register_model_heads
from models.base import _home_win_prob_from_margin
from models.calibration import ConditionalSDModel, guardrail_margin_sd
from pipelines.projections import (
    logistic_win_prob,
    project_game,
    total_from_ratings,
    matchup_total_from_averages,
)
from eval.validation import get_validation_config

_LOGGER = logging.getLogger(__name__)


class EloMarginHead(Head):
    """Derives margin_mean and margin_sd from Elo ratings."""

    @property
    def name(self) -> str:
        return "elo_margin_head"

    def produces(self) -> set[str]:
        return {"margin_mean", "margin_sd"}

    def requires(self) -> set[str]:
        # These come from context, not the dataframe
        return {"home_team", "away_team"}

    def apply(self, df: pd.DataFrame, context: dict[str, Any]) -> None:
        """
        Derive margin_mean and margin_sd from Elo ratings using the same logic as
        _rating_projection_engine in projection_engines.py.
        """
        if "margin_mean" not in df.columns:
            df["margin_mean"] = None
        if "margin_sd" not in df.columns:
            df["margin_sd"] = None

        # Extract context parameters
        ratings = context.get("ratings", {})
        rating_units = context.get("rating_units")
        if rating_units != "points":
            raise ValueError(
                "Elo margin head requires ratings in points units. "
                "Set projection_context['rating_units'] = 'points'."
            )

        home_advantage = float(context.get("home_advantage", 0.0))
        neutral = bool(context.get("neutral", False))
        sport = context.get("sport")
        cfg = get_validation_config(sport)

        raw_margin_sd = context.get("margin_std")
        if raw_margin_sd is None or raw_margin_sd <= 0:
            raw_margin_sd = DEFAULT_MARGIN_SD_FALLBACK

        conditional_sd_intercept = context.get("conditional_sd_intercept")
        conditional_sd_slope = context.get("conditional_sd_slope")

        # Process each row
        for idx, row in df.iterrows():
            home_team = row.get("home_team")
            away_team = row.get("away_team")

            home_rating = ratings.get(home_team)
            away_rating = ratings.get(away_team)

            if home_rating is None or away_rating is None:
                df.at[idx, "margin_mean"] = None
                df.at[idx, "margin_sd"] = None
                continue

            # Compute margin using project_game helper
            margin_neutral = home_rating - away_rating
            margin_with_hfa = margin_neutral + (0.0 if neutral else home_advantage)

            df.at[idx, "margin_mean"] = margin_with_hfa

            # Compute margin SD using conditional model or fallback
            if (
                margin_with_hfa is not None
                and conditional_sd_intercept is not None
                and conditional_sd_slope is not None
            ):
                margin_sd = ConditionalSDModel(
                    intercept=float(conditional_sd_intercept),
                    slope=float(conditional_sd_slope),
                ).predict(
                    margin_with_hfa,
                    guardrail_min=cfg.margin_sd_min,
                    guardrail_max=cfg.margin_sd_max,
                    fallback_sd=LEAGUE_MARGIN_SD_DEFAULT,
                    logger_override=_LOGGER,
                    log_context={
                        "model_id": "elo",
                        "game_id": row.get("game_id"),
                        "date": row.get("game_date"),
                        "home_team": home_team,
                        "away_team": away_team,
                    },
                    debug_assert=False,
                )
            else:
                margin_sd, _ = guardrail_margin_sd(
                    raw_margin_sd,
                    fallback_sd=LEAGUE_MARGIN_SD_DEFAULT,
                    guardrail_min=cfg.margin_sd_min,
                    guardrail_max=cfg.margin_sd_max,
                )

            df.at[idx, "margin_sd"] = margin_sd if margin_with_hfa is not None else None


class EloTotalHead(Head):
    """Derives total_mean and total_sd from Elo learned parameters."""

    @property
    def name(self) -> str:
        return "elo_total_head"

    def produces(self) -> set[str]:
        return {"total_mean", "total_sd"}

    def requires(self) -> set[str]:
        # margin_mean produced by earlier head, plus context fields
        return {"home_team", "away_team"}

    def apply(self, df: pd.DataFrame, context: dict[str, Any]) -> None:
        """
        Derive total_mean and total_sd from Elo model parameters.
        
        Mirrors logic in _rating_projection_engine: tries model-specific total formula,
        then scoring averages, then base_total fallback.
        """
        if "total_mean" not in df.columns:
            df["total_mean"] = None
        if "total_sd" not in df.columns:
            df["total_sd"] = None

        ratings = context.get("ratings", {})
        base_total = float(context.get("base_total", 0.0))
        scoring_averages = context.get("scoring_averages", {})
        total_intercept = context.get("total_intercept")
        total_slope = context.get("total_slope")
        total_std = context.get("total_std")

        if total_std is None or total_std <= 0:
            total_std = DEFAULT_TOTAL_SD_FALLBACK

        for idx, row in df.iterrows():
            home_team = row.get("home_team")
            away_team = row.get("away_team")

            home_rating = ratings.get(home_team)
            away_rating = ratings.get(away_team)

            if home_rating is None or away_rating is None:
                df.at[idx, "total_mean"] = None
                df.at[idx, "total_sd"] = None
                continue

            # Try model-specific formula
            model_total = None
            if total_intercept is not None and total_slope is not None:
                model_total = total_from_ratings(
                    home_team,
                    away_team,
                    ratings,
                    intercept=float(total_intercept),
                    slope=float(total_slope),
                )

            # Try scoring averages
            matchup_total = matchup_total_from_averages(
                home_team, away_team, scoring_averages
            )

            # Select which total to use
            applied_total = model_total or matchup_total or (base_total if base_total > 0 else None)

            df.at[idx, "total_mean"] = applied_total
            df.at[idx, "total_sd"] = total_std if applied_total is not None else None


class EloWinProbHead(Head):
    """Derives p_home_win from margin distribution using logistic curve (Elo-specific)."""

    @property
    def name(self) -> str:
        return "elo_win_prob_head"

    def produces(self) -> set[str]:
        return {
            "p_home_win",
            "projected_win_prob",
            "model_p_home_win",
            "normal_p_home_win",
            "logistic_home_win_prob",
            "win_prob_source",
            "margin_dist_assumption",
        }

    def requires(self) -> set[str]:
        # margin_mean and margin_sd must be available from earlier head
        return {"margin_mean", "margin_sd"}

    def apply(self, df: pd.DataFrame, context: dict[str, Any]) -> None:
        """
        Derive win probabilities from margin distribution.
        
        For Elo:
        - Computes normal CDF from margin (normal_p_home_win)
        - Computes logistic probability using learned win_prob_k and win_prob_bias
        - Uses logistic as the primary p_home_win
        - Mirrors _elo_projection_engine behavior
        """
        for col in self.produces():
            if col not in df.columns:
                df[col] = None

        ratings = context.get("ratings", {})
        win_prob_k = float(context.get("win_prob_k", 6.566641127986305))  # DEFAULT_WIN_PROB_K
        win_prob_bias = float(context.get("win_prob_bias", 0.0))

        for idx, row in df.iterrows():
            margin_mean = row.get("margin_mean")
            margin_sd = row.get("margin_sd")
            home_team = row.get("home_team")
            away_team = row.get("away_team")

            home_rating = ratings.get(home_team)
            away_rating = ratings.get(away_team)

            if (
                margin_mean is None
                or margin_sd is None
                or home_rating is None
                or away_rating is None
            ):
                for col in self.produces():
                    df.at[idx, col] = None
                continue

            # Compute normal CDF-based probability (what margin distribution implies)
            normal_p = _home_win_prob_from_margin(margin_mean, margin_sd)

            # Compute logistic probability: Elo uses this as primary
            # margin_mean is home - away; to get away_minus_home spread for logistic:
            projected_spread = -margin_mean
            # Apply bias adjustment (as in projection engine)
            adjusted_spread = projected_spread - win_prob_bias
            logistic_p = logistic_win_prob(adjusted_spread, win_prob_k)

            # Assign fields: logistic is primary for Elo
            df.at[idx, "normal_p_home_win"] = normal_p
            df.at[idx, "logistic_home_win_prob"] = logistic_p
            df.at[idx, "projected_win_prob"] = logistic_p  # Elo uses logistic
            df.at[idx, "model_p_home_win"] = logistic_p
            df.at[idx, "p_home_win"] = logistic_p  # Canonical field
            df.at[idx, "win_prob_source"] = "logistic"
            df.at[idx, "margin_dist_assumption"] = "normal_approx"


class EloProjectedScoresHead(Head):
    """Derives projected_home_score and projected_away_score from margin and total."""

    @property
    def name(self) -> str:
        return "elo_projected_scores_head"

    def produces(self) -> set[str]:
        return {"projected_home_score", "projected_away_score", "projected_total"}

    def requires(self) -> set[str]:
        return {"margin_mean", "total_mean"}

    def apply(self, df: pd.DataFrame, context: dict[str, Any]) -> None:
        """Compute team scores from total and margin."""
        for col in self.produces():
            if col not in df.columns:
                df[col] = None

        for idx, row in df.iterrows():
            margin_mean = row.get("margin_mean")
            total_mean = row.get("total_mean")

            if margin_mean is None or total_mean is None:
                for col in self.produces():
                    df.at[idx, col] = None
                continue

            projected_home_score = (total_mean + margin_mean) / 2.0
            projected_away_score = (total_mean - margin_mean) / 2.0
            projected_total = projected_home_score + projected_away_score

            df.at[idx, "projected_home_score"] = projected_home_score
            df.at[idx, "projected_away_score"] = projected_away_score
            df.at[idx, "projected_total"] = projected_total


def create_elo_head_sequence() -> HeadSequence:
    """Factory function to create the Elo head sequence."""
    heads = [
        EloMarginHead(),
        EloTotalHead(),
        EloWinProbHead(),
        EloProjectedScoresHead(),
    ]
    return HeadSequence(heads)


# Register Elo heads at module load time
register_model_heads("elo", create_elo_head_sequence)
