"""Projection heads for TOOR model."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from config import (
    DEFAULT_MARGIN_SD_FALLBACK,
    DEFAULT_TOTAL_MEAN_FALLBACK,
    DEFAULT_TOTAL_SD_FALLBACK,
    DEFAULT_WIN_PROB_K,
    LEAGUE_MARGIN_SD_DEFAULT,
    MARGIN_SD_GUARDRAIL_MAX,
    MARGIN_SD_GUARDRAIL_MIN,
)
from forecasting.heads.base import Head, HeadSequence
from forecasting.heads.registry import register_model_heads
from models.base import _home_win_prob_from_margin
from models.calibration import (
    ConditionalSDModel,
    align_spread_with_margin,
    guardrail_margin_sd,
)
from pipelines.projections import (
    logistic_win_prob,
    win_prob_distribution,
)
from eval.validation import get_validation_config

_LOGGER = logging.getLogger(__name__)


class ToorMarginHead(Head):
    """Derives margin_mean and margin_sd from TOOR ratings and coefficients."""

    @property
    def name(self) -> str:
        return "toor_margin_head"

    def produces(self) -> set[str]:
        return {"margin_mean", "margin_sd"}

    def requires(self) -> set[str]:
        # These come from context, not the dataframe
        return {"home_team", "away_team"}

    def apply(self, df: pd.DataFrame, context: dict[str, Any]) -> None:
        """
        Derive margin_mean and margin_sd from TOOR coefficients and team strengths.
        
        TOOR uses:
        - margin = home_advantage * (1 if not neutral else 0) + home_coeff * home_strength + away_coeff * away_strength
        - margin_sd is either from conditional SD model or the error_term with guardrails
        """
        if "margin_mean" not in df.columns:
            df["margin_mean"] = None
        if "margin_sd" not in df.columns:
            df["margin_sd"] = None

        # Extract context parameters
        strengths = context.get("signed_strengths", {})
        sport = context.get("sport")
        cfg = get_validation_config(sport)

        # TOOR coefficients
        home_advantage = float(context.get("home_advantage", 3.362))
        home_coeff = float(context.get("home_coeff", 17.373))
        away_coeff = float(context.get("away_coeff", -14.855))
        error_term = float(context.get("error_term", 31.155))

        # Conditional SD model (optional)
        conditional_sd_intercept = context.get("conditional_sd_intercept")
        conditional_sd_slope = context.get("conditional_sd_slope")

        # Raw margin SD fallback
        raw_margin_sd = context.get("margin_std")
        if raw_margin_sd is None or raw_margin_sd <= 0:
            raw_margin_sd = error_term if error_term > 0 else DEFAULT_MARGIN_SD_FALLBACK

        # Process each row
        for idx, row in df.iterrows():
            home_team = row.get("home_team")
            away_team = row.get("away_team")

            home_strength = strengths.get(home_team)
            away_strength = strengths.get(away_team)

            if home_strength is None or away_strength is None:
                df.at[idx, "margin_mean"] = None
                df.at[idx, "margin_sd"] = None
                continue

            # Compute margin: home_advantage applies only at non-neutral venues
            neutral = bool(row.get("neutral", False))
            neutral_contrib = 0.0 if neutral else home_advantage
            margin_with_coeff = (
                neutral_contrib
                + home_coeff * home_strength
                + away_coeff * away_strength
            )

            df.at[idx, "margin_mean"] = margin_with_coeff

            # Compute margin SD using conditional model or fallback
            if (
                margin_with_coeff is not None
                and conditional_sd_intercept is not None
                and conditional_sd_slope is not None
            ):
                margin_sd = ConditionalSDModel(
                    intercept=float(conditional_sd_intercept),
                    slope=float(conditional_sd_slope),
                ).predict(
                    margin_with_coeff,
                    guardrail_min=cfg.margin_sd_min,
                    guardrail_max=cfg.margin_sd_max,
                    fallback_sd=LEAGUE_MARGIN_SD_DEFAULT,
                    logger_override=_LOGGER,
                    log_context={
                        "model_id": "toor",
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

            df.at[idx, "margin_sd"] = margin_sd if margin_with_coeff is not None else None


class ToorTotalHead(Head):
    """Derives total_mean and total_sd from TOOR learned parameters."""

    @property
    def name(self) -> str:
        return "toor_total_head"

    def produces(self) -> set[str]:
        return {"total_mean", "total_sd"}

    def requires(self) -> set[str]:
        return {"home_team", "away_team"}

    def apply(self, df: pd.DataFrame, context: dict[str, Any]) -> None:
        """
        Derive total_mean and total_sd from TOOR model parameters.
        
        TOOR uses league-average total (from fit) or falls back to defaults.
        """
        if "total_mean" not in df.columns:
            df["total_mean"] = None
        if "total_sd" not in df.columns:
            df["total_sd"] = None

        # Extract TOOR total parameters
        total_mean = context.get("total_mean")
        total_sd = context.get("total_sd")

        # Fallbacks
        if total_mean is None or total_mean <= 0:
            total_mean = DEFAULT_TOTAL_MEAN_FALLBACK
        if total_sd is None or total_sd <= 0:
            total_sd = DEFAULT_TOTAL_SD_FALLBACK

        # Apply to all rows (TOOR uses league average)
        for idx in range(len(df)):
            df.at[idx, "total_mean"] = total_mean
            df.at[idx, "total_sd"] = total_sd


class ToorWinProbHead(Head):
    """Derives p_home_win from margin distribution using logistic curve (TOOR-specific)."""

    @property
    def name(self) -> str:
        return "toor_win_prob_head"

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
        Derive win probabilities from margin distribution using logistic curve.
        
        For TOOR:
        - Uses logistic probability with learned win_prob_k and winprob_bias
        - Derives from -margin (as spread) aligned with spread convention
        """
        for col in self.produces():
            if col not in df.columns:
                df[col] = None

        win_prob_k = float(context.get("win_prob_k", DEFAULT_WIN_PROB_K))
        winprob_bias = float(context.get("winprob_bias", 0.0))

        if win_prob_k <= 0:
            win_prob_k = DEFAULT_WIN_PROB_K

        for idx, row in df.iterrows():
            margin_mean = row.get("margin_mean")
            margin_sd = row.get("margin_sd")

            if margin_mean is None or margin_sd is None:
                for col in self.produces():
                    df.at[idx, col] = None
                continue

            # Convert margin to spread (negative sign convention)
            projected_spread = -float(margin_mean)
            
            # Apply win_prob_bias alignment
            adjusted_spread = align_spread_with_margin(
                margin_mean, projected_spread - winprob_bias
            )
            
            # Compute logistic probability
            logistic_prob = logistic_win_prob(adjusted_spread, win_prob_k)

            # Compute normal CDF (for reference, though TOOR primarily uses logistic)
            normal_prob = _home_win_prob_from_margin(margin_mean, margin_sd)

            # Derive win prob distribution
            win_prob_dist = win_prob_distribution(
                logistic_prob,
                win_prob_k=win_prob_k,
                margin_std=margin_sd,
            )

            # Fill all win prob fields
            df.at[idx, "p_home_win"] = logistic_prob
            df.at[idx, "projected_win_prob"] = logistic_prob
            df.at[idx, "model_p_home_win"] = logistic_prob
            df.at[idx, "normal_p_home_win"] = normal_prob
            df.at[idx, "logistic_home_win_prob"] = logistic_prob
            df.at[idx, "win_prob_source"] = "logistic"
            df.at[idx, "margin_dist_assumption"] = "normal_approx"


class ToorScoresHead(Head):
    """Derives projected home/away scores from margin and total (internal use)."""

    @property
    def name(self) -> str:
        return "toor_scores_head"

    def produces(self) -> set[str]:
        return {
            "projected_home_score",
            "projected_away_score",
            "projected_total",
        }

    def requires(self) -> set[str]:
        return {"margin_mean", "total_mean"}

    def apply(self, df: pd.DataFrame, context: dict[str, Any]) -> None:
        """
        Derive projected scores from margin and total.
        
        home_score = (total + margin) / 2
        away_score = (total - margin) / 2
        """
        for col in self.produces():
            if col not in df.columns:
                df[col] = None

        for idx, row in df.iterrows():
            margin_mean = row.get("margin_mean")
            total_mean = row.get("total_mean")

            if margin_mean is None or total_mean is None:
                df.at[idx, "projected_home_score"] = None
                df.at[idx, "projected_away_score"] = None
                df.at[idx, "projected_total"] = total_mean
                continue

            projected_home = 0.5 * (total_mean + margin_mean)
            projected_away = 0.5 * (total_mean - margin_mean)

            df.at[idx, "projected_home_score"] = projected_home
            df.at[idx, "projected_away_score"] = projected_away
            df.at[idx, "projected_total"] = total_mean


def create_toor_head_sequence() -> HeadSequence:
    """Factory for TOOR head sequence in canonical field derivation order.
    
    Order matters: margins must come before win probs, totals independent.
    Scores come last since they depend on both margin and total.
    """
    return HeadSequence(
        heads=[
            ToorMarginHead(),
            ToorTotalHead(),
            ToorWinProbHead(),
            ToorScoresHead(),
        ]
    )


# Register TOOR heads in the global registry
register_model_heads("toor", create_toor_head_sequence)
