"""Projection heads for Bradley-Terry model."""

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
from models.calibration import guardrail_margin_sd
from eval.validation import get_validation_config

_LOGGER = logging.getLogger(__name__)


class BtMarginHead(Head):
    """Derives margin_mean and margin_sd from Bradley-Terry ratings.
    
    Uses Thurstone-Mosteller interpretation:
    - margin_mean = strength_diff + home_advantage
    - margin_sd from fit calibration (margin_sigma)
    """

    @property
    def name(self) -> str:
        return "bt_margin_head"

    def produces(self) -> set[str]:
        return {"margin_mean", "margin_sd"}

    def requires(self) -> set[str]:
        return {"home_team", "away_team"}

    def apply(self, df: pd.DataFrame, context: dict[str, Any]) -> None:
        """
        Derive margin_mean and margin_sd from Bradley-Terry parameters.
        
        Bradley-Terry learns:
        - Strength ratings for each team
        - Home-field advantage (hfa_logit)
        - Calibration.margin_sigma (empirical noise from fit)
        
        margin_mean = (home_strength - away_strength) + hfa (if not neutral)
        margin_sd = margin_sigma (with guardrails)
        """
        if "margin_mean" not in df.columns:
            df["margin_mean"] = None
        if "margin_sd" not in df.columns:
            df["margin_sd"] = None

        # Extract context parameters
        ratings = context.get("ratings", {})
        hfa_logit = float(context.get("home_advantage", 0.0))
        sport = context.get("sport")
        cfg = get_validation_config(sport)

        # Calibration margin_sigma (learned during fit)
        margin_sigma = context.get("margin_sigma")
        if margin_sigma is None or margin_sigma <= 0:
            margin_sigma = DEFAULT_MARGIN_SD_FALLBACK

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

            # Margin = strength difference + home advantage
            neutral = bool(row.get("neutral", False))
            margin_neutral = home_rating - away_rating
            margin_with_hfa = margin_neutral + (0.0 if neutral else hfa_logit)

            df.at[idx, "margin_mean"] = margin_with_hfa

            # Apply guardrails to margin_sd
            margin_sd, _ = guardrail_margin_sd(
                margin_sigma,
                fallback_sd=LEAGUE_MARGIN_SD_DEFAULT,
                guardrail_min=cfg.margin_sd_min,
                guardrail_max=cfg.margin_sd_max,
            )

            df.at[idx, "margin_sd"] = margin_sd if margin_with_hfa is not None else None


class BtWinProbHead(Head):
    """Derives p_home_win from Bradley-Terry ratings using logistic link.
    
    Bradley-Terry is fundamentally a logistic model:
    p_home_win = sigmoid(strength_diff + hfa)
    where sigmoid is the logistic function: 1 / (1 + exp(-x))
    """

    @property
    def name(self) -> str:
        return "bt_win_prob_head"

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
        # margin_mean and margin_sd should be available from BtMarginHead
        return {"margin_mean", "margin_sd"}

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid."""
        if x >= 0:
            z = np.exp(-x)
            return float(1.0 / (1.0 + z))
        z = np.exp(x)
        return float(z / (1.0 + z))

    def apply(self, df: pd.DataFrame, context: dict[str, Any]) -> None:
        """
        Derive win probabilities from Bradley-Terry formulation.
        
        Bradley-Terry uses logistic: p = sigmoid(strength_diff + hfa)
        We also compute normal CDF from margin distribution for comparison.
        """
        for col in self.produces():
            if col not in df.columns:
                df[col] = None

        ratings = context.get("ratings", {})
        hfa_logit = float(context.get("home_advantage", 0.0))
        temp = float(context.get("temp", 1.0))

        for idx, row in df.iterrows():
            home_team = row.get("home_team")
            away_team = row.get("away_team")
            margin_mean = row.get("margin_mean")
            margin_sd = row.get("margin_sd")

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

            # Logistic probability: the Bradley-Terry canonical formulation
            neutral = bool(row.get("neutral", False))
            score = home_rating - away_rating
            if not neutral:
                score += hfa_logit
            if temp != 0:
                score = score / temp
            logistic_p = self._sigmoid(score)

            # Normal CDF from margin distribution (for comparison)
            normal_p = _home_win_prob_from_margin(margin_mean, margin_sd)

            # Assign fields: logistic is primary for Bradley-Terry
            df.at[idx, "normal_p_home_win"] = normal_p
            df.at[idx, "logistic_home_win_prob"] = logistic_p
            df.at[idx, "projected_win_prob"] = logistic_p
            df.at[idx, "model_p_home_win"] = logistic_p
            df.at[idx, "p_home_win"] = logistic_p  # Canonical field
            df.at[idx, "win_prob_source"] = "logistic"
            df.at[idx, "margin_dist_assumption"] = "normal_approx"


class BtTotalHead(Head):
    """Derives total_mean and total_sd from Bradley-Terry calibration.
    
    Bradley-Terry fits a linear regression of total on |d_value| during calibration:
    total_mean = total_c + total_u * |d_value|
    total_sd from residuals (with guardrails)
    """

    @property
    def name(self) -> str:
        return "bt_total_head"

    def produces(self) -> set[str]:
        return {"total_mean", "total_sd"}

    def requires(self) -> set[str]:
        return {"home_team", "away_team"}

    def apply(self, df: pd.DataFrame, context: dict[str, Any]) -> None:
        """
        Derive total_mean and total_sd from Bradley-Terry calibration parameters.
        
        BT calibration fits:
        - total_mean = total_c + total_u * |d_value|
        - total_sd from residuals with guardrails
        """
        if "total_mean" not in df.columns:
            df["total_mean"] = None
        if "total_sd" not in df.columns:
            df["total_sd"] = None

        ratings = context.get("ratings", {})
        hfa_logit = float(context.get("home_advantage", 0.0))

        # Calibration coefficients
        total_c = float(context.get("total_c", 0.0))
        total_u = float(context.get("total_u", 0.0))
        total_sigma = context.get("total_sigma")

        if total_sigma is None or total_sigma <= 0:
            total_sigma = DEFAULT_TOTAL_SD_FALLBACK

        for idx, row in df.iterrows():
            home_team = row.get("home_team")
            away_team = row.get("away_team")

            home_rating = ratings.get(home_team)
            away_rating = ratings.get(away_team)

            if home_rating is None or away_rating is None:
                df.at[idx, "total_mean"] = None
                df.at[idx, "total_sd"] = None
                continue

            # d_value used in calibration
            d_value = home_rating - away_rating
            neutral = bool(row.get("neutral", False))
            if not neutral:
                d_value += hfa_logit

            # Total from regression
            total_mean = total_c + total_u * abs(d_value)

            df.at[idx, "total_mean"] = total_mean
            df.at[idx, "total_sd"] = total_sigma


class BtProjectedScoresHead(Head):
    """Derives projected_home_score and projected_away_score from margin and total."""

    @property
    def name(self) -> str:
        return "bt_projected_scores_head"

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


def create_bradley_terry_head_sequence() -> HeadSequence:
    """Factory function to create the Bradley-Terry head sequence."""
    heads = [
        BtMarginHead(),
        BtTotalHead(),
        BtWinProbHead(),
        BtProjectedScoresHead(),
    ]
    return HeadSequence(heads)


# Register Bradley-Terry heads at module load time
register_model_heads("bradley-terry", create_bradley_terry_head_sequence)
