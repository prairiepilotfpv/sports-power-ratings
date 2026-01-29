"""Projection heads for GSSD model."""

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


class GssdMarginHead(Head):
    """Derives margin_mean and margin_sd from GSSD coefficients and team stats."""

    @property
    def name(self) -> str:
        return "gssd_margin_head"

    def produces(self) -> set[str]:
        return {"margin_mean", "margin_sd"}

    def requires(self) -> set[str]:
        # These come from context, not the dataframe
        return {"home_team", "away_team"}

    def apply(self, df: pd.DataFrame, context: dict[str, Any]) -> None:
        """
        Derive margin_mean and margin_sd from GSSD coefficients and team stats.
        
        GSSD margin = intercept + beta_pfh*pfh + beta_pah*pah + beta_pfa*pfa + beta_paa*paa + home_advantage*(0 if neutral else 1)
        margin_sd is either from conditional SD model or the error_term with guardrails.
        """
        if "margin_mean" not in df.columns:
            df["margin_mean"] = None
        if "margin_sd" not in df.columns:
            df["margin_sd"] = None

        # Extract context parameters
        sport = context.get("sport")
        cfg = get_validation_config(sport)

        # GSSD coefficients
        intercept = float(context.get("intercept", 0.0))
        beta_pfh = float(context.get("beta_pfh", 0.0))
        beta_pah = float(context.get("beta_pah", 0.0))
        beta_pfa = float(context.get("beta_pfa", 0.0))
        beta_paa = float(context.get("beta_paa", 0.0))
        home_advantage_points = float(context.get("home_advantage_points", 0.0))
        error_term = float(context.get("error_term", 0.0))

        # Conditional SD model (optional)
        conditional_sd_intercept = context.get("conditional_sd_intercept")
        conditional_sd_slope = context.get("conditional_sd_slope")

        # Raw margin SD fallback
        raw_margin_sd = error_term if error_term > 0 else DEFAULT_MARGIN_SD_FALLBACK
        
        # Team stats dictionary (from context, keyed by team name)
        team_stats = context.get("team_stats", {})

        # Process each row
        for idx, row in df.iterrows():
            home_team = row.get("home_team")
            away_team = row.get("away_team")

            home_info = team_stats.get(home_team)
            away_info = team_stats.get(away_team)

            if home_info is None or away_info is None:
                df.at[idx, "margin_mean"] = None
                df.at[idx, "margin_sd"] = None
                continue

            # Extract team stats
            home_pfh = float(home_info.get("pfh", 0.0))
            home_pah = float(home_info.get("pah", 0.0))
            away_pfa = float(away_info.get("pfa", 0.0))
            away_paa = float(away_info.get("paa", 0.0))

            # Compute margin: home_advantage applies only at non-neutral venues
            neutral = bool(row.get("neutral", False))
            neutral_contrib = 0.0 if neutral else home_advantage_points
            margin_with_coeff = (
                intercept
                + beta_pfh * home_pfh
                + beta_pah * home_pah
                + beta_pfa * away_pfa
                + beta_paa * away_paa
                + neutral_contrib
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
                        "model_id": "gssd",
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

            # Strict check: margin_sd must be finite and positive
            if not np.isfinite(margin_sd) or margin_sd <= 0:
                game_id = row.get("game_id", f"{row.get('game_date')}|{away_team}|{home_team}")
                raise RuntimeError(
                    f"GSSD margin_sd guardrail produced invalid value: "
                    f"sd={margin_sd} (finite={np.isfinite(margin_sd)}, positive={margin_sd > 0}); "
                    f"game_id={game_id}, raw_sd={raw_margin_sd}, margin_mean={margin_with_coeff}"
                )

            df.at[idx, "margin_sd"] = margin_sd


class GssdTotalHead(Head):
    """Derives total_mean and total_sd from GSSD learned parameters."""

    @property
    def name(self) -> str:
        return "gssd_total_head"

    def produces(self) -> set[str]:
        return {"total_mean", "total_sd"}

    def requires(self) -> set[str]:
        return {"home_team", "away_team"}

    def apply(self, df: pd.DataFrame, context: dict[str, Any]) -> None:
        """
        Derive total_mean and total_sd from GSSD model parameters.
        
        GSSD uses league-average total (from fit) or falls back to defaults.
        If available, also computes team-adjusted total using pfh + paa and pfa + pah average.
        """
        if "total_mean" not in df.columns:
            df["total_mean"] = None
        if "total_sd" not in df.columns:
            df["total_sd"] = None

        # Extract GSSD total parameters
        total_mean = context.get("total_mean")
        total_sd = context.get("total_sd")
        team_stats = context.get("team_stats", {})

        # Fallbacks
        if total_mean is None or total_mean <= 0:
            total_mean = DEFAULT_TOTAL_MEAN_FALLBACK
        if total_sd is None or total_sd <= 0:
            total_sd = DEFAULT_TOTAL_SD_FALLBACK

        # Apply to all rows
        for idx, row in df.iterrows():
            home_team = row.get("home_team")
            away_team = row.get("away_team")

            # Try to compute team-adjusted total
            home_info = team_stats.get(home_team)
            away_info = team_stats.get(away_team)
            
            if home_info is not None and away_info is not None:
                # GSSD's canonical total derivation: average of team scoring contexts
                home_pfh = float(home_info.get("pfh", 0.0))
                away_paa = float(away_info.get("paa", 0.0))
                away_pfa = float(away_info.get("pfa", 0.0))
                home_pah = float(home_info.get("pah", 0.0))
                team_adjusted_total = (home_pfh + away_paa + away_pfa + home_pah) / 2.0
                df.at[idx, "total_mean"] = team_adjusted_total
            else:
                df.at[idx, "total_mean"] = total_mean

            df.at[idx, "total_sd"] = total_sd


class GssdWinProbHead(Head):
    """Derives p_home_win from margin distribution using logistic curve (GSSD-specific)."""

    @property
    def name(self) -> str:
        return "gssd_win_prob_head"

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
        
        For GSSD:
        - Uses logistic probability with learned win_prob_k and winprob_bias (like TOOR)
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
            game_id = row.get("game_id", f"{row.get('game_date')}|{row.get('away_team')}|{row.get('home_team')}")

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

            # Compute normal CDF (for reference)
            normal_prob = _home_win_prob_from_margin(margin_mean, margin_sd)

            # Strict check: p_home_win must be finite and in [0, 1]
            if not np.isfinite(logistic_prob) or logistic_prob < 0 or logistic_prob > 1:
                raise RuntimeError(
                    f"GSSD p_home_win produced invalid value: "
                    f"p_home_win={logistic_prob} (finite={np.isfinite(logistic_prob)}, in [0,1]={0 <= logistic_prob <= 1}); "
                    f"game_id={game_id}, margin_mean={margin_mean}, margin_sd={margin_sd}, "
                    f"adjusted_spread={adjusted_spread}, win_prob_k={win_prob_k}"
                )

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


class GssdScoresHead(Head):
    """Derives projected home/away scores from margin and total (internal use)."""

    @property
    def name(self) -> str:
        return "gssd_scores_head"

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


def create_gssd_head_sequence() -> HeadSequence:
    """Factory for GSSD head sequence in canonical field derivation order.
    
    Order matters: margins must come before win probs, totals can be independent.
    Scores come last since they depend on both margin and total.
    """
    return HeadSequence(
        heads=[
            GssdMarginHead(),
            GssdTotalHead(),
            GssdWinProbHead(),
            GssdScoresHead(),
        ]
    )


# Register GSSD heads in the global registry
register_model_heads("gssd", create_gssd_head_sequence)
