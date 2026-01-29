"""Projection heads for Poisson model."""

from __future__ import annotations

import logging
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import skellam

from config import (
    DEFAULT_MARGIN_SD_FALLBACK,
    DEFAULT_TOTAL_SD_FALLBACK,
    DEFAULT_POISSON_OVERDISPERSION,
)
from forecasting.heads.base import Head, HeadSequence
from forecasting.heads.registry import register_model_heads
from models.base import _home_win_prob_from_margin

_LOGGER = logging.getLogger(__name__)


def _clip_prob(prob: float, eps: float = 1e-6) -> float:
    """Clip probability to valid range [eps, 1-eps]."""
    return float(min(max(prob, eps), 1.0 - eps))


def _skellam_home_win_prob(
    lambda_home: float,
    lambda_away: float,
    *,
    fallback_sd: float,
    tie_split_home: float = 0.5,
) -> float:
    """Compute home win probability using Skellam distribution.
    
    Args:
        lambda_home: Expected home goals (Poisson rate)
        lambda_away: Expected away goals (Poisson rate)
        fallback_sd: Fallback standard deviation for normal approximation
        tie_split_home: Fraction of tie probability assigned to home team.
    
    Returns:
        Home win probability (clipped to avoid extreme values)
    """
    try:
        pmf_zero = float(skellam.pmf(0, lambda_home, lambda_away))
        sf_zero = float(skellam.sf(0, lambda_home, lambda_away))
        prob = sf_zero + tie_split_home * pmf_zero
    except Exception:
        prob = _home_win_prob_from_margin(
            lambda_home - lambda_away,
            fallback_sd if fallback_sd > 0 else DEFAULT_MARGIN_SD_FALLBACK,
        )
    return _clip_prob(prob)


class PoissonMarginHead(Head):
    """Derives margin_mean and margin_sd from Poisson attack/defense ratings.
    
    Poisson model produces expected goals for home and away teams.
    margin_mean = lambda_home - lambda_away
    margin_sd derived from the variance of the difference distribution.
    """

    @property
    def name(self) -> str:
        return "poisson_margin_head"

    def produces(self) -> set[str]:
        return {"margin_mean", "margin_sd"}

    def requires(self) -> set[str]:
        return {"home_team", "away_team"}

    def apply(self, df: pd.DataFrame, context: dict[str, Any]) -> None:
        """
        Derive margin_mean and margin_sd from Poisson attack/defense ratings.
        
        Context contains:
        - attack: dict {team: attack_strength} or numpy array indexed by team_index
        - defense: dict {team: defense_strength} or numpy array indexed by team_index
        - mu: global strength scale
        - home_advantage: home advantage in log scale
        - kappa: overdispersion factor
        - team_index: {team_name: index} for array indexing (optional)
        """
        if "margin_mean" not in df.columns:
            df["margin_mean"] = None
        if "margin_sd" not in df.columns:
            df["margin_sd"] = None

        attack_raw = context.get("attack", {})
        defense_raw = context.get("defense", {})
        team_index = context.get("team_index", {})
        mu = float(context.get("mu", 0.0))
        home_advantage = float(context.get("home_advantage", 0.0))
        kappa = float(context.get("kappa", DEFAULT_POISSON_OVERDISPERSION))

        # Convert numpy arrays to dicts if needed
        attack = attack_raw if isinstance(attack_raw, dict) else {}
        defense = defense_raw if isinstance(defense_raw, dict) else {}
        
        if isinstance(attack_raw, np.ndarray) and team_index:
            reverse_index = {idx: team for team, idx in team_index.items()}
            for idx, val in enumerate(attack_raw):
                team = reverse_index.get(idx)
                if team:
                    attack[team] = float(val)
        
        if isinstance(defense_raw, np.ndarray) and team_index:
            reverse_index = {idx: team for team, idx in team_index.items()}
            for idx, val in enumerate(defense_raw):
                team = reverse_index.get(idx)
                if team:
                    defense[team] = float(val)

        # Process each row
        for idx, row in df.iterrows():
            home_team = row.get("home_team")
            away_team = row.get("away_team")

            home_attack = attack.get(home_team)
            home_defense = defense.get(home_team)
            away_attack = attack.get(away_team)
            away_defense = defense.get(away_team)

            if (
                home_attack is None or home_defense is None
                or away_attack is None or away_defense is None
            ):
                df.at[idx, "margin_mean"] = None
                df.at[idx, "margin_sd"] = None
                continue

            # Expected goals
            neutral = bool(row.get("neutral", False))
            hfa_contrib = 0.0 if neutral else home_advantage

            lambda_home = np.exp(mu + hfa_contrib + home_attack - away_defense)
            lambda_away = np.exp(mu + away_attack - home_defense)

            # Margin distribution
            margin_mean = float(lambda_home - lambda_away)

            # Variance: Poisson sum variance is sum of rates, then apply overdispersion
            # For margin (difference of independent Poisson): Var(H-A) = Var(H) + Var(A) = lambda_H + lambda_A (under Poisson)
            # With overdispersion: Var = kappa * (lambda_H + lambda_A)
            total_rate = float(lambda_home + lambda_away)
            variance = float(max(kappa * total_rate, 0.0))
            margin_sd_raw = float(sqrt(variance)) if variance >= 0 else 0.0

            # Apply guardrails
            if margin_sd_raw <= 0:
                margin_sd = DEFAULT_MARGIN_SD_FALLBACK
            else:
                margin_sd = margin_sd_raw

            df.at[idx, "margin_mean"] = margin_mean
            df.at[idx, "margin_sd"] = margin_sd


class PoissonTotalHead(Head):
    """Derives total_mean and total_sd from Poisson expected goals.
    
    total_mean = lambda_home + lambda_away
    total_sd derived from the variance of the sum distribution.
    """

    @property
    def name(self) -> str:
        return "poisson_total_head"

    def produces(self) -> set[str]:
        return {"total_mean", "total_sd"}

    def requires(self) -> set[str]:
        return {"home_team", "away_team"}

    def apply(self, df: pd.DataFrame, context: dict[str, Any]) -> None:
        """
        Derive total_mean and total_sd from Poisson expected goals.
        
        total_mean = lambda_home + lambda_away
        total_sd from overdispersed variance
        """
        if "total_mean" not in df.columns:
            df["total_mean"] = None
        if "total_sd" not in df.columns:
            df["total_sd"] = None

        attack_raw = context.get("attack", {})
        defense_raw = context.get("defense", {})
        team_index = context.get("team_index", {})
        mu = float(context.get("mu", 0.0))
        home_advantage = float(context.get("home_advantage", 0.0))
        kappa = float(context.get("kappa", DEFAULT_POISSON_OVERDISPERSION))

        # Convert numpy arrays to dicts if needed
        attack = attack_raw if isinstance(attack_raw, dict) else {}
        defense = defense_raw if isinstance(defense_raw, dict) else {}
        
        if isinstance(attack_raw, np.ndarray) and team_index:
            reverse_index = {idx: team for team, idx in team_index.items()}
            for idx, val in enumerate(attack_raw):
                team = reverse_index.get(idx)
                if team:
                    attack[team] = float(val)
        
        if isinstance(defense_raw, np.ndarray) and team_index:
            reverse_index = {idx: team for team, idx in team_index.items()}
            for idx, val in enumerate(defense_raw):
                team = reverse_index.get(idx)
                if team:
                    defense[team] = float(val)

        for idx, row in df.iterrows():
            home_team = row.get("home_team")
            away_team = row.get("away_team")

            home_attack = attack.get(home_team)
            home_defense = defense.get(home_team)
            away_attack = attack.get(away_team)
            away_defense = defense.get(away_team)

            if (
                home_attack is None or home_defense is None
                or away_attack is None or away_defense is None
            ):
                df.at[idx, "total_mean"] = None
                df.at[idx, "total_sd"] = None
                continue

            # Expected goals
            neutral = bool(row.get("neutral", False))
            hfa_contrib = 0.0 if neutral else home_advantage

            lambda_home = np.exp(mu + hfa_contrib + home_attack - away_defense)
            lambda_away = np.exp(mu + away_attack - home_defense)

            # Total distribution
            total_mean = float(lambda_home + lambda_away)

            # Variance: for sum of independent Poisson with overdispersion
            # Var(H+A) = Var(H) + Var(A) = kappa * (lambda_H + lambda_A)
            variance = float(max(kappa * total_mean, 0.0))
            total_sd_raw = float(sqrt(variance)) if variance >= 0 else 0.0

            # Apply guardrails
            if total_sd_raw <= 0:
                total_sd = DEFAULT_TOTAL_SD_FALLBACK
            else:
                total_sd = total_sd_raw

            df.at[idx, "total_mean"] = total_mean
            df.at[idx, "total_sd"] = total_sd


class PoissonWinProbHead(Head):
    """Derives p_home_win from Poisson distribution using Skellam.
    
    Win probability comes from the Skellam distribution (difference of Poisson).
    p_home_win = P(lambda_home > lambda_away)
    """

    @property
    def name(self) -> str:
        return "poisson_win_prob_head"

    def produces(self) -> set[str]:
        return {
            "p_home_win",
            "projected_win_prob",
            "model_p_home_win",
            "normal_p_home_win",
            "win_prob_source",
            "margin_dist_assumption",
        }

    def requires(self) -> set[str]:
        # margin_mean and margin_sd should be available from PoissonMarginHead
        return {"margin_mean", "margin_sd"}

    def apply(self, df: pd.DataFrame, context: dict[str, Any]) -> None:
        """
        Derive win probabilities from Poisson distribution using Skellam.
        
        p_home_win = P(home_score > away_score) using Skellam distribution
        """
        for col in self.produces():
            if col not in df.columns:
                df[col] = None

        attack_raw = context.get("attack", {})
        defense_raw = context.get("defense", {})
        team_index = context.get("team_index", {})
        mu = float(context.get("mu", 0.0))
        home_advantage = float(context.get("home_advantage", 0.0))
        tie_split_home = float(context.get("tie_split_home", 0.5))

        # Convert numpy arrays to dicts if needed
        attack = attack_raw if isinstance(attack_raw, dict) else {}
        defense = defense_raw if isinstance(defense_raw, dict) else {}
        
        if isinstance(attack_raw, np.ndarray) and team_index:
            reverse_index = {idx: team for team, idx in team_index.items()}
            for idx, val in enumerate(attack_raw):
                team = reverse_index.get(idx)
                if team:
                    attack[team] = float(val)
        
        if isinstance(defense_raw, np.ndarray) and team_index:
            reverse_index = {idx: team for team, idx in team_index.items()}
            for idx, val in enumerate(defense_raw):
                team = reverse_index.get(idx)
                if team:
                    defense[team] = float(val)

        for idx, row in df.iterrows():
            home_team = row.get("home_team")
            away_team = row.get("away_team")
            margin_mean = row.get("margin_mean")
            margin_sd = row.get("margin_sd")

            home_attack = attack.get(home_team)
            home_defense = defense.get(home_team)
            away_attack = attack.get(away_team)
            away_defense = defense.get(away_team)

            if (
                home_attack is None or home_defense is None
                or away_attack is None or away_defense is None
                or margin_mean is None
                or margin_sd is None
            ):
                for col in self.produces():
                    df.at[idx, col] = None
                continue

            # Expected goals (same as in margin head)
            neutral = bool(row.get("neutral", False))
            hfa_contrib = 0.0 if neutral else home_advantage

            lambda_home = np.exp(mu + hfa_contrib + home_attack - away_defense)
            lambda_away = np.exp(mu + away_attack - home_defense)

            # Skellam-based probability (primary)
            skellam_p = _skellam_home_win_prob(
                lambda_home, lambda_away,
                fallback_sd=margin_sd,
                tie_split_home=tie_split_home,
            )

            # Normal CDF from margin distribution (for comparison)
            normal_p = _home_win_prob_from_margin(margin_mean, margin_sd)

            # Assign fields: Skellam is primary for Poisson
            df.at[idx, "normal_p_home_win"] = normal_p
            df.at[idx, "model_p_home_win"] = skellam_p
            df.at[idx, "projected_win_prob"] = skellam_p
            df.at[idx, "p_home_win"] = skellam_p  # Canonical field
            df.at[idx, "win_prob_source"] = "poisson_skellam"
            df.at[idx, "margin_dist_assumption"] = "skellam"


class PoissonProjectedScoresHead(Head):
    """Derives projected_home_score and projected_away_score from margin and total."""

    @property
    def name(self) -> str:
        return "poisson_projected_scores_head"

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


def create_poisson_head_sequence() -> HeadSequence:
    """Factory function to create the Poisson head sequence."""
    heads = [
        PoissonMarginHead(),
        PoissonTotalHead(),
        PoissonWinProbHead(),
        PoissonProjectedScoresHead(),
    ]
    return HeadSequence(heads)


# Register Poisson heads at module load time
register_model_heads("poisson", create_poisson_head_sequence)
