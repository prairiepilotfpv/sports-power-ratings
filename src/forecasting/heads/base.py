"""Base protocol for projection heads."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

_LOGGER = logging.getLogger(__name__)


class Head(ABC):
    """
    Abstract base class for projection heads.
    
    A head is a component that derives one or more canonical forecast fields
    from a model's learned parameters and game context. Multiple heads can be
    composed to produce all canonical outputs (p_home_win, margin_mean, margin_sd,
    total_mean, total_sd).
    
    Each head declares:
    - name: identifier (for logging/debugging)
    - produces(): set of field names this head generates
    - requires(): set of field names required in input df (model context)
    - apply(df, context): transforms df in-place by filling produced fields
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this head (e.g., "elo_margin_head").
        
        Used for logging and debugging to trace which heads were applied.
        """
        pass

    @abstractmethod
    def produces(self) -> set[str]:
        """
        Return set of canonical field names this head generates.
        
        Examples:
        - {"margin_mean", "margin_sd"}
        - {"p_home_win"}
        - {"total_mean", "total_sd"}
        - {"projected_home_score", "projected_away_score"}
        
        Returns:
            Set of canonical field names (columns that will be added/filled in df).
        """
        pass

    @abstractmethod
    def requires(self) -> set[str]:
        """
        Return set of context/df fields required to apply this head.
        
        Examples:
        - {"home_rating", "away_rating"} (from projection context)
        - {"margin_mean", "margin_sd"} (intermediate fields produced by earlier heads)
        
        If a required field is missing, apply() should raise ValueError with diagnostic info.
        
        Returns:
            Set of required field names.
        """
        pass

    @abstractmethod
    def apply(
        self,
        df: pd.DataFrame,
        context: dict[str, Any],
    ) -> None:
        """
        Derive produced fields from context and existing df columns.
        
        Modifies df in-place, adding/overwriting columns listed in produces().
        
        Args:
            df: Game forecast DataFrame with columns:
                - home_team, away_team (always present)
                - Any columns from context (ratings, game_date, etc.)
            context: Model-specific calibration/projection context dict with:
                - ratings: {team_name: float_rating}
                - home_advantage: float
                - win_prob_k: float
                - margin_std: float | None
                - total_std: float | None
                - conditional_sd_intercept: float | None
                - conditional_sd_slope: float | None
                - total_intercept: float | None
                - total_slope: float | None
                - base_total: float
                - sport: str (for config/guardrails)
                - (and other model-specific params)
        
        Raises:
            ValueError: If required fields are missing or invalid.
        """
        pass


class HeadSequence:
    """Ordered composition of heads applied to forecasts."""

    def __init__(self, heads: list[Head]) -> None:
        """
        Initialize with an ordered list of heads.
        
        Heads are applied in order; earlier heads' produces() may be required
        by later heads (via requires()).
        
        Args:
            heads: List of Head instances.
        """
        self.heads = heads

    def apply(self, df: pd.DataFrame, context: dict[str, Any]) -> dict[str, list[str]]:
        """
        Apply all heads in sequence, validating dependencies.
        
        Args:
            df: Game forecast DataFrame.
            context: Model projection context.
        
        Returns:
            Dict with keys:
                - "applied_heads": list of head names in order applied
                - "filled_fields": list of fields that were added/modified
        
        Raises:
            ValueError: If any head's requires() are not satisfied.
        """
        applied_heads = []
        filled_fields = []

        for head in self.heads:
            required = head.requires()
            available = set(df.columns) | set(context.keys())

            missing = required - available
            if missing:
                raise ValueError(
                    f"Head '{head.name}' requires missing fields: {missing}. "
                    f"Available: {available}."
                )

            head.apply(df, context)
            applied_heads.append(head.name)
            filled_fields.extend(head.produces())

            if _LOGGER.isEnabledFor(logging.DEBUG):
                _LOGGER.debug(
                    f"[heads] applied head={head.name} produces={head.produces()}",
                    extra={
                        "head_name": head.name,
                        "produced_fields": list(head.produces()),
                    },
                )

        return {
            "applied_heads": applied_heads,
            "filled_fields": filled_fields,
        }
