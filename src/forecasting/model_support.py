"""Model support matrix for heads mode.

Defines which models support which markets (ML/SPREAD/TOTAL) in heads mode.
Used to filter ensemble weights and validate configuration consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Set

__all__ = ["ModelSupport", "get_model_support", "get_supported_markets"]


@dataclass(frozen=True)
class ModelSupport:
    """Support matrix for a model across markets.
    
    Attributes:
        supports_ml: Whether model has a direct ML (p_home_win) head.
        supports_spread: Whether model has a SPREAD (margin_mean/sd) head.
        supports_total: Whether model has a TOTAL (total_mean/sd) head.
        native_fields: Set of canonical field names this model natively produces.
        derived_fields: Set of canonical field names derived from native fields.
    """

    supports_ml: bool
    supports_spread: bool
    supports_total: bool
    native_fields: Set[str]
    derived_fields: Set[str]

    def supports_market(self, market: str) -> bool:
        """Check if model supports a given market.
        
        Args:
            market: "ML", "SPREAD", or "TOTAL" (case-insensitive).
        
        Returns:
            True if model supports the market.
        """
        market_upper = market.upper()
        if market_upper == "ML":
            return self.supports_ml
        elif market_upper == "SPREAD":
            return self.supports_spread
        elif market_upper == "TOTAL":
            return self.supports_total
        return False

    def all_supported_markets(self) -> list[str]:
        """Return list of all supported markets in canonical order."""
        result = []
        if self.supports_ml:
            result.append("ML")
        if self.supports_spread:
            result.append("SPREAD")
        if self.supports_total:
            result.append("TOTAL")
        return result


# Registry mapping model_id -> ModelSupport
# Based on Phase 4-5 heads implementations:
# - Phase 4: Elo, TOOR, GSSD heads implemented
# - Phase 5: Bradley-Terry and Poisson heads implemented
_MODEL_SUPPORT_REGISTRY: dict[str, ModelSupport] = {
    "bradley-terry": ModelSupport(
        supports_ml=True,
        supports_spread=True,
        supports_total=True,
        native_fields={"p_home_win", "margin_mean", "margin_sd", "total_mean", "total_sd"},
        derived_fields={"projected_home_score", "projected_away_score", "projected_total"},
    ),
    "elo": ModelSupport(
        supports_ml=True,
        supports_spread=True,
        supports_total=True,
        native_fields={"model_p_home_win", "margin_mean", "margin_sd", "total_mean", "total_sd"},
        derived_fields={"projected_home_score", "projected_away_score", "projected_total"},
    ),
    "toor": ModelSupport(
        supports_ml=True,
        supports_spread=True,
        supports_total=True,
        native_fields={"model_p_home_win", "margin_mean", "margin_sd", "total_mean", "total_sd"},
        derived_fields={"projected_home_score", "projected_away_score", "projected_total"},
    ),
    "gssd": ModelSupport(
        supports_ml=True,
        supports_spread=True,
        supports_total=True,
        native_fields={"model_p_home_win", "margin_mean", "margin_sd", "total_mean", "total_sd"},
        derived_fields={"projected_home_score", "projected_away_score", "projected_total"},
    ),
    "poisson": ModelSupport(
        supports_ml=True,
        supports_spread=True,
        supports_total=True,
        native_fields={"p_home_win", "margin_mean", "margin_sd", "total_mean", "total_sd"},
        derived_fields={"projected_home_score", "projected_away_score", "projected_total"},
    ),
}


def get_model_support(model_id: str) -> ModelSupport | None:
    """Retrieve the support matrix for a model.
    
    Args:
        model_id: Model identifier (e.g., "elo", "bradley-terry").
    
    Returns:
        ModelSupport instance or None if model is not registered.
    """
    return _MODEL_SUPPORT_REGISTRY.get(model_id)


def get_supported_markets(model_id: str) -> list[str]:
    """Get list of markets supported by a model.
    
    Args:
        model_id: Model identifier.
    
    Returns:
        List of market names ("ML", "SPREAD", "TOTAL") in canonical order,
        or empty list if model not found.
    """
    support = get_model_support(model_id)
    if support is None:
        return []
    return support.all_supported_markets()


def filter_models_for_market(model_ids: list[str], market: str) -> tuple[list[str], list[str]]:
    """Filter a list of models to only those supporting a given market.
    
    Args:
        model_ids: List of model identifiers.
        market: "ML", "SPREAD", or "TOTAL".
    
    Returns:
        Tuple of (supported_models, unsupported_models).
    """
    supported = []
    unsupported = []
    for model_id in model_ids:
        support = get_model_support(model_id)
        if support is None or not support.supports_market(market):
            unsupported.append(model_id)
        else:
            supported.append(model_id)
    return supported, unsupported
