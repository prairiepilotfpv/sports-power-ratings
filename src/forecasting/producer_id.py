"""Producer ID and provenance normalization for heads mode.

Standardizes producer labels to follow the contract:
- ensemble_<market>_v1 for ensemble outputs
- model_name for single model outputs
- calibration tags may be appended with '+' separator
- no "direct" terminology in heads mode
"""

from __future__ import annotations

import logging
from typing import Literal

_LOGGER = logging.getLogger(__name__)

Market = Literal["ML", "SPREAD", "TOTAL"]

# Canonical ensemble producer IDs
ENSEMBLE_ML_V1 = "ensemble_ml_v1"
ENSEMBLE_SPREAD_V1 = "ensemble_spread_v1"
ENSEMBLE_TOTAL_V1 = "ensemble_total_v1"

CANONICAL_ENSEMBLE_IDS = {
    "ML": ENSEMBLE_ML_V1,
    "SPREAD": ENSEMBLE_SPREAD_V1,
    "TOTAL": ENSEMBLE_TOTAL_V1,
}


def normalize_win_prob_source(
    source: str | None,
    market: Market,
    *,
    heads_mode: bool = False,
) -> str | None:
    """
    Normalize win_prob_source label to canonical form.
    
    Ensures:
    - Ensemble outputs use ensemble_<market>_v1 format
    - Single model outputs use model_name format
    - Calibration tags are appended with '+' if present
    - No "direct" terminology in heads mode
    
    Args:
        source: Current source label (may include calibration tags).
        market: Market type ("ML", "SPREAD", or "TOTAL").
        heads_mode: Whether heads mode is enabled.
    
    Returns:
        Normalized source label, or None if source is None.
    """
    if source is None:
        return None
    
    source = str(source).strip()
    if not source:
        return None
    
    # Extract calibration tags if present (appended with +)
    parts = source.split("+")
    base_source = parts[0].strip()
    calibration_tags = [p.strip() for p in parts[1:] if p.strip()]
    
    # Remove "direct" terminology in heads mode
    if heads_mode and base_source == "direct":
        base_source = "model_direct"  # Fallback marker, prefer model_name
    
    # Normalize ensemble producers
    if base_source.startswith("ensemble_"):
        base_source = CANONICAL_ENSEMBLE_IDS.get(market.upper(), base_source)
    
    # Reconstruct with calibration tags
    result = base_source
    if calibration_tags:
        result = result + "+" + "+".join(sorted(calibration_tags))
    
    return result


def get_ensemble_producer_id(market: Market) -> str:
    """
    Get the canonical ensemble producer ID for a market.
    
    Args:
        market: Market type ("ML", "SPREAD", or "TOTAL").
    
    Returns:
        Canonical ensemble producer ID (e.g., "ensemble_ml_v1").
    """
    return CANONICAL_ENSEMBLE_IDS.get(market.upper(), f"ensemble_{market.lower()}_v1")


def is_ensemble_producer(source: str | None) -> bool:
    """
    Check if a producer ID is an ensemble-based source.
    
    Args:
        source: Producer ID to check (may include calibration tags).
    
    Returns:
        True if source is an ensemble producer.
    """
    if source is None:
        return False
    
    source = str(source).strip()
    if not source:
        return False
    
    # Extract base source (before calibration tags)
    base_source = source.split("+")[0].strip()
    
    return base_source in CANONICAL_ENSEMBLE_IDS.values()


def is_valid_producer_in_market(source: str | None, market: Market) -> bool:
    """
    Validate that a producer ID is valid for a given market.
    
    Args:
        source: Producer ID to validate.
        market: Market type.
    
    Returns:
        True if producer is valid for the market.
    
    Note:
        For ML: can be ensemble_ml_v1 or a single model source.
        For SPREAD/TOTAL: must be ensemble_spread_v1 or ensemble_total_v1.
    """
    if source is None:
        return False
    
    source = str(source).strip()
    base_source = source.split("+")[0].strip()
    
    market_upper = market.upper()
    expected_ensemble = CANONICAL_ENSEMBLE_IDS.get(market_upper)
    
    if market_upper in ("SPREAD", "TOTAL"):
        # SPREAD/TOTAL must be ensemble
        return base_source == expected_ensemble
    elif market_upper == "ML":
        # ML can be ensemble or single model
        return base_source == expected_ensemble or _is_valid_model_name(base_source)
    
    return False


def _is_valid_model_name(name: str) -> bool:
    """
    Check if a name looks like a valid model name (not "direct").
    
    Args:
        name: Name to check.
    
    Returns:
        True if name appears to be a model name.
    """
    if not name or name in ("direct", "model_direct", "none", "unknown"):
        return False
    
    # Model names are typically lowercase with underscores/hyphens
    # e.g., "elo", "bradley-terry", "toor", "gssd", "poisson"
    return name.replace("_", "").replace("-", "").isalnum()


def validate_producer_in_heads_mode(source: str | None, market: Market) -> tuple[bool, str | None]:
    """
    Validate a producer ID is compliant with heads mode contract.
    
    Args:
        source: Producer ID to validate.
        market: Market type.
    
    Returns:
        Tuple of (is_valid, error_message).
        is_valid=True means source complies with heads mode contract.
        error_message is None if valid, otherwise contains reason for rejection.
    """
    if source is None:
        return False, "producer ID cannot be None"
    
    source_str = str(source).strip()
    base_source = source_str.split("+")[0].strip()
    
    # Check for forbidden "direct" terminology
    if base_source == "direct":
        return False, f"Producer ID '{source}' uses forbidden 'direct' terminology in heads mode. Use model name or ensemble_*_v1 format."
    
    # Validate for market
    if not is_valid_producer_in_market(source_str, market):
        expected = CANONICAL_ENSEMBLE_IDS.get(market.upper(), f"ensemble_{market.lower()}_v1")
        if market.upper() in ("SPREAD", "TOTAL"):
            return False, f"Producer ID '{source}' invalid for {market}. Must be '{expected}' for {market} market."
        else:
            return False, f"Producer ID '{source}' invalid for {market}. Must be '{expected}' or a model name."
    
    return True, None
