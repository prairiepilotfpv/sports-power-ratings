"""
Standardized JSON schema for ensemble component representation.

This module defines the uniform schema used across ML, SPREAD, and TOTAL ensembles
for representing individual model contributions to ensemble predictions.
"""

from typing import Any, TypedDict, Optional


class EnsembleComponent(TypedDict, total=False):
    """
    Standardized schema for an ensemble component.

    This schema is used consistently across all three market types (ML, SPREAD, TOTAL).

    Fields:
        model: Model name (string)
        weight: Normalized weight in ensemble (0.0 to 1.0, sum to 1.0 across components)
        value: The model's prediction value:
            - For ML: probability (0.0 to 1.0)
            - For SPREAD: projected margin (home - away)
            - For TOTAL: projected total points
        uncertainty: Optional uncertainty measure:
            - For ML: None (not applicable)
            - For SPREAD: margin standard deviation
            - For TOTAL: total standard deviation
    """
    model: str
    weight: float
    value: Optional[float]
    uncertainty: Optional[float]


def normalize_legacy_component(comp: dict[str, Any], market: str) -> EnsembleComponent:
    """
    Normalize a legacy component to the standardized schema.

    Legacy schemas:
    - ML: {"model": str, "prob": float, "weight": float}
    - SPREAD: {"model": str, "margin_mean": float, "margin_sd": float, "w": float}
    - TOTAL: {"model": str, "total_mean": float, "total_sd": float, "w": float}

    Args:
        comp: Component dictionary in legacy format
        market: Market type ("ML", "SPREAD", "TOTAL")

    Returns:
        Component in standardized schema
    """
    result: EnsembleComponent = {
        "model": comp["model"],
        "weight": 0.0,
        "value": None,
        "uncertainty": None,
    }

    if market == "ML":
        # Legacy ML: {"prob": float, "weight": float}
        result["weight"] = float(comp.get("weight", 0.0))
        result["value"] = comp.get("prob")  # May be None
        result["uncertainty"] = None
    elif market == "SPREAD":
        # Legacy SPREAD: {"margin_mean": float, "margin_sd": float, "w": float}
        result["weight"] = float(comp.get("w", comp.get("weight", 0.0)))
        result["value"] = comp.get("margin_mean", comp.get("value"))
        result["uncertainty"] = comp.get("margin_sd", comp.get("uncertainty"))
    elif market == "TOTAL":
        # Legacy TOTAL: {"total_mean": float, "total_sd": float, "w": float}
        result["weight"] = float(comp.get("w", comp.get("weight", 0.0)))
        result["value"] = comp.get("total_mean", comp.get("value"))
        result["uncertainty"] = comp.get("total_sd", comp.get("uncertainty"))

    return result


def create_component(
    model: str,
    weight: float,
    value: Optional[float],
    uncertainty: Optional[float] = None
) -> EnsembleComponent:
    """
    Create a standardized ensemble component.

    Args:
        model: Model name
        weight: Normalized weight (should sum to 1.0 across all components)
        value: Prediction value (prob/margin/total)
        uncertainty: Optional uncertainty (SD for spread/total, None for ML)

    Returns:
        Component in standardized schema
    """
    return {
        "model": model,
        "weight": float(weight),
        "value": value,
        "uncertainty": uncertainty,
    }


def sort_components(components: list[EnsembleComponent]) -> list[EnsembleComponent]:
    """
    Sort components by model name for deterministic ordering.

    Args:
        components: List of ensemble components

    Returns:
        Sorted list of components
    """
    return sorted(components, key=lambda c: c["model"])
