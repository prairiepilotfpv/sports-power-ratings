"""Registry and factory for projection heads."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

_LOGGER = logging.getLogger(__name__)


# Global registry: model_id -> HeadSequence factory function
_HEAD_REGISTRY: dict[str, callable] = {}


def register_model_heads(model_id: str, factory: callable) -> None:
    """
    Register a head sequence factory for a model.
    
    Args:
        model_id: Model identifier (e.g., "elo", "toor").
        factory: Callable that returns a HeadSequence instance.
                 Signature: factory() -> HeadSequence
    """
    _HEAD_REGISTRY[model_id] = factory
    if _LOGGER.isEnabledFor(logging.DEBUG):
        _LOGGER.debug(
            f"[heads] registered model_heads for model_id={model_id}",
            extra={"model_id": model_id},
        )


def get_model_heads(model_id: str) -> callable | None:
    """
    Retrieve the head sequence factory for a model.
    
    Args:
        model_id: Model identifier.
    
    Returns:
        Factory function or None if not registered.
    """
    return _HEAD_REGISTRY.get(model_id)


def apply_heads(
    model_id: str,
    df: pd.DataFrame,
    context: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Apply registered heads for a model to a forecast DataFrame.
    
    If no heads are registered for the model, returns None (caller must
    fall back to projection engine or other derivation method).
    
    Args:
        model_id: Model identifier.
        df: Game forecast DataFrame.
        context: Model projection context.
    
    Returns:
        Dict with "applied_heads" and "filled_fields" lists, or None if
        no heads registered for this model.
    
    Raises:
        ValueError: If head dependencies are not satisfied.
    """
    factory = get_model_heads(model_id)
    if factory is None:
        return None

    head_sequence = factory()
    result = head_sequence.apply(df, context)

    if _LOGGER.isEnabledFor(logging.DEBUG):
        _LOGGER.debug(
            f"[heads] model={model_id} applied_heads={result['applied_heads']} "
            f"filled_fields={result['filled_fields']}",
            extra={
                "model_id": model_id,
                "applied_heads": result["applied_heads"],
                "filled_fields": result["filled_fields"],
            },
        )

    return result
