"""Policy helpers for selecting active tuning metrics."""

from __future__ import annotations


def default_active_metric_for_model(model_id: str) -> str:
    """Return the active tuning metric for a model id."""
    if model_id in {"bradley-terry", "elo"}:
        return "log_loss"
    if model_id in {"gssd", "toor", "poisson"}:
        return "mae_margin"
    return "log_loss"
