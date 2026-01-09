"""Guardrails for downstream evaluation of prediction outputs."""

from __future__ import annotations

import logging
from typing import Iterable, Tuple

import pandas as pd

from config import (
    MARGIN_SD_GUARDRAIL_MAX,
    MARGIN_SD_GUARDRAIL_MIN,
)
from eval.validation import get_validation_config, validate_prediction_row, ValidationConfig

logger = logging.getLogger(__name__)


def _normalize_enabled_sports(enabled_sports: Iterable[str] | None) -> set[str] | None:
    if enabled_sports is None:
        return None
    normalized: set[str] = set()
    for sport in enabled_sports:
        if not sport or not sport.strip():
            continue
        normalized.add(sport.strip().lower())
    return normalized if normalized else None


def _should_apply_guardrail(sport: str | None, enabled: set[str] | None) -> bool:
    if enabled is None:
        return True
    if sport is None:
        return False
    return sport.strip().lower() in enabled


def enforce_margin_sd_guardrail(
    predictions_df: pd.DataFrame,
    *,
    sport: str | None = None,
    enabled_sports: Iterable[str] | None = None,
    min_margin_sd: float = MARGIN_SD_GUARDRAIL_MIN,
    max_margin_sd: float = MARGIN_SD_GUARDRAIL_MAX,
) -> pd.DataFrame:
    """Drop predictions with margin SD outside the configured tolerance."""
    if predictions_df.empty:
        return predictions_df
    enabled = _normalize_enabled_sports(enabled_sports)
    if not _should_apply_guardrail(sport, enabled):
        return predictions_df

    sd_series = pd.to_numeric(
        predictions_df.get("margin_sd", pd.Series(dtype=float)),
        errors="coerce",
    )
    exclude_mask = sd_series.notna() & (
        (sd_series < min_margin_sd) | (sd_series > max_margin_sd)
    )
    if not exclude_mask.any():
        return predictions_df

    dropped = int(exclude_mask.sum())
    filtered = predictions_df.loc[~exclude_mask].copy()
    logger.warning(
        "Excluded %d predictions with margin_sd outside [%.1f, %.1f]%s",
        dropped,
        min_margin_sd,
        max_margin_sd,
        f" for sport={sport}" if sport else "",
    )
    return filtered


def apply_prediction_validation(
    predictions_df: pd.DataFrame,
    *,
    sport: str | None = None,
    validation_config: ValidationConfig | None = None,
    include_reasons: bool = False,
) -> Tuple[pd.DataFrame, list[tuple[str | None, str | None, list[str]]]]:
    """Filter predictions using the centralized validator.

    Returns (filtered_df, exclusions) where exclusions is a list of
    (model, game_id, reasons).
    """
    if predictions_df.empty:
        return predictions_df, []

    reasons_col: list[list[str]] = []
    validity: list[bool] = []
    exclusions: list[tuple[str | None, str | None, list[str]]] = []
    # Resolve a sport-specific config when one isn't explicitly passed in.
    config = validation_config if validation_config is not None else get_validation_config(sport)
    for _, row in predictions_df.iterrows():
        ok, reasons = validate_prediction_row(row.to_dict(), config=config)
        validity.append(ok)
        reasons_col.append(reasons)
        if not ok:
            exclusions.append((row.get("model"), row.get("game_id"), reasons))

    filtered = predictions_df.loc[validity].copy()
    if include_reasons:
        filtered["validation_reasons"] = [r for ok, r in zip(validity, reasons_col, strict=False) if ok]

    for model, game_id, reasons in exclusions:
        logger.warning(
            "Excluding prediction model=%s game_id=%s sport=%s reasons=%s",
            model,
            game_id,
            sport or "",
            reasons,
        )

    return filtered, exclusions