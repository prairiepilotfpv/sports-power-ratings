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
        # Default behavior: when `sport` is None (global invocation), apply guardrail.
        # When a specific sport is provided, only apply by default for NBA to avoid
        # enforcing NBA-specific guardrails on other sports unless explicitly enabled.
        if sport is None:
            return True
        return sport.strip().lower() == "nba"
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
    drop_invalid: bool = True,
    log_reasons_per_row: bool = True,
) -> Tuple[pd.DataFrame, list[tuple[str | None, str | None, list[str]]]]:
    """Filter predictions using the centralized validator.

    Returns (filtered_df, exclusions) where exclusions is a list of
    (model, game_id, reasons). When drop_invalid is False, rows are kept but
    annotated with invalid reasons; only extreme corruption triggers a drop.
    """
    if predictions_df.empty:
        return predictions_df, []

    reasons_col: list[list[str]] = []
    validity: list[bool] = []
    exclusions: list[tuple[str | None, str | None, list[str]]] = []
    hard_drop_reasons = {"prob_out_of_bounds"}
    # Try to infer sport from the predictions frame when one isn't explicitly passed in.
    if sport is None and "sport" in predictions_df.columns:
        non_null = predictions_df["sport"].dropna()
        if not non_null.empty:
            sport = str(non_null.iloc[0]).strip().lower()

    # Resolve a sport-specific config when one isn't explicitly passed in.
    config = validation_config if validation_config is not None else get_validation_config(sport)
    for _, row in predictions_df.iterrows():
        ok, reasons = validate_prediction_row(
            row.to_dict(), config=config, require_score_bounds=(sport is not None)
        )
        hard_invalid = any(reason in hard_drop_reasons for reason in reasons)
        keep_row = ok or (not drop_invalid and not hard_invalid)
        validity.append(keep_row)
        reasons_col.append(reasons)
        if not keep_row:
            model_ref = row.get("model") or row.get("model_id")
            exclusions.append((model_ref, row.get("game_id"), reasons))

    filtered = predictions_df.loc[validity].copy()

    kept_reasons = [r for keep, r in zip(validity, reasons_col, strict=False) if keep]
    if include_reasons:
        filtered["validation_reasons"] = kept_reasons
    if not drop_invalid:
        filtered["__invalid_reasons"] = kept_reasons

    # Emit per-row exclusion warnings when requested. For bulk workflows
    # (e.g., tuning/backtest) callers may pass `log_reasons_per_row=False`
    # to avoid noisy logs; those callers should instead aggregate `exclusions`
    # and emit a single summary per candidate when appropriate.
    if log_reasons_per_row:
        for model, game_id, reasons in exclusions:
            logger.warning(
                "Excluding prediction model=%s game_id=%s sport=%s reasons=%s",
                model,
                game_id,
                sport or "",
                reasons,
            )

    return filtered, exclusions