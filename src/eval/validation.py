"""Prediction row validation for EV guardrails."""

from __future__ import annotations

import ast
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Any, Tuple, List

from config import (
    MARGIN_SD_GUARDRAIL_MAX,
    MARGIN_SD_GUARDRAIL_MIN,
    PROJECTED_SCORE_MAX,
    PROJECTED_SCORE_MIN,
    PROJECTED_TOTAL_TOLERANCE,
    TOTAL_SD_GUARDRAIL_MAX,
    TOTAL_SD_GUARDRAIL_MIN,
)


@dataclass(frozen=True)
class ValidationConfig:
    margin_sd_min: float = MARGIN_SD_GUARDRAIL_MIN
    margin_sd_max: float = MARGIN_SD_GUARDRAIL_MAX
    total_sd_min: float = TOTAL_SD_GUARDRAIL_MIN
    total_sd_max: float = TOTAL_SD_GUARDRAIL_MAX
    score_min: float = PROJECTED_SCORE_MIN
    score_max: float = PROJECTED_SCORE_MAX
    total_tolerance: float = PROJECTED_TOTAL_TOLERANCE
    allowed_margin_dists: frozenset[str] = field(
        default_factory=lambda: frozenset({"normal_approx", "empirical", "none"})
    )


# Default (legacy) validation config — kept for backwards compatibility.
DEFAULT_VALIDATION_CONFIG = ValidationConfig()

# Backwards-compatible aliases used across the codebase and tests
NBA_VALIDATION_CONFIG = DEFAULT_VALIDATION_CONFIG


# Sport-specific overrides. Add new sports here to keep the system sports-agnostic.
SPORT_VALIDATION_CONFIGS: dict[str, ValidationConfig] = {
    # NBA uses the default guardrails defined in src/config.py
    "nba": DEFAULT_VALIDATION_CONFIG,
    # NHL: projected scores are much lower; loosen score bounds and SD minima
    "nhl": ValidationConfig(
        margin_sd_min=1.0,
        margin_sd_max=20.0,
        total_sd_min=1.0,
        total_sd_max=30.0,
        score_min=0.0,
        score_max=15.0,
        total_tolerance=2.0,
    ),
}


_LOGGER = logging.getLogger(__name__)
_DEBUG_PRED_VALIDATE = os.getenv("DEBUG_PRED_VALIDATE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def get_validation_config(sport: str | None) -> ValidationConfig:
    """Return a ValidationConfig for the given sport (case-insensitive).

    If sport is None or an unknown sport, returns DEFAULT_VALIDATION_CONFIG.
    """
    if not sport:
        return DEFAULT_VALIDATION_CONFIG
    key = str(sport).strip().lower()
    return SPORT_VALIDATION_CONFIGS.get(key, DEFAULT_VALIDATION_CONFIG)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def _first_present(row: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key not in row:
            continue
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            continue
        return value
    return None


def validate_prediction_row(
    row: Mapping[str, Any],
    config: ValidationConfig = DEFAULT_VALIDATION_CONFIG,
    *,
    require_score_bounds: bool = True,
) -> Tuple[bool, List[str]]:
    """Validate a single prediction row and return (is_valid, reasons).

    Checks are limited to prediction sanity and internal consistency; actual
    outcomes must never influence validity. If `require_score_bounds` is False,
    the projected home/away score bounds checks are skipped. This is used by
    callers that do not supply an explicit sport context (e.g., generic
    backtests) to avoid applying NBA-specific score guardrails.
    """
    reasons: list[str] = []

    margin_sd_val = _coerce_float(row.get("margin_sd") or row.get("margin_std"))
    if margin_sd_val is None:
        reasons.append("missing_margin_sd")
    elif not (config.margin_sd_min <= margin_sd_val <= config.margin_sd_max):
        reasons.append("invalid_margin_sd")

    # Determine if a total/total_mean is present; only then require a total SD.
    total_raw = _first_present(row, ("projected_total", "pred_total", "total", "total_mean"))
    total_val = _coerce_float(total_raw)

    total_sd_val = _coerce_float(row.get("total_sd") or row.get("total_std"))
    if total_val is not None:
        if total_sd_val is None:
            reasons.append("missing_total_sd")
        elif not (config.total_sd_min <= total_sd_val <= config.total_sd_max):
            reasons.append("invalid_total_sd")

    home_score_raw = _first_present(row, ("projected_home_score", "home_score_pred"))
    away_score_raw = _first_present(row, ("projected_away_score", "away_score_pred"))
    extra_raw = row.get("extra")
    extra: dict[str, Any] = {}
    if isinstance(extra_raw, dict):
        extra = extra_raw
    elif extra_raw and isinstance(extra_raw, str) and extra_raw.strip().startswith("{"):
        try:
            parsed_extra = ast.literal_eval(extra_raw)
            if isinstance(parsed_extra, dict):
                extra = parsed_extra
        except Exception:
            # best-effort: if parsing fails, fall back to top-level fields
            pass
    if extra.get("projected_home_score") is not None:
        home_score_raw = extra.get("projected_home_score")
    if extra.get("projected_away_score") is not None:
        away_score_raw = extra.get("projected_away_score")
    home_score_val = _coerce_float(home_score_raw)
    away_score_val = _coerce_float(away_score_raw)
    scores_present = home_score_val is not None and away_score_val is not None
    partial_scores = (home_score_val is None) != (away_score_val is None)

    if require_score_bounds and partial_scores:
        reasons.append("missing_score")

    if require_score_bounds and scores_present:
        if not (config.score_min <= home_score_val <= config.score_max):
            reasons.append("score_out_of_bounds")
        if not (config.score_min <= away_score_val <= config.score_max):
            reasons.append("score_out_of_bounds")

    # total_val computed above
    if total_val is not None and scores_present:
        derived_total = home_score_val + away_score_val
        if abs(derived_total - total_val) > config.total_tolerance:
            if _DEBUG_PRED_VALIDATE:
                _log_total_inconsistency(
                    row,
                    derived_total=derived_total,
                    total_val=float(total_val),
                    home_score=home_score_val,
                    away_score=away_score_val,
                    extra=extra,
                )
            reasons.append("total_inconsistent")

    for prob_key in (
        "model_p_home_win",
        "normal_p_home_win",
        "home_win_prob",
        "away_win_prob",
        "winner_win_prob",
        "logistic_home_win_prob",
    ):
        val = _coerce_float(row.get(prob_key))
        if val is None:
            continue
        if val < 0.0 or val > 1.0:
            reasons.append("prob_out_of_bounds")
            break

    margin_dist_assumption = row.get("margin_dist_assumption")
    if margin_dist_assumption:
        if str(margin_dist_assumption) not in config.allowed_margin_dists:
            reasons.append("invalid_margin_dist")

    return len(reasons) == 0, reasons


def _log_total_inconsistency(
    row: Mapping[str, Any],
    *,
    derived_total: float,
    total_val: float,
    home_score: float,
    away_score: float,
    extra: Mapping[str, Any],
) -> None:
    delta = derived_total - total_val
    payload = {
        "game_id": row.get("game_id"),
        "pred_total": row.get("pred_total"),
        "total_mean": row.get("total_mean"),
        "projected_total": row.get("projected_total"),
        "projected_home_score": row.get("projected_home_score"),
        "projected_away_score": row.get("projected_away_score"),
        "extra_total_mean": extra.get("total_mean"),
        "extra_projected_total": extra.get("projected_total"),
        "extra_projected_home_score": extra.get("projected_home_score"),
        "extra_projected_away_score": extra.get("projected_away_score"),
        "home_score": home_score,
        "away_score": away_score,
        "derived_total": derived_total,
        "total_val": total_val,
        "delta": delta,
    }
    _LOGGER.warning("total_inconsistent diagnostic %s", payload)
