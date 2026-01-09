"""Prediction row validation for EV guardrails."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Any, Tuple, List
import math

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
        if key in row:
            return row.get(key)
    return None


def validate_prediction_row(row: Mapping[str, Any], config: ValidationConfig = DEFAULT_VALIDATION_CONFIG) -> Tuple[bool, List[str]]:
    """Validate a single prediction row and return (is_valid, reasons)."""
    reasons: list[str] = []

    margin_sd_val = _coerce_float(row.get("margin_sd") or row.get("margin_std"))
    if margin_sd_val is None:
        reasons.append("missing_margin_sd")
    elif not (config.margin_sd_min <= margin_sd_val <= config.margin_sd_max):
        reasons.append("invalid_margin_sd")

    total_sd_val = _coerce_float(row.get("total_sd") or row.get("total_std"))
    if total_sd_val is None:
        reasons.append("missing_total_sd")
    elif not (config.total_sd_min <= total_sd_val <= config.total_sd_max):
        reasons.append("invalid_total_sd")

    home_score_raw = _first_present(row, ("projected_home_score", "home_score_pred", "home_score"))
    away_score_raw = _first_present(row, ("projected_away_score", "away_score_pred", "away_score"))
    home_score_val = _coerce_float(home_score_raw)
    away_score_val = _coerce_float(away_score_raw)
    if home_score_val is None or away_score_val is None:
        reasons.append("missing_score")
    else:
        if not (config.score_min <= home_score_val <= config.score_max):
            reasons.append("score_out_of_bounds")
        if not (config.score_min <= away_score_val <= config.score_max):
            reasons.append("score_out_of_bounds")

    total_raw = _first_present(row, ("projected_total", "total", "total_mean"))
    total_val = _coerce_float(total_raw)
    if total_val is not None and home_score_val is not None and away_score_val is not None:
        derived_total = home_score_val + away_score_val
        if abs(derived_total - total_val) > config.total_tolerance:
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
