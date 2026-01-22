"""Adapter that exposes Bradley-Terry native forecasts as contracts."""

from __future__ import annotations

import math
from typing import Any, Mapping

from config import DEFAULT_MARGIN_SD_FALLBACK, DEFAULT_TOTAL_SD_FALLBACK
from models.base import resolve_model_identity, validate_probability
from models.bradley_terry import BradleyTerry
from models.forecast_contract import (
    ForecastContract,
    MLForecast,
    SpreadForecast,
    TotalForecast,
)


class BTForecastAdapter:
    """Build forecast contracts from Bradley-Terry matchup projections."""

    @staticmethod
    def from_native_projection(
        *,
        model: BradleyTerry | Any,
        projection: Mapping[str, Any] | None,
        game_id: str | None,
        date: str | None,
        home_team: str,
        away_team: str,
    ) -> ForecastContract | None:
        if projection is None:
            return None
        identity = resolve_model_identity(model)
        model_id = identity.get("model_id")
        if model_id is None:
            return None
        model_version = identity.get("model_version")
        metadata: Mapping[str, Any] = {
            "model_id": model_id,
            "model_version": model_version,
            "params": identity.get("params"),
        }

        warnings: list[str] = []

        ml_forecast, source_ml = _build_ml_forecast(projection, warnings)
        spread_forecast, source_spread = _build_spread_forecast(projection, warnings)
        total_forecast, source_total = _build_total_forecast(projection, warnings)

        if ml_forecast is None and spread_forecast is None and total_forecast is None:
            warnings.append("empty_native_forecast")
            return None

        projected_home_score = _as_float(projection.get("projected_home_score"))
        projected_away_score = _as_float(projection.get("projected_away_score"))
        projected_total = _as_float(
            projection.get("projected_total") or projection.get("total_mean")
        )

        return ForecastContract(
            game_id=game_id,
            date=str(date) if date is not None else "",
            home_team=home_team,
            away_team=away_team,
            model_id=model_id,
            model_version=model_version,
            metadata=metadata,
            ml=ml_forecast,
            spread=spread_forecast,
            total=total_forecast,
            source_ml=source_ml,
            source_spread=source_spread,
            source_total=source_total,
            projected_home_score=projected_home_score,
            projected_away_score=projected_away_score,
            projected_total=projected_total,
            warnings=warnings,
        )


def _build_ml_forecast(
    projection: Mapping[str, Any], warnings: list[str]
) -> tuple[MLForecast | None, str]:
    raw_prob = _as_float(projection.get("model_p_home_win"))
    source = "native"
    if raw_prob is None:
        raw_prob = _as_float(projection.get("p_home_win"))
        source = "derived" if raw_prob is not None else "missing"
    if raw_prob is None:
        warnings.append("ml_missing")
        return None, source
    p_home = validate_probability(raw_prob, field_name="p_home_win")
    p_away = 1.0 - p_home
    p_away = max(0.0, min(1.0, p_away))
    return MLForecast(p_home_win=p_home, p_away_win=p_away, source=source), source


def _build_spread_forecast(
    projection: Mapping[str, Any], warnings: list[str]
) -> tuple[SpreadForecast | None, str]:
    margin_mean = projection.get("margin_mean")
    if margin_mean is None:
        warnings.append("spread_missing")
        return None, "missing"
    margin_sd = _coerce_sd(projection.get("margin_sd"), DEFAULT_MARGIN_SD_FALLBACK)
    if margin_sd is None:
        warnings.append("spread_sd_missing")
        return None, "missing"
    return (
        SpreadForecast(
            margin_mean=float(margin_mean),
            margin_sd=margin_sd,
            source="native",
        ),
        "native",
    )


def _build_total_forecast(
    projection: Mapping[str, Any], warnings: list[str]
) -> tuple[TotalForecast | None, str]:
    total_mean = projection.get("total_mean")
    if total_mean is None:
        warnings.append("total_missing")
        return None, "missing"
    total_sd = _coerce_sd(projection.get("total_sd"), DEFAULT_TOTAL_SD_FALLBACK)
    if total_sd is None:
        warnings.append("total_sd_missing")
        return None, "missing"
    return (
        TotalForecast(
            total_mean=float(total_mean),
            total_sd=total_sd,
            source="native",
        ),
        "native",
    )


def _coerce_sd(value: Any, fallback: float) -> float | None:
    try:
        sd = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(sd) or sd <= 0.0:
        return fallback
    return sd


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
