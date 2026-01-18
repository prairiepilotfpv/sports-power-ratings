"""Contract definitions and validation helpers for game data and predictions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Mapping

import pandas as pd

from models.base import (
    GamePrediction,
    REQUIRED_PREDICTION_METADATA_KEYS,
    require_columns,
    validate_probability,
    validate_win_prob_dist,
)


# projected_win_prob_dist is retained as a legacy alias; model_win_prob_samples is the canonical
# list of model probability samples (not an outcome distribution). Outcome distributions are
# represented via margin/total means and standard deviations.
SCHEDULE_EXPORT_COLUMNS: list[str] = [
    "date",
    "game_id",
    "status",
    "projection_status",
    "params_source",
    "tuned_metric_used",
    "tuning_run_id",
    "params_market",
    "home_team",
    "away_team",
    "neutral",
    "overtime",
    "home_score",
    "away_score",
    "result_margin",
    "result_total",
    "home_rating",
    "away_rating",
    "home_advantage",
    "projected_winner",
    "projected_spread",
    "projected_home_spread",
    "projected_win_prob",
    "model_p_home_win",
    "normal_p_home_win",
    "home_win_prob",
    "away_win_prob",
    "home_win_prob_raw",
    "away_win_prob_raw",
    "home_win_prob_calibrated",
    "away_win_prob_calibrated",
    "winner_win_prob",
    "logistic_home_win_prob",
    "win_prob_source",
    "margin_dist_assumption",
    "projected_win_prob_dist",
    "projected_home_score",
    "projected_away_score",
    "projected_total",
    "margin_mean",
    "margin_sd",
    "total_mean",
    "total_sd",
    "margin_dist_params",
    "total_dist_params",
    "model_win_prob_samples",
    "model_win_prob",
    "margin_std",
    "total_std",
]


@dataclass(frozen=True)
class GameRecord:
    """Canonical game record for ingest and storage."""

    date: date
    home_team: str
    away_team: str
    home_score: int | None
    away_score: int | None
    neutral: bool = False
    overtime: bool = False
    decision_type: str | None = None
    game_id: str | None = None
    sport: str | None = None
    season: str | None = None
    division: str | None = None
    conference: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class ModelInput:
    """Canonical model input for predictions."""

    date: Any
    home_team: str
    away_team: str
    neutral: bool = False
    game_id: str | None = None


@dataclass(frozen=True)
class ScheduleExportRow:
    """Canonical schedule export row."""

    date: str
    game_id: str
    status: str
    projection_status: str
    params_source: str
    tuned_metric_used: str | None
    home_team: str
    away_team: str
    neutral: bool
    overtime: bool
    home_score: int | float | None
    away_score: int | float | None
    result_margin: int | float | None
    result_total: int | float | None
    home_rating: float | None
    away_rating: float | None
    home_advantage: float | None
    projected_winner: str | None
    projected_spread: float | None
    projected_home_spread: float | None
    projected_win_prob: float | None
    model_p_home_win: float | None
    normal_p_home_win: float | None
    home_win_prob: float | None
    away_win_prob: float | None
    home_win_prob_raw: float | None
    away_win_prob_raw: float | None
    home_win_prob_calibrated: float | None
    away_win_prob_calibrated: float | None
    winner_win_prob: float | None
    logistic_home_win_prob: float | None
    win_prob_source: str | None
    margin_dist_assumption: str | None
    projected_win_prob_dist: str | None
    projected_home_score: float | None
    projected_away_score: float | None
    projected_total: float | None
    margin_mean: float | None
    margin_sd: float | None
    total_mean: float | None
    total_sd: float | None
    margin_dist_params: str | None
    total_dist_params: str | None
    model_win_prob_samples: str | None
    model_win_prob: float | None


def build_game_id(date_value: Any, home_team: str, away_team: str) -> str:
    """Generate a deterministic game_id fallback."""
    parsed = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(parsed):
        date_str = str(date_value)
    else:
        date_str = parsed.date().isoformat()
    return f"{date_str}_{home_team}_{away_team}"


def validate_game_records(
    df: pd.DataFrame,
    *,
    require_scores: bool = True,
    context: str = "dataset",
) -> pd.DataFrame:
    """Validate a game dataset and return a cleaned copy."""
    if df is None:
        raise ValueError(f"{context} is required.")

    required = ["date", "home_team", "away_team"]
    if require_scores:
        required.extend(["home_score", "away_score"])
    require_columns(df, required)

    validated = df.copy()
    validated = _ensure_boolean_column(
        validated,
        target="neutral",
        alias="neutral_site",
        default=False,
    )
    validated = _ensure_boolean_column(
        validated,
        target="overtime",
        alias=None,
        default=False,
    )

    parsed_dates = pd.to_datetime(validated["date"], errors="coerce", format="mixed")
    if parsed_dates.isna().any():
        bad_values = validated.loc[parsed_dates.isna(), "date"].head(5).tolist()
        raise ValueError(f"Invalid dates in column 'date': {bad_values}")
    validated["date"] = parsed_dates.dt.normalize()

    for col in ["home_score", "away_score"]:
        if col not in validated.columns:
            continue
        numeric = pd.to_numeric(validated[col], errors="coerce")
        invalid_mask = validated[col].notna() & numeric.isna()
        if invalid_mask.any():
            bad_values = validated.loc[invalid_mask, col].head(5).tolist()
            raise ValueError(f"Invalid values in column '{col}': {bad_values}")
        if (numeric.dropna() < 0).any():
            bad_values = numeric[numeric < 0].head(5).tolist()
            raise ValueError(f"Negative values in column '{col}': {bad_values}")
        validated[col] = numeric

    if "status" in validated.columns:
        status = validated["status"].astype(str).str.lower()
        final_mask = status == "final"
        if final_mask.any():
            missing_scores = (
                validated.loc[final_mask, ["home_score", "away_score"]]
                .isna()
                .any(axis=1)
            )
            if missing_scores.any():
                raise ValueError("Final games must include home_score and away_score.")

    validated = ensure_game_id(validated)
    return validated


def validate_model_input(
    df: pd.DataFrame,
    *,
    context: str = "model input",
) -> pd.DataFrame:
    """Validate model prediction inputs and backfill expected fields."""
    if df is None:
        raise ValueError(f"{context} is required.")

    require_columns(df, ["date", "home_team", "away_team"])
    validated = df.copy()
    validated = _ensure_boolean_column(
        validated,
        target="neutral",
        alias="neutral_site",
        default=False,
    )
    validated = ensure_game_id(validated)

    parsed_dates = pd.to_datetime(validated["date"], errors="coerce", format="mixed")
    if parsed_dates.isna().any():
        bad_values = validated.loc[parsed_dates.isna(), "date"].head(5).tolist()
        raise ValueError(f"Invalid dates in column 'date': {bad_values}")

    return validated


def validate_predictions(
    predictions: Iterable[GamePrediction],
    *,
    context: str = "model output",
) -> list[GamePrediction]:
    """Validate model predictions and return them as a list."""
    if predictions is None:
        raise ValueError(f"{context} is required.")

    validated: list[GamePrediction] = []
    for idx, prediction in enumerate(predictions):
        if not isinstance(prediction, GamePrediction):
            raise ValueError(
                f"{context} predictions must be GamePrediction instances; "
                f"got {type(prediction).__name__} at index {idx}."
            )
        if not prediction.game_id:
            raise ValueError(f"{context} prediction missing game_id at index {idx}.")
        if not prediction.date:
            raise ValueError(f"{context} prediction missing date at index {idx}.")
        if not prediction.home_team:
            raise ValueError(f"{context} prediction missing home_team at index {idx}.")
        if not prediction.away_team:
            raise ValueError(f"{context} prediction missing away_team at index {idx}.")
        validate_probability(prediction.p_home_win, field_name="p_home_win")
        validate_win_prob_dist(prediction.win_prob_dist)
        missing_meta = [
            key
            for key in REQUIRED_PREDICTION_METADATA_KEYS
            if key not in prediction.metadata
        ]
        if missing_meta:
            raise ValueError(
                f"{context} prediction metadata missing keys: {', '.join(missing_meta)} "
                f"for game_id={prediction.game_id!r}."
            )
        validated.append(prediction)
    return validated


def validate_schedule_export_frame(
    schedule_df: pd.DataFrame,
    *,
    expected_columns: Iterable[str] = SCHEDULE_EXPORT_COLUMNS,
    context: str = "schedule export",
) -> pd.DataFrame:
    """Ensure a consistent schedule export schema."""
    expected = list(expected_columns)
    expected_set = set(expected)
    missing = [col for col in expected if col not in schedule_df.columns]
    extra = [col for col in schedule_df.columns if col not in expected_set]
    if missing or extra:
        raise ValueError(
            f"{context} column mismatch. Missing: {missing or 'none'}, extra: {extra or 'none'}"
        )
    return schedule_df[expected]


def ensure_game_id(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a stable game_id by building one from date/teams when missing."""
    if "game_id" not in df.columns:
        df = df.copy()
        df["game_id"] = None

    missing_mask = df["game_id"].isna() | (df["game_id"].astype(str).str.strip() == "")
    if missing_mask.any():
        missing_rows = df.loc[missing_mask, ["date", "home_team", "away_team"]]
        parsed_dates = pd.to_datetime(missing_rows["date"], errors="coerce")
        date_strs = parsed_dates.dt.date.astype("string")
        date_strs = date_strs.where(parsed_dates.notna(), missing_rows["date"].astype(str))
        game_ids = (
            date_strs
            + "_"
            + missing_rows["home_team"].astype(str)
            + "_"
            + missing_rows["away_team"].astype(str)
        )
        df.loc[missing_mask, "game_id"] = game_ids
    return df


def _ensure_boolean_column(
    df: pd.DataFrame,
    *,
    target: str,
    alias: str | None,
    default: bool,
) -> pd.DataFrame:
    """Populate a boolean column with an optional alias as a fallback."""
    series = (
        df[target]
        if target in df.columns
        else pd.Series(pd.NA, index=df.index)
    )
    if alias and alias in df.columns:
        alias_series = df[alias]
        series = series.where(~series.isna(), alias_series)
    df[target] = series.apply(lambda value: _coerce_bool(value, default=default))
    return df


def _coerce_bool(value: Any, *, default: bool) -> bool:
    """Convert truthy/falsy values to bool, falling back to default on nulls."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    return bool(value)
