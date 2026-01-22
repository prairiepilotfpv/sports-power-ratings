"""Contract definitions and validation helpers for game data and predictions."""

from __future__ import annotations

import warnings
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
    "params_source_label",
    "params_source_run_id",
    "tuned_metric_used",
    "params_metric_optimized",
    "params_best_score",
    "params_fingerprint",
    "params_nonempty",
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
    params_source_label: str | None = None
    params_source_run_id: str | None = None
    tuned_metric_used: str | None = None
    params_metric_optimized: str | None = None
    params_best_score: float | None = None
    params_fingerprint: str | None = None
    params_nonempty: bool | None = None
    home_team: str | None = None
    away_team: str | None = None
    neutral: bool = False
    overtime: bool = False
    home_score: int | float | None = None
    away_score: int | float | None = None
    result_margin: int | float | None = None
    result_total: int | float | None = None
    home_rating: float | None = None
    away_rating: float | None = None
    home_advantage: float | None = None
    projected_winner: str | None = None
    projected_spread: float | None = None
    projected_home_spread: float | None = None
    projected_win_prob: float | None = None
    model_p_home_win: float | None = None
    normal_p_home_win: float | None = None
    home_win_prob: float | None = None
    away_win_prob: float | None = None
    home_win_prob_raw: float | None = None
    away_win_prob_raw: float | None = None
    home_win_prob_calibrated: float | None = None
    away_win_prob_calibrated: float | None = None
    winner_win_prob: float | None = None
    logistic_home_win_prob: float | None = None
    win_prob_source: str | None = None
    margin_dist_assumption: str | None = None
    projected_win_prob_dist: str | None = None
    projected_home_score: float | None = None
    projected_away_score: float | None = None
    projected_total: float | None = None
    margin_mean: float | None = None
    margin_sd: float | None = None
    total_mean: float | None = None
    total_sd: float | None = None
    margin_dist_params: str | None = None
    total_dist_params: str | None = None
    model_win_prob_samples: str | None = None
    model_win_prob: float | None = None


def build_game_id(date_value: Any, home_team: str, away_team: str) -> str:
    """Generate a deterministic game_id fallback.

    .. deprecated::
        Use :func:`src.utils.game_id.make_game_id` instead, which produces
        a canonical hash-based ID format: ``{sport}:{season}:{date}:{hash12}``.
        This function will be removed in a future release.
    """
    warnings.warn(
        "build_game_id is deprecated; use src.utils.game_id.make_game_id instead. "
        "The canonical format is '{sport}:{season}:{date}:{hash12}'.",
        DeprecationWarning,
        stacklevel=2,
    )
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
    sport: str | None = None,
    season: str | None = None,
) -> pd.DataFrame:
    """Validate a game dataset and return a cleaned copy.

    Parameters
    ----------
    sport, season : str, optional
        When provided, ``ensure_game_id`` will use the canonical hash-based
        game_id format instead of the legacy ``{date}_{home}_{away}`` format.
    """""
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

    validated = ensure_game_id(validated, sport=sport, season=season)
    return validated


def validate_model_input(
    df: pd.DataFrame,
    *,
    context: str = "model input",
    sport: str | None = None,
    season: str | None = None,
) -> pd.DataFrame:
    """Validate model prediction inputs and backfill expected fields.

    Parameters
    ----------
    sport, season : str, optional
        When provided, ``ensure_game_id`` will use the canonical hash-based
        game_id format instead of the legacy ``{date}_{home}_{away}`` format.
    """""
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
    validated = ensure_game_id(validated, sport=sport, season=season)

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


def ensure_game_id(
    df: pd.DataFrame,
    *,
    sport: str | None = None,
    season: str | None = None,
) -> pd.DataFrame:
    """Guarantee a stable game_id by building one from date/teams when missing.

    When ``sport`` and ``season`` are provided (or exist as columns in ``df``),
    uses the canonical :func:`src.utils.game_id.make_game_id` format:
    ``{sport}:{season}:{date}:{hash12}``.

    Falls back to the legacy ``{date}_{home}_{away}`` format when sport/season
    are unavailable. The legacy format is deprecated and will eventually be
    removed.
    """
    if "game_id" not in df.columns:
        df = df.copy()
        df["game_id"] = None

    missing_mask = df["game_id"].isna() | (df["game_id"].astype(str).str.strip() == "")
    if not missing_mask.any():
        return df

    # Import canonical game_id generator
    from src.utils.game_id import make_game_id

    df = df.copy()
    for idx in df.index[missing_mask]:
        row = df.loc[idx]
        # Resolve sport/season: prefer params, then columns, then None
        row_sport = sport or (row.get("sport") if "sport" in df.columns else None)
        row_season = season or (row.get("season") if "season" in df.columns else None)
        date_val = row["date"]
        home = str(row["home_team"])
        away = str(row["away_team"])

        if row_sport and row_season:
            # Use canonical format
            try:
                df.at[idx, "game_id"] = make_game_id(
                    row_sport, row_season, date_val, away, home
                )
                continue
            except Exception:
                pass  # Fall through to legacy format

        # Legacy fallback (deprecated)
        parsed = pd.to_datetime(date_val, errors="coerce")
        if pd.isna(parsed):
            date_str = str(date_val)
        else:
            date_str = parsed.date().isoformat()
        df.at[idx, "game_id"] = f"{date_str}_{home}_{away}"

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
