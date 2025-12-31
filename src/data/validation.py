"""Validation helpers for inbound game datasets."""

from __future__ import annotations

import pandas as pd

from models.base import require_columns


REQUIRED_COLUMNS = ["date", "home_team", "away_team", "home_score", "away_score"]


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a game dataset and return a cleaned copy."""
    if df is None:
        raise ValueError("Dataset is required.")

    require_columns(df, REQUIRED_COLUMNS)
    validated = df.copy()

    parsed_dates = pd.to_datetime(validated["date"], errors="coerce", format="mixed")
    if parsed_dates.isna().any():
        bad_values = validated.loc[parsed_dates.isna(), "date"].head(5).tolist()
        raise ValueError(f"Invalid dates in column 'date': {bad_values}")
    validated["date"] = parsed_dates.dt.normalize()

    for col in ["home_score", "away_score"]:
        # Score columns are optional for future games, but must be numeric if provided.
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
        # "final" rows must include scores so downstream pipelines can trust the data.
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

    validated = _ensure_game_id(validated)
    duplicate_ids = validated["game_id"].duplicated(keep=False)
    if duplicate_ids.any():
        dup_values = (
            validated.loc[duplicate_ids, "game_id"]
            .astype(str)
            .dropna()
            .unique()
            .tolist()[:5]
        )
        raise ValueError(f"Duplicate game_id values found: {dup_values}")

    return validated


def _ensure_game_id(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a stable game_id by building one from date/teams when missing."""
    if "game_id" not in df.columns:
        df = df.copy()
        df["game_id"] = None

    missing_mask = df["game_id"].isna() | (df["game_id"].astype(str).str.strip() == "")
    if missing_mask.any():
        df.loc[missing_mask, "game_id"] = df.loc[missing_mask].apply(
            lambda row: f"{row['date'].date()}_{row['home_team']}_{row['away_team']}",
            axis=1,
        )
    return df
