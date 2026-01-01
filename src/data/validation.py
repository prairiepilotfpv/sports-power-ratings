"""Validation helpers for inbound game datasets."""

from __future__ import annotations

import pandas as pd

from contracts import validate_game_records


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a game dataset and return a cleaned copy."""
    validated = validate_game_records(df, require_scores=True, context="Dataset")
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
