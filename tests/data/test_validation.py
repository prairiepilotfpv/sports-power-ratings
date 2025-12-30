import pandas as pd
import pytest

from data.validation import validate_dataset


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "home_team": ["Home A", "Home B"],
            "away_team": ["Away A", "Away B"],
            "home_score": [100, 95],
            "away_score": [90, 101],
            "game_id": ["game-1", "game-2"],
        }
    )


def test_validate_dataset_missing_columns_raises() -> None:
    df = _base_df().drop(columns=["away_score"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataset(df)


def test_validate_dataset_rejects_invalid_date() -> None:
    df = _base_df()
    df.loc[0, "date"] = "not-a-date"
    with pytest.raises(ValueError, match="Invalid dates in column 'date'"):
        validate_dataset(df)


def test_validate_dataset_rejects_invalid_score_type() -> None:
    df = _base_df()
    df["home_score"] = df["home_score"].astype("object")
    df.loc[0, "home_score"] = "oops"
    with pytest.raises(ValueError, match="Invalid values in column 'home_score'"):
        validate_dataset(df)


def test_validate_dataset_rejects_negative_scores() -> None:
    df = _base_df()
    df.loc[0, "away_score"] = -1
    with pytest.raises(ValueError, match="Negative values in column 'away_score'"):
        validate_dataset(df)


def test_validate_dataset_requires_scores_for_final_status() -> None:
    df = _base_df()
    df["status"] = ["final", "scheduled"]
    df.loc[0, "home_score"] = None
    with pytest.raises(ValueError, match="Final games must include home_score and away_score"):
        validate_dataset(df)


def test_validate_dataset_rejects_duplicate_game_id() -> None:
    df = _base_df()
    df.loc[1, "game_id"] = "game-1"
    with pytest.raises(ValueError, match="Duplicate game_id values found"):
        validate_dataset(df)
