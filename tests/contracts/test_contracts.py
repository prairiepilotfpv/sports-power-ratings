from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from contracts import (
    SCHEDULE_EXPORT_COLUMNS,
    build_game_id,
    validate_game_records,
    validate_model_input,
    validate_predictions,
    validate_schedule_export_frame,
)
from models.base import GamePrediction


def test_contract_end_to_end_sample_game() -> None:
    games_df = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "home_team": "Home",
                "away_team": "Away",
                "home_score": 100,
                "away_score": 90,
                "neutral": False,
            }
        ]
    )
    validated_games = validate_game_records(games_df, require_scores=True)
    upcoming = validated_games.drop(columns=["home_score", "away_score"])
    model_input = validate_model_input(upcoming)

    prediction = GamePrediction(
        game_id=str(model_input.loc[0, "game_id"]),
        date=str(model_input.loc[0, "date"]),
        home_team=str(model_input.loc[0, "home_team"]),
        away_team=str(model_input.loc[0, "away_team"]),
        p_home_win=0.62,
        pred_margin=4.5,
        metadata={"model_id": "demo", "model_version": "1.0", "params": {}},
    )
    assert validate_predictions([prediction])

    schedule_row = {
        "date": "2024-01-01",
        "game_id": "gid-1",
        "status": "scheduled",
        "home_team": "Home",
        "away_team": "Away",
        "neutral": False,
        "overtime": False,
        "home_score": None,
        "away_score": None,
        "result_margin": None,
        "result_total": None,
        "home_rating": 1.2,
        "away_rating": -0.4,
        "home_advantage": 3.0,
        "projected_winner": "Home",
        "projected_spread": -4.5,
        "projected_home_spread": 4.5,
        "projected_win_prob": 0.62,
        "projected_win_prob_dist": '[{"p_home_win": 0.62, "weight": 1.0}]',
        "projected_home_score": 101.2,
        "projected_away_score": 96.7,
        "projected_total": 197.9,
        "margin_std": 11.3,
        "total_std": 20.2,
    }
    schedule_df = pd.DataFrame([schedule_row], columns=SCHEDULE_EXPORT_COLUMNS)
    assert list(validate_schedule_export_frame(schedule_df).columns) == SCHEDULE_EXPORT_COLUMNS


def test_model_input_accepts_neutral_site_alias() -> None:
    df = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "home_team": "Home",
                "away_team": "Away",
                "neutral_site": True,
            }
        ]
    )
    validated = validate_model_input(df)
    assert bool(validated.loc[0, "neutral"]) is True


def test_missing_game_id_uses_deterministic_fallback() -> None:
    df = pd.DataFrame(
        [
            {
                "date": date(2024, 1, 3),
                "home_team": "Home",
                "away_team": "Away",
            }
        ]
    )
    expected = build_game_id(date(2024, 1, 3), "Home", "Away")
    first = validate_model_input(df)
    second = validate_model_input(df)
    assert first.loc[0, "game_id"] == expected
    assert second.loc[0, "game_id"] == expected


def test_prediction_validation_rejects_invalid_payload() -> None:
    bad_prediction = GamePrediction(
        game_id="",
        date="2024-01-01",
        home_team="Home",
        away_team="Away",
        p_home_win=0.5,
        metadata={"model_id": "demo", "model_version": "1.0", "params": {}},
    )

    with pytest.raises(ValueError, match="missing game_id"):
        validate_predictions([bad_prediction])


def test_schedule_export_validation_rejects_missing_columns() -> None:
    schedule_df = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "game_id": "gid-1",
                "status": "scheduled",
            }
        ]
    )

    with pytest.raises(ValueError, match="column mismatch"):
        validate_schedule_export_frame(schedule_df)
