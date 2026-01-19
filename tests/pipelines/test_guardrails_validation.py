from __future__ import annotations

import pandas as pd

from pipelines.guardrails import apply_prediction_validation


def test_apply_prediction_validation_keeps_rows_when_drop_invalid_false() -> None:
    df = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "p_home_win": 0.6,
                "pred_total": 210.0,
                "margin_sd": pd.NA,
                "total_sd": 9.0,
            }
        ]
    )

    filtered, exclusions = apply_prediction_validation(
        df, sport="nba", drop_invalid=False, include_reasons=True
    )

    assert len(filtered) == 1
    assert "__invalid_reasons" in filtered.columns
    assert "missing_margin_sd" in filtered["__invalid_reasons"].iloc[0]
    assert exclusions == []


def test_prediction_validation_ignores_actual_scores_for_consistency_checks() -> None:
    df = pd.DataFrame(
        [
            {
                "game_id": "g2",
                "home_score": 100,
                "away_score": 90,
                "pred_total": 210.0,
                "margin_sd": 5.0,
                "total_sd": 9.0,
                "p_home_win": 0.55,
            }
        ]
    )

    filtered, _ = apply_prediction_validation(
        df, sport="nba", drop_invalid=False, include_reasons=True
    )

    assert len(filtered) == 1
    assert filtered["__invalid_reasons"].iloc[0] == []


def test_prediction_validation_flags_projected_inconsistency_but_keeps_rows() -> None:
    df = pd.DataFrame(
        [
            {
                "game_id": "g3",
                "projected_home_score": 110.0,
                "projected_away_score": 90.0,
                "pred_total": 150.0,
                "margin_sd": 4.0,
                "total_sd": 8.0,
                "p_home_win": 0.6,
            }
        ]
    )

    filtered, exclusions = apply_prediction_validation(
        df, sport="nba", drop_invalid=False, include_reasons=True
    )

    assert len(filtered) == 1
    reasons = filtered["__invalid_reasons"].iloc[0]
    assert "total_inconsistent" in reasons
    assert exclusions == []
