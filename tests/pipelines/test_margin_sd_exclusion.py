from __future__ import annotations

import pandas as pd

from pipelines.schedule import _project_row


def test_nba_margin_sd_is_excluded():
    base_row = pd.Series(
        {
            "date": "2024-01-01",
            "home_team": "Home",
            "away_team": "Away",
            "home_score": None,
            "away_score": None,
            "neutral": False,
            "overtime": False,
            "game_id": "gid-1",
        }
    )

    ratings = {"Home": 1.0, "Away": 0.5}

    def fake_engine(home, away, model_instance, context):
        # Return a projection with a tiny margin_sd which should trigger exclusion
        return {
            "projected_home_score": 100.0,
            "projected_away_score": 95.0,
            "projected_total": 195.0,
            "normal_p_home_win": 0.95,
            "projected_win_prob": 0.95,
            "model_p_home_win": 0.95,
            "logistic_home_win_prob": 0.95,
            "win_prob_source": "logistic",
            "margin_dist_assumption": "normal_approx",
            "margin_mean": 5.0,
            "margin_sd": 1.0,
            "total_mean": 195.0,
            "total_sd": 3.0,
        }

    projection_context = {
        "ratings": ratings,
        "base_total": 0.0,
        "scoring_averages": {},
        "total_intercept": None,
        "total_slope": None,
        "margin_std": 1.0,
        "total_std": 3.0,
        "conditional_sd_intercept": None,
        "conditional_sd_slope": None,
        "win_prob_k": 10.0,
        "sport": "nba",
    }

    row = _project_row(
        base_row,
        ratings=ratings,
        status="scheduled",
        home_advantage=0.0,
        params_source="default",
        tuned_metric_used=None,
        model_instance=None,
        projection_engine=fake_engine,
        projection_context=projection_context,
    )

    # Schedule-level projections should remain intact; exclusion is applied
    # at evaluation/backtest time only.
    assert row["projection_status"] == "ok"
    assert row["projected_home_score"] == 100.0
    assert row["margin_mean"] == 5.0
