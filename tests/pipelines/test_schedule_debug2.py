from __future__ import annotations

from pipelines.schedule import _project_row
from pipelines.projection_engines import get_projection_engine
from models.bradley_terry import BradleyTerry
from datetime import date
import pandas as pd


def test_debug_bt_projection_prints():
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
    model_instance = BradleyTerry(max_iter=50)
    model_instance.fit(
        [
            {"home_team": "Home", "away_team": "Away", "home_score": 100, "away_score": 90},
            {"home_team": "Away", "away_team": "Home", "home_score": 95, "away_score": 105},
        ]
    )
    model_instance.fit(
        [
            {"home_team": "Home", "away_team": "Away", "home_score": 3, "away_score": 1, "neutral": False},
            {"home_team": "Home", "away_team": "Away", "home_score": 4, "away_score": 2, "neutral": False},
        ]
    )
    ratings = dict(model_instance.rankings())
    projection_engine = get_projection_engine(model_instance)
    projection_context = {
        "ratings": ratings,
        "base_total": 0.0,
        "scoring_averages": {},
        "total_intercept": None,
        "total_slope": None,
        "margin_std": 8.0,
        "total_std": 15.0,
        "conditional_sd_intercept": None,
        "conditional_sd_slope": None,
        "win_prob_k": 10.0,
    }
    positive = _project_row(
        base_row,
        ratings=ratings,
        status="scheduled",
        home_advantage=0.0,
        params_source="default",
        tuned_metric_used=None,
        model_instance=model_instance,
        projection_engine=projection_engine,
        projection_context=projection_context,
    )
    print('DEBUG_PROJECTION:', positive)
    assert positive['home_win_prob'] is not None
    assert positive['home_win_prob'] > 0.5
