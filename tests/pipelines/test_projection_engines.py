from __future__ import annotations

import pytest

from models.elo import EloModel
from models.bradley_terry import BradleyTerry
from pipelines.projection_engines import _bt_projection_engine, get_projection_engine


def test_bt_projection_engine_emits_direct_probability() -> None:
    model = BradleyTerry()
    model.ratings.update({"Alpha": 1.0, "Beta": 0.0})

    projection = _bt_projection_engine(
        "Alpha",
        "Beta",
        model,
        {"neutral": False},
    )

    model_p = projection["model_p_home_win"]
    assert model_p is not None
    assert projection["projected_win_prob"] == pytest.approx(model_p)
    assert projection["win_prob_source"] == "direct"
    assert projection["normal_p_home_win"] is None


def test_elo_projection_engine_prefers_logistic_probability() -> None:
    model = EloModel()
    projection_engine = get_projection_engine(model)
    ratings = {"Home": 1600.0, "Away": 1500.0}
    context = {
        "ratings": ratings,
        "rating_units": "points",
        "margin_std": 5.0,
        "total_std": 10.0,
        "scoring_averages": {},
        "win_prob_k": 15.0,
    }

    projection = projection_engine("Home", "Away", model, context)
    logistic_p = projection["logistic_home_win_prob"]
    assert projection["margin_mean"] is not None
    assert logistic_p is not None
    assert projection["win_prob_source"] == "logistic"
    assert projection["projected_win_prob"] == pytest.approx(logistic_p)
    assert projection["model_p_home_win"] == pytest.approx(logistic_p)
    assert projection["normal_p_home_win"] is not None
