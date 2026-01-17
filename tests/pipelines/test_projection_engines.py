from __future__ import annotations

import pytest

from models.bradley_terry import BradleyTerry
from pipelines.projection_engines import _bt_projection_engine


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
