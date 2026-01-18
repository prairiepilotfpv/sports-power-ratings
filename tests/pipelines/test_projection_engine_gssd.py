from __future__ import annotations

import math
import pytest

from models.base import ModelMetadata
from pipelines.projection_engines import get_projection_engine


class _DummyGSSDModel:
    def __init__(self) -> None:
        self._meta = ModelMetadata(
            model_id="gssd",
            model_version="1.0",
            params={"winprob_bias": 2.0},
            supports_margin=True,
            supports_total=True,
            supports_win_prob=True,
        )

    def metadata(self) -> ModelMetadata:
        return self._meta


def test_gssd_projection_engine_prefers_logistic_probability() -> None:
    model = _DummyGSSDModel()
    engine = get_projection_engine(model)

    context = {
        "ratings": {"Alpha": 5.0, "Beta": 0.0},
        "home_advantage": 0.0,
        "neutral": False,
        "rating_units": "points",
        "win_prob_k": 10.0,
        "winprob_bias": 2.0,
    }

    projection = engine("Alpha", "Beta", model, context)

    assert projection["win_prob_source"] == "logistic"
    normal_prob = projection.get("normal_p_home_win")
    logistic_prob = projection.get("model_p_home_win")
    assert normal_prob is not None
    assert logistic_prob is not None
    # Logistic prob should incorporate bias and differ from the margin-normal diag stream
    assert not math.isclose(normal_prob, logistic_prob)
    assert projection.get("projected_win_prob") == pytest.approx(logistic_prob)
    assert projection.get("logistic_home_win_prob") == pytest.approx(logistic_prob)
