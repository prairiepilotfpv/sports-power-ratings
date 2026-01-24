import numpy as np
import pandas as pd
import pytest

from typing import Any

from pipelines.projection_engines import _poisson_projection_engine
from models.gssd import GSSDModel
from models.poisson import PoissonModel
from models.toor import TOORModel
from models.zsd import ZSDModel


def _training_games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2025-10-25",
                "home_team": "Home",
                "away_team": "Away",
                "home_score": 112,
                "away_score": 105,
            },
            {
                "date": "2025-10-26",
                "home_team": "Away",
                "away_team": "Home",
                "home_score": 99,
                "away_score": 101,
            },
            {
                "date": "2025-10-27",
                "home_team": "Home",
                "away_team": "Away",
                "home_score": 108,
                "away_score": 110,
            },
        ]
    )


def _upcoming_game() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-01-05",
                "home_team": "Home",
                "away_team": "Away",
            }
        ]
    )


def _assert_projection_matches_prediction(canonical: dict[str, float], prediction: Any) -> None:
    assert canonical["model_p_home_win"] == pytest.approx(prediction.p_home_win, rel=1e-9)
    assert canonical["pred_margin"] == pytest.approx(prediction.pred_margin)
    assert canonical["pred_total"] == pytest.approx(prediction.pred_total)
    assert canonical["margin_mean"] == pytest.approx(prediction.pred_margin)
    assert canonical["total_mean"] == pytest.approx(prediction.pred_total)
    assert canonical["pred_margin"] is not None
    assert canonical["pred_total"] is not None


def test_toor_projection_helper_matches_backtest_output() -> None:
    model = TOORModel()
    model.fit(_training_games())
    prediction = model.predict(_upcoming_game())[0]
    canonical = model.project_matchup(
        prediction.home_team,
        prediction.away_team,
        neutral=False,
        sport="nba",
        date=prediction.date,
        game_id=prediction.game_id,
    )
    _assert_projection_matches_prediction(canonical, prediction)


def test_gssd_projection_helper_matches_backtest_output() -> None:
    model = GSSDModel()
    model.fit(_training_games())
    prediction = model.predict(_upcoming_game())[0]
    canonical = model.project_matchup(
        prediction.home_team,
        prediction.away_team,
        neutral=False,
        sport="nba",
        date=prediction.date,
        game_id=prediction.game_id,
    )
    _assert_projection_matches_prediction(canonical, prediction)


def test_zsd_projection_helper_matches_backtest_output() -> None:
    model = ZSDModel(random_seed=11)
    model.fit(_training_games())
    prediction = model.predict(_upcoming_game())[0]
    canonical = model.project_matchup(
        prediction.home_team,
        prediction.away_team,
        neutral=False,
        sport="nba",
        date=prediction.date,
        game_id=prediction.game_id,
    )
    _assert_projection_matches_prediction(canonical, prediction)


def test_poisson_projection_engine_and_prediction_share_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    model = PoissonModel(random_seed=42, n_simulations=10)
    model.fit(_training_games())
    samples = (
        np.array([2.0, 2.0, 3.0, 1.0], dtype=float),
        np.array([1.0, 1.0, 0.0, 2.0], dtype=float),
    )

    def _fixed_samples(*args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        return samples

    monkeypatch.setattr(model._rating_model, "simulate_matchup", _fixed_samples)
    prediction = model.predict(_upcoming_game())[0]
    projection = _poisson_projection_engine(
        prediction.home_team,
        prediction.away_team,
        model._rating_model,
        {"neutral": False, "n_simulations": 4},
    )

    assert projection["model_p_home_win"] == pytest.approx(prediction.p_home_win)
    assert projection["margin_mean"] == pytest.approx(prediction.pred_margin)
    assert projection["total_mean"] == pytest.approx(prediction.pred_total)
    assert projection["win_prob_source"] == "sample"
