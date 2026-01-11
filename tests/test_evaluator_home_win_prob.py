from __future__ import annotations

from src.eval.evaluator import _home_win_prob


def test_home_win_prob_prefers_calibrated_fields() -> None:
    row = {
        "model_p_home_win": 0.55,
        "home_win_prob": 0.56,
        "home_win_prob_calibrated": 0.62,
        "p_home_win_calibrated": 0.64,
    }
    assert _home_win_prob(row) == 0.64


def test_home_win_prob_falls_back_to_raw() -> None:
    row = {
        "model_p_home_win": 0.51,
        "home_win_prob": 0.52,
        "normal_p_home_win": 0.53,
        "logistic_home_win_prob": 0.54,
    }
    assert _home_win_prob(row) == 0.51
