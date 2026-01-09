import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from eval.validation import validate_prediction_row, NBA_VALIDATION_CONFIG
from eval.evaluator import evaluate_market_rows


def _base_prediction(**overrides):
    base = {
        "game_id": "g1",
        "date": "2025-11-10",
        "home_team": "Pelicans",
        "away_team": "Wizards",
        "model": "elo",
        "margin_mean": 6.0,
        "margin_sd": 12.0,
        "total_mean": 235.0,
        "total_sd": 20.0,
        "projected_home_score": 118.0,
        "projected_away_score": 117.0,
        "model_p_home_win": 0.6,
        "normal_p_home_win": 0.6,
        "margin_dist_assumption": "normal_approx",
    }
    base.update(overrides)
    return base


def test_toor_predictions_excluded_from_aggregation():
    preds = pd.DataFrame(
        [
            _base_prediction(model="toor", margin_sd=1.0),
            _base_prediction(model="elo", model_p_home_win=0.61),
        ]
    )
    markets = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "market_type": "ML",
                "selection": "Pelicans",
                "line": 0.0,
                "odds": -110,
                "home_team": "Pelicans",
                "away_team": "Wizards",
            }
        ]
    )

    opportunities, debug_df = evaluate_market_rows(preds, markets, debug=True)

    assert len(opportunities) == 1
    prob = opportunities.iloc[0]["model_prob"]
    assert prob == pytest.approx(0.61, rel=1e-6)
    assert set(debug_df.iloc[0]["per_model_probs"].keys()) == {"elo"}


def test_under_over_ev_signals_use_totals_and_poisson_weight():
    preds = pd.DataFrame(
        [
            _base_prediction(model="bradley-terry", total_mean=233.0, total_sd=20.0, projected_home_score=116.0, projected_away_score=117.0),
            _base_prediction(model="elo", total_mean=235.0, total_sd=20.0, projected_home_score=117.0, projected_away_score=118.0),
            _base_prediction(model="gssd", total_mean=236.0, total_sd=20.0, projected_home_score=118.0, projected_away_score=118.0),
            _base_prediction(model="poisson", total_mean=241.0, total_sd=15.0, projected_home_score=121.0, projected_away_score=120.0),
        ]
    )
    markets = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "market_type": "total",
                "selection": "Under",
                "line": 241.5,
                "odds": -105,
                "home_team": "Pelicans",
                "away_team": "Wizards",
            },
            {
                "game_id": "g1",
                "market_type": "total",
                "selection": "Over",
                "line": 242.5,
                "odds": -105,
                "home_team": "Pelicans",
                "away_team": "Wizards",
            },
        ]
    )

    opportunities, _ = evaluate_market_rows(preds, markets)
    under_edge = opportunities[opportunities["selection"] == "Under"]["edge"].iloc[0]
    over_edge = opportunities[opportunities["selection"] == "Over"]["edge"].iloc[0]
    assert under_edge is not None and under_edge > 0
    assert over_edge is not None and over_edge < 0


def test_sd_bounds_and_prob_bounds_enforced():
    bad_margin = _base_prediction(margin_sd=40.0)
    ok, reasons = validate_prediction_row(bad_margin, config=NBA_VALIDATION_CONFIG)
    assert ok is False
    assert "invalid_margin_sd" in reasons

    bad_total = _base_prediction(total_sd=5.0)
    ok_total, reasons_total = validate_prediction_row(bad_total, config=NBA_VALIDATION_CONFIG)
    assert ok_total is False
    assert "invalid_total_sd" in reasons_total

    bad_prob = _base_prediction(model_p_home_win=1.2)
    ok_prob, reasons_prob = validate_prediction_row(bad_prob, config=NBA_VALIDATION_CONFIG)
    assert ok_prob is False
    assert "prob_out_of_bounds" in reasons_prob


def test_spread_prob_ignores_projected_spread_field():
    preds = pd.DataFrame(
        [
            _base_prediction(model="elo", margin_mean=0.0, margin_sd=10.0, projected_spread=-20.0),
        ]
    )
    markets = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "market_type": "spread",
                "selection": "Pelicans",
                "line": 0.0,
                "odds": -110,
                "home_team": "Pelicans",
                "away_team": "Wizards",
            }
        ]
    )
    opportunities, _ = evaluate_market_rows(preds, markets)
    prob = opportunities.iloc[0]["model_prob"]
    assert prob == pytest.approx(0.5, rel=1e-6)


def test_evaluator_is_deterministic():
    preds = pd.DataFrame([_base_prediction(model="elo")])
    markets = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "market_type": "ML",
                "selection": "Pelicans",
                "line": 0.0,
                "odds": -120,
                "home_team": "Pelicans",
                "away_team": "Wizards",
            }
        ]
    )
    first, _ = evaluate_market_rows(preds, markets)
    second, _ = evaluate_market_rows(preds, markets)
    assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
