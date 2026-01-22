"""Tests for the new forecast contract and BT native integration."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from config import DEFAULT_WIN_PROB_K
from models.forecast_contract import (
    ForecastContract,
    MLForecast,
    SpreadForecast,
    TotalForecast,
)
from pipelines.matchups import team_home_advantages
from pipelines.projection_engines import get_projection_engine
from pipelines.schedule import _project_row, _rating_lookup
from pipelines.run_rankings import build_rankings


def test_forecast_contract_serialization() -> None:
    contract = ForecastContract(
        game_id="game-42",
        date="2025-11-01",
        home_team="Tigers",
        away_team="Bears",
        model_id="bradley-terry",
        model_version="1.0",
        metadata={"model_id": "bradley-terry", "params": {}},
        ml=MLForecast(p_home_win=0.61, p_away_win=0.39, source="native"),
        spread=SpreadForecast(margin_mean=3.5, margin_sd=12.0, source="native"),
        total=TotalForecast(total_mean=205.0, total_sd=10.0, source="native"),
        source_ml="native",
        source_spread="native",
        source_total="native",
        projected_home_score=106.2,
        projected_away_score=102.7,
        projected_total=208.9,
        warnings=["test-warning"],
    )
    payload = contract.to_dict()
    assert payload["game_id"] == "game-42"
    assert payload["ml"]["p_home_win"] == pytest.approx(0.61)
    payload_json = contract.to_json()
    assert "ml" in payload_json
    assert "spread" in payload_json


def test_bt_native_forecast_does_not_change_schedule_row() -> None:
    played = pd.DataFrame(
        [
            {
                "date": "2025-11-01",
                "home_team": "Lakers",
                "away_team": "Celtics",
                "home_score": 110,
                "away_score": 105,
            },
            {
                "date": "2025-11-02",
                "home_team": "Celtics",
                "away_team": "Lakers",
                "home_score": 108,
                "away_score": 101,
            },
        ]
    )
    rankings, model_instance = build_rankings(
        played,
        model="bradley-terry",
        include_implied_points=True,
        return_model=True,
    )
    ratings = _rating_lookup(rankings)
    home = "Lakers"
    away = "Celtics"
    projection_engine = get_projection_engine(model_instance)
    projection_context: dict[str, Any] = {
        "ratings": ratings,
        "base_total": 0.0,
        "scoring_averages": {},
        "total_intercept": 0.0,
        "total_slope": 0.0,
        "margin_std": None,
        "total_std": None,
        "conditional_sd_intercept": None,
        "conditional_sd_slope": None,
        "win_prob_k": DEFAULT_WIN_PROB_K,
        "winprob_bias": 0.0,
        "sport": "nba",
        "sd_sample_size": 0,
        "sd_residual_min": None,
        "sd_residual_max": None,
        "rating_units": "points",
    }
    home_advantages = team_home_advantages(played, ratings)
    upcoming_row = pd.Series(
        {
            "date": "2026-01-20",
            "home_team": home,
            "away_team": away,
            "neutral": False,
            "game_id": "test-matchup",
        }
    )

    scheduled = _project_row(
        upcoming_row,
        ratings=ratings,
        status="scheduled",
        home_advantage=home_advantages.get(home, 0.0),
        params_source="local-test",
        params_source_label="local-test",
        params_source_run_id="run-1",
        tuned_metric_used=None,
        params_metric_optimized=None,
        params_best_score=None,
        params_fingerprint=None,
        params_nonempty=True,
        params_run_id="run-1",
        params_market="ml",
        model_instance=model_instance,
        projection_engine=projection_engine,
        projection_context=projection_context,
    )

    native_projection = model_instance.project_matchup(home, away, neutral=False)

    assert scheduled["margin_mean"] == pytest.approx(native_projection["margin_mean"])
    assert scheduled["margin_sd"] == pytest.approx(native_projection["margin_sd"])
    assert scheduled["total_mean"] == pytest.approx(native_projection["total_mean"])
    assert scheduled["total_sd"] == pytest.approx(native_projection["total_sd"])
    assert scheduled["projected_home_score"] == pytest.approx(
        native_projection["projected_home_score"]
    )
    assert scheduled["projected_away_score"] == pytest.approx(
        native_projection["projected_away_score"]
    )
    home_prob = native_projection.get("model_p_home_win")
    assert home_prob is not None
    assert scheduled["home_win_prob"] == pytest.approx(home_prob)
    assert scheduled["away_win_prob"] == pytest.approx(1.0 - home_prob)