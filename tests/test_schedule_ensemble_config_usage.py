from __future__ import annotations

from datetime import date

import pandas as pd

from markets.base import Market
from pipelines import schedule


class _DummyResolution:
    def __init__(self) -> None:
        self.params = None
        self.params_source = "default"
        self.tuned_metric_used = None
        self.source_run_id = None
        self.market = Market.ML.name


def test_market_forecasts_respect_allowed_models(monkeypatch) -> None:
    """Test that _build_market_forecasts_for_ensembles respects allowed models per market.
    
    NOTE: This test passes forecast_params_by_model parameter that the function
    does not currently accept. This parameter is intended to pass pre-computed
    forecast parameters per model to avoid redundant computation during ensemble
    building. Either the function signature needs to be updated to accept this
    parameter, or the test should be updated to not pass it.
    """
    calls: list[str] = []

    def fake_build_schedule_dataframe(
        df,
        *,
        db_path,
        sport,
        season,
        model,
        upcoming_only,
        model_params,
        params_source,
        tuned_metric_used,
        forecast_params=None,
        params_run_id=None,
        params_market=Market.ML.name,
        **kwargs,
    ) -> pd.DataFrame:
        calls.append(model)
        return pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "home_win_prob": 0.6,
                    "model_p_home_win": 0.6,
                    "margin_mean": 1.0,
                    "margin_sd": 1.0,
                    "total_mean": 200.0,
                    "total_sd": 10.0,
                    "total": None,
                }
            ]
        )
    monkeypatch.setattr(schedule, "_build_schedule_dataframe", fake_build_schedule_dataframe)
    monkeypatch.setattr(schedule, "get_active_model_market_params", lambda *a, **k: None)
    monkeypatch.setattr(schedule, "resolve_model_params_with_metadata", lambda *a, **k: _DummyResolution())

    allowed = {
        Market.ML.name: ["elo"],
        Market.SPREAD.name: ["elo"],
        Market.TOTAL.name: ["elo"],
    }

    rows = schedule._build_market_forecasts_for_ensembles(
        pd.DataFrame([{"game_id": "g1"}]),
        db_path="/tmp/db",
        sport="nba",
        season="2025-26",
        models=["elo", "gssd"],
        as_of_date=date(2025, 1, 1),
        forecast_params_by_model={"elo": None, "gssd": None},
        allowed_models_by_market=allowed,
    )

    assert all(entry["model_name"] == "elo" for entry in rows[Market.ML.name])
    assert calls.count("elo") == 3  # once per market
    assert "gssd" not in calls
