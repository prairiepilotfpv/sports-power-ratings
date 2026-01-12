from datetime import date

import pandas as pd
import pytest

from ensemble.total_v1 import TotalWeightedAverageEnsemble
from src.pipelines.schedule import _build_bets_dataframe


def test_total_ensemble_applies_to_bets_dataframe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    as_of = date(2025, 1, 1)
    bets_schedule_df = pd.DataFrame(
        [
            {
                "date": as_of.isoformat(),
                "status": "scheduled",
                "game_id": "g1",
                "away_team": "A",
                "home_team": "H",
                "projected_home_score": 100.0,
                "projected_away_score": 90.0,
                "projected_total": 190.0,
                "home_win_prob": 0.55,
                "away_win_prob": 0.45,
                "win_prob_source": "direct",
                "margin_mean": 3.0,
                "margin_sd": 2.0,
                "total_mean": 190.0,
                "total_sd": 11.0,
                "ml_ensemble_components_json": "[]",
                "spread_source": "direct",
                "spread_ensemble_components_json": "[]",
                "total_source": "direct",
            }
        ]
    )
    forecast_df = pd.DataFrame(
        [
            {"game_id": "g1", "model_name": "a", "total_mean": 200.0, "total_sd": 10.0},
            {"game_id": "g1", "model_name": "b", "total_mean": 210.0, "total_sd": 14.0},
        ]
    )

    total_ensemble = TotalWeightedAverageEnsemble("TEST", "2025")
    total_mean_raw, total_sd_raw, components_json = total_ensemble.combine(forecast_df)
    assert total_mean_raw is not None
    mask = bets_schedule_df["game_id"] == "g1"
    bets_schedule_df.loc[mask, "total"] = total_mean_raw
    if total_sd_raw is not None:
        bets_schedule_df.loc[mask, "total_sd"] = total_sd_raw
    bets_schedule_df.loc[mask, "total_source"] = total_ensemble.ensemble_id
    bets_schedule_df.loc[mask, "total_ensemble_components_json"] = components_json

    bets_df = _build_bets_dataframe(
        bets_schedule_df,
        model_name="model-a",
        as_of_date=as_of,
        review_run_id="rr1",
    )

    total_rows = bets_df[bets_df["market_type"] == "total"]
    assert not total_rows.empty
    assert total_rows.iloc[0]["total"] == pytest.approx(total_mean_raw, rel=1e-6)
    assert total_rows.iloc[0]["total_sd"] == pytest.approx(total_sd_raw, rel=1e-6)
    assert total_rows.iloc[0]["total_source"] == "ensemble_total_v1"
    assert total_rows.iloc[0]["market_forecast_source"] == "ensemble_total_v1"
    assert total_rows.iloc[0]["total_ensemble_components_json"]

    ml_rows = bets_df[bets_df["market_type"] == "ML"]
    assert ml_rows.iloc[0]["total_ensemble_components_json"] == ""
    spread_rows = bets_df[bets_df["market_type"] == "spread"]
    assert spread_rows.iloc[0]["total_ensemble_components_json"] == ""
