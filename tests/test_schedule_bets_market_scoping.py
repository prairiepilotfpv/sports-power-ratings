from datetime import date

import pandas as pd

from src.pipelines.schedule import _build_bets_dataframe


def test_bets_dataframe_scopes_forecasts_by_market():
    as_of = date(2025, 11, 10)
    schedule_df = pd.DataFrame(
        [
            {
                "date": as_of.isoformat(),
                "status": "scheduled",
                "game_id": "g1",
                "away_team": "A",
                "home_team": "H",
                "home_win_prob": 0.6,
                "away_win_prob": 0.4,
                "win_prob_source": "model_x",
                "margin_mean": 3.0,
                "margin_sd": 1.2,
                "projected_home_score": 110.0,
                "projected_away_score": 100.0,
                "total_sd": 5.0,
                "ml_ensemble_components_json": "[]",
                "spread_source": "model_y",
                "spread_ensemble_components_json": "[]",
            }
        ]
    )

    df = _build_bets_dataframe(
        schedule_df, model_name="m", as_of_date=as_of, review_run_id="rr1"
    )

    ml_rows = df[df["market_type"] == "ML"]
    assert not ml_rows.empty
    assert ml_rows.iloc[0]["home_win_prob"] == 0.6
    assert ml_rows.iloc[0]["margin_mean"] == ""
    assert ml_rows.iloc[0]["total"] == ""

    spread_rows = df[df["market_type"] == "spread"]
    assert not spread_rows.empty
    assert spread_rows.iloc[0]["margin_mean"] == 3.0
    assert spread_rows.iloc[0]["home_win_prob"] == ""
    assert spread_rows.iloc[0]["total"] == ""

    total_rows = df[df["market_type"] == "total"]
    assert not total_rows.empty
    assert total_rows.iloc[0]["total"] == 210.0
    assert total_rows.iloc[0]["margin_mean"] == ""
    assert total_rows.iloc[0]["home_win_prob"] == ""
