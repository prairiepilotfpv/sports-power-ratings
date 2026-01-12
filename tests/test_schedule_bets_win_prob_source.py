from datetime import date

import pandas as pd

from src.pipelines.schedule import _build_bets_dataframe


def test_bets_dataframe_includes_win_prob_source(tmp_path):
    as_of = date(2025, 11, 10)
    # Build a minimal schedule_df with a single scheduled game on as_of date
    schedule_df = pd.DataFrame([
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
            "total": 210.5,
            "total_sd": 5.0,
            "ml_ensemble_components_json": "[]",
        }
    ])

    df = _build_bets_dataframe(
        schedule_df, model_name="m", as_of_date=as_of, review_run_id="rr1"
    )
    assert "win_prob_source" in df.columns
    # Check values propagate to the ML moneyline rows (first two rows are ML selections)
    ml_rows = df[df["market_type"] == "ML"]
    assert not ml_rows.empty
    assert ml_rows.iloc[0]["win_prob_source"] == "model_x"
