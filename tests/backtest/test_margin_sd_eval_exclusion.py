from __future__ import annotations

import pandas as pd

from models.base import GamePrediction, ModelMetadata, BaseModel
from backtest.runner import run_backtest, ensure_eval_schema


class DummyModel(BaseModel):
    def metadata(self):
        return ModelMetadata(
            model_id="dummy",
            model_version="1",
            params={},
            supports_margin=True,
            supports_total=True,
            supports_win_prob=True,
        )

    def fit(self, games_df: pd.DataFrame) -> None:
        return None

    def predict(self, upcoming_games_df: pd.DataFrame) -> list[GamePrediction]:
        preds: list[GamePrediction] = []
        for _, row in upcoming_games_df.iterrows():
            pred = GamePrediction(
                game_id=row.get("game_id") or "gid",
                date=row["date"],
                home_team=row["home_team"],
                away_team=row["away_team"],
                p_home_win=0.95,
                pred_margin=5.0,
                pred_total=195.0,
                margin_mean=5.0,
                margin_sd=1.0,
                total_mean=195.0,
                total_sd=3.0,
                win_prob_source="logistic",
                margin_dist_assumption="normal_approx",
                metadata={"model_id": "dummy", "model_version": "1", "params": {}},
            )
            preds.append(pred)
        return preds


def test_backtest_excludes_nba_margin_sd():
    # Build a tiny games dataset with one training row and one evaluation row
    games = pd.DataFrame(
        [
            {
                "date": pd.to_datetime("2023-12-31"),
                "home_team": "A",
                "away_team": "B",
                "home_score": 100,
                "away_score": 90,
                "game_id": "g-train",
            },
            {
                "date": pd.to_datetime("2024-01-01"),
                "home_team": "Home",
                "away_team": "Away",
                "home_score": None,
                "away_score": None,
                "game_id": "g-test",
            },
        ]
    )

    outputs = run_backtest(lambda: DummyModel(), games, sport="nba")

    # Raw predictions should be present; metrics should report zero scorable
    # games because outcomes are missing (no crash, stable schema).
    assert not outputs.predictions.empty
    row = outputs.metrics_overall.iloc[0]
    assert int(row.get("ml_games", 0)) == 0
    assert int(row.get("margin_games", 0)) == 0
    assert int(row.get("total_games", 0)) == 0
    assert pd.isna(row.get("mae_margin"))
    assert pd.isna(row.get("mae_total"))


def test_ensure_eval_schema_produces_required_columns():
    base = pd.DataFrame(
        [
            {
                "date": pd.to_datetime("2025-01-01"),
                "home_team": "H",
                "away_team": "A",
                # intentionally omit scores and predictions
            }
        ]
    )

    framed = ensure_eval_schema(base)

    required = {
        "home_score",
        "away_score",
        "actual_margin",
        "actual_total",
        "home_win",
        "pred_margin",
        "pred_total",
        "p_home_win",
        "margin_sd",
        "total_sd",
        "total_mean",
        "margin_mean",
    }

    assert required.issubset(set(framed.columns))
    # All derived columns should be NA when inputs are missing
    for col in required:
        assert framed[col].isna().all()
