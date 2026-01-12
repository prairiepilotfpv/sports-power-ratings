from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.runner import run_backtest
from models.base import BaseModel, GamePrediction, ModelMetadata

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "mini_nba.csv"


class DummyTotalModel(BaseModel):
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id="dummy-total",
            model_version="v1",
            params={},
            supports_margin=True,
            supports_total=True,
            supports_win_prob=True,
        )

    def fit(self, games_df) -> None:
        return None

    def predict(self, upcoming_games_df) -> list[GamePrediction]:
        preds: list[GamePrediction] = []
        for row in upcoming_games_df.itertuples(index=False):
            preds.append(
                GamePrediction(
                    game_id=str(getattr(row, "game_id", "")),
                    date=str(getattr(row, "date")),
                    home_team=str(getattr(row, "home_team")),
                    away_team=str(getattr(row, "away_team")),
                    p_home_win=0.5,
                    pred_margin=0.0,
                    pred_total=200.0,
                    margin_sd=10.0,
                    total_sd=10.0,
                    metadata={
                        "model_id": "dummy-total",
                        "model_version": "v1",
                        "params": {},
                    },
                    extra={
                        "projected_home_score": 100.0,
                        "projected_away_score": 100.0,
                    },
                )
            )
        return preds


def test_backtest_includes_mae_total(tmp_path: Path) -> None:
    games_df = pd.read_csv(FIXTURE_PATH)
    games_df["date"] = pd.to_datetime(games_df["date"]).dt.date
    dates = sorted(games_df["date"].unique())
    start_date = dates[1].isoformat()
    outputs = run_backtest(
        model_factory=DummyTotalModel,
        games_df=games_df,
        start_date=start_date,
        end_date=start_date,
        output_dir=tmp_path / "dummy",
        model_name="dummy-total",
    )
    metrics = outputs.metrics_overall.iloc[0].to_dict()
    assert "mae_total" in metrics
    assert metrics["mae_total"] is not None
