from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from backtest.runner import run_backtest
from models.base import BaseModel
from models.registry import get_backtest_model, list_backtest_models

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "mini_nba.csv"


class SpyModel(BaseModel):
    def __init__(self, model: BaseModel, counter: Counter[str]) -> None:
        self._model = model
        self._counter = counter

    def metadata(self):
        return self._model.metadata()

    def fit(self, games_df):
        self._model.fit(games_df)

    def predict(self, upcoming_games_df):
        model_id = self.metadata().model_id
        self._counter[model_id] += 1
        return self._model.predict(upcoming_games_df)


def test_backtest_pipeline_runs_all_models(tmp_path, monkeypatch) -> None:
    games_df = pd.read_csv(FIXTURE_PATH)
    games_df["date"] = pd.to_datetime(games_df["date"]).dt.date

    dates = sorted(games_df["date"].unique())
    assert len(dates) > 1, "Fixture must include at least two dates."
    start_date = dates[15] if len(dates) > 15 else dates[1]
    start_date_str = start_date.isoformat()

    model_ids = []
    for model_name in list_backtest_models():
        model_cls = get_backtest_model(model_name)
        model_ids.append(model_cls().metadata().model_id)
    assert len(model_ids) == len(set(model_ids)), "Expected unique model_id values."

    predict_counts: Counter[str] = Counter()
    export_counts: Counter[str] = Counter()
    predictions_by_model: dict[str, pd.DataFrame] = {}

    def export_spy(outputs, *, run_id: str) -> None:
        model_id = outputs.predictions["model_id"].iloc[0]
        export_counts[model_id] += 1
        predictions_by_model[model_id] = outputs.predictions

    monkeypatch.setattr("backtest.runner.export_backtest_outputs", export_spy)

    for model_name in list_backtest_models():
        model_cls = get_backtest_model(model_name)
        expected_model_id = model_cls().metadata().model_id

        def factory(model_cls=model_cls) -> BaseModel:
            return SpyModel(model_cls(), predict_counts)

        outputs = run_backtest(
            model_factory=factory,
            games_df=games_df,
            start_date=start_date_str,
            end_date=start_date_str,
            output_dir=tmp_path / model_name,
            model_name=model_name,
        )

        assert (
            not outputs.predictions.empty
        ), "Expected predictions for pipeline output."
        assert outputs.predictions["model_id"].nunique() == 1
        assert outputs.predictions["model_id"].iloc[0] == expected_model_id

    assert set(predict_counts.keys()) == set(model_ids)
    for model_id in model_ids:
        assert predict_counts[model_id] == 1
        assert export_counts[model_id] == 1

    prediction_frames = list(predictions_by_model.values())
    assert len(prediction_frames) == len(model_ids)
    assert len({id(frame) for frame in prediction_frames}) == len(prediction_frames)
