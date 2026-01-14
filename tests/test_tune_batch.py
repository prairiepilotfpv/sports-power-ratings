from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipelines import tune_batch


class _DummyOutputs:
    def __init__(self, model: str, metric: str) -> None:
        self.model = model
        self.metric = metric
        self.best_score = 1.23
        self.output_dir = Path("/tmp/out")
        self.applied = True
        self.results = pd.DataFrame([{"run_id": f"run-{model}-{metric}"}])


def test_tune_batch_defaults_models_from_config(monkeypatch) -> None:
    config_called = []
    tuning_calls: list[tuple[str, str]] = []

    def fake_load_config(sport: str, season: str, path_override=None):
        config_called.append((sport, season))
        return {
            "sport": sport,
            "season": season,
            "markets": {
                "ML": {"models": ["elo", "gssd"]},
                "SPREAD": {"models": ["toor"]},
            },
        }

    def fake_run_tuning_pipeline(**kwargs):
        tuning_calls.append((kwargs["model"], kwargs["metric"]))
        return _DummyOutputs(kwargs["model"], kwargs["metric"])

    def fake_list_backtest_models():  # pragma: no cover - only used if config missing
        return ["fallback"]

    monkeypatch.setattr(tune_batch, "load_ensemble_config", fake_load_config)
    monkeypatch.setattr(tune_batch, "run_tuning_pipeline", fake_run_tuning_pipeline)
    monkeypatch.setattr(tune_batch, "list_backtest_models", fake_list_backtest_models)
    monkeypatch.setenv("SPR_TUNE_BATCH_BACKEND", "thread")

    results = tune_batch.run_tune_batch(
        sport="nba",
        season="2025-26",
        start_date="2024-11-01",
        end_date="2024-12-01",
        csv_path="/tmp/data.csv",
        db_path="/tmp/db.sqlite",
    )

    assert config_called == [("nba", "2025-26")]
    # Union of models across markets
    assert set(m for m, _ in tuning_calls) == {"elo", "gssd", "toor"}
    # Default metrics applied
    assert set(metric for _, metric in tuning_calls) == set(tune_batch.DEFAULT_BATCH_METRICS)

    leaderboard = tune_batch.summarize_tune_batch(results)
    assert set(leaderboard) == set(tune_batch.DEFAULT_BATCH_METRICS)


def test_tune_batch_splits_jobs_across_runs(monkeypatch) -> None:
    tuning_calls: list[tuple[str, str, int]] = []

    def fake_run_tuning_pipeline(**kwargs):
        tuning_calls.append((kwargs["model"], kwargs["metric"], kwargs["jobs"]))
        return _DummyOutputs(kwargs["model"], kwargs["metric"])

    def fake_list_backtest_models():
        return ["elo", "gssd"]

    monkeypatch.setattr(tune_batch, "run_tuning_pipeline", fake_run_tuning_pipeline)
    monkeypatch.setattr(tune_batch, "list_backtest_models", fake_list_backtest_models)
    monkeypatch.setenv("SPR_TUNE_BATCH_BACKEND", "thread")

    results = tune_batch.run_tune_batch(
        sport="nba",
        season="2025-26",
        start_date="2024-11-01",
        end_date="2024-12-01",
        csv_path="/tmp/data.csv",
        db_path="/tmp/db.sqlite",
        models=["elo", "gssd"],
        metrics=["log_loss", "mae_total"],
        jobs=4,
    )

    assert len(results) == 4
    assert all(jobs == 1 for _, _, jobs in tuning_calls)
