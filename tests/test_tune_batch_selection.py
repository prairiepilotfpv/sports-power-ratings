from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipelines import tune_batch


class DummyOutputs:
    def __init__(self, run_id: str):
        self.best_score = 0.0
        self.output_dir = Path("outputs/tuning/mock")
        self.results = pd.DataFrame([{"run_id": run_id}])
        self.applied = True


def test_tune_batch_excludes_experimental_by_default(monkeypatch, tmp_path):
    calls: list[str] = []

    def fake_run_tuning_pipeline(**kwargs):
        calls.append(kwargs["model"])
        return DummyOutputs(run_id=f"run-{kwargs['model']}")

    config = {
        "markets": {
            "ML": {"models": ["bradley-terry", "bradley_terry_hfa"]},
            "SPREAD": {"models": ["elo"]},
        },
        "_meta": {},
    }

    monkeypatch.setattr(tune_batch, "run_tuning_pipeline", fake_run_tuning_pipeline)
    monkeypatch.setattr(tune_batch, "load_ensemble_config", lambda sport, season, available_models=None: config)
    monkeypatch.chdir(tmp_path)

    tune_batch.run_tune_batch(
        sport="nba",
        season="2025-26",
        start_date="2024-11-01",
        end_date="2024-12-01",
        csv_path=tmp_path / "games.csv",
        db_path=tmp_path / "db.sqlite",
        metrics=["log_loss"],
        include_experimental=False,
    )

    assert calls == ["bradley-terry", "elo"]


def test_tune_batch_includes_union_from_config(monkeypatch, tmp_path):
    calls: list[str] = []

    def fake_run_tuning_pipeline(**kwargs):
        calls.append(kwargs["model"])
        return DummyOutputs(run_id=f"run-{kwargs['model']}")

    config = {
        "markets": {
            "ML": {"models": ["elo"]},
            "SPREAD": {"models": ["gssd"]},
            "TOTAL": {"models": ["poisson"]},
        },
        "_meta": {},
    }

    monkeypatch.setattr(tune_batch, "run_tuning_pipeline", fake_run_tuning_pipeline)
    monkeypatch.setattr(tune_batch, "load_ensemble_config", lambda sport, season, available_models=None: config)
    monkeypatch.chdir(tmp_path)

    tune_batch.run_tune_batch(
        sport="nba",
        season="2025-26",
        start_date="2024-11-01",
        end_date="2024-12-01",
        csv_path=tmp_path / "games.csv",
        db_path=tmp_path / "db.sqlite",
        metrics=["log_loss"],
        include_experimental=True,
    )

    assert set(calls) == {"elo", "gssd", "poisson"}
