from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

from data.repository import load_active_tuned_metric, save_games
from ingest.schema import GameResult
from pipelines.schedule import build_schedule_with_projections
from pipelines.tuning import run_tuning_pipeline


class _FakeOutputs:
    def __init__(self, score: float) -> None:
        self.metrics_overall = pd.DataFrame(
            [
                {
                    "log_loss": score,
                    "brier_score": score,
                    "mae_margin": score,
                }
            ]
        )


def test_tuning_persists_params_in_db(tmp_path: Path, monkeypatch) -> None:
    scores = [2.0, 1.0, 1.0]

    def fake_run_backtest(*_args, **_kwargs):
        return _FakeOutputs(scores.pop(0))

    monkeypatch.setattr("pipelines.tuning.run_backtest", fake_run_backtest)

    db_path = tmp_path / "tuning.db"
    run_tuning_pipeline(
        csv_path=Path("tests/fixtures/mini_nba.csv"),
        model="elo",
        start_date="2024-10-24",
        end_date="2024-11-05",
        metric="log_loss",
        output_dir=tmp_path,
        grid_override={"k_factor": [10.0]},
        apply_best=True,
        db_path=db_path,
        sport="nba",
        season="2024-25",
    )

    with sqlite3.connect(db_path) as conn:
        tuned_row = conn.execute(
            """
            SELECT params_json, best_score
            FROM model_tuned_params
            WHERE sport = ? AND season = ? AND model = ? AND metric = ?
            """,
            ("nba", "2024-25", "elo", "log_loss"),
        ).fetchone()
        assert tuned_row is not None
        metrics_row = conn.execute(
            """
            SELECT tuned_params_json, tuned_params_metric
            FROM model_metrics
            WHERE sport = ? AND season = ? AND model = ?
            """,
            ("nba", "2024-25", "elo"),
        ).fetchone()
        assert metrics_row is not None
        assert metrics_row[0]
        assert metrics_row[1] == "log_loss"


def test_schedule_uses_tuned_params(tmp_path: Path, monkeypatch, capsys) -> None:
    scores = [2.0, 1.0, 1.0]

    def fake_run_backtest(*_args, **_kwargs):
        return _FakeOutputs(scores.pop(0))

    monkeypatch.setattr("pipelines.tuning.run_backtest", fake_run_backtest)

    db_path = tmp_path / "games.db"
    games = [
        GameResult(
            date=date(2024, 1, 1),
            home_team="Team A",
            away_team="Team B",
            home_score=100,
            away_score=90,
            sport="nba",
            season="2024-25",
        ),
        GameResult(
            date=date(2024, 1, 2),
            home_team="Team B",
            away_team="Team C",
            home_score=95,
            away_score=110,
            sport="nba",
            season="2024-25",
        ),
    ]
    save_games(db_path, games)

    run_tuning_pipeline(
        csv_path=Path("tests/fixtures/mini_nba.csv"),
        model="elo",
        start_date="2024-10-24",
        end_date="2024-11-05",
        metric="log_loss",
        output_dir=tmp_path,
        grid_override={"k_factor": [10.0]},
        apply_best=True,
        db_path=db_path,
        sport="nba",
        season="2024-25",
    )

    build_schedule_with_projections(
        db_path,
        sport="nba",
        season="2024-25",
        model="elo",
        output_path=tmp_path / "schedule.csv",
    )

    captured = capsys.readouterr()
    assert "Using tuned params from DB (metric=log_loss) for model=elo" in captured.out


def test_active_metric_policy_applies_for_multiple_models(
    tmp_path: Path, monkeypatch
) -> None:
    def fake_run_backtest(*_args, **_kwargs):
        output_dir = str(_kwargs.get("output_dir", ""))
        score = 2.0 if "__baseline" in output_dir else 1.0
        return _FakeOutputs(score)

    monkeypatch.setattr("pipelines.tuning.run_backtest", fake_run_backtest)
    monkeypatch.setattr(
        "models.registry.list_backtest_models", lambda: ["elo", "gssd"]
    )
    monkeypatch.setattr(
        "pipelines.tuning.list_metrics", lambda: ["log_loss", "mae_margin"]
    )

    db_path = tmp_path / "games.db"
    games = [
        GameResult(
            date=date(2024, 1, 1),
            home_team="Team A",
            away_team="Team B",
            home_score=100,
            away_score=90,
            sport="nba",
            season="2024-25",
        ),
        GameResult(
            date=date(2024, 1, 2),
            home_team="Team B",
            away_team="Team C",
            home_score=95,
            away_score=110,
            sport="nba",
            season="2024-25",
        ),
    ]
    save_games(db_path, games)

    grid_override = {
        "elo": {
            "k_factor": [10.0],
            "home_advantage": [0.0],
            "initial_rating": [1500.0],
            "min_rating": [1.0],
        },
        "gssd": {},
    }
    grid_path = tmp_path / "grid.json"
    grid_path.write_text(json.dumps(grid_override), encoding="utf-8")

    class Args:
        command = "tune"
        model = "all"
        metric = "all"
        start = "2024-10-24"
        end = "2024-11-05"
        window = "expanding"
        rolling_days = None
        rolling_games = None
        output_dir = str(tmp_path)
        csv = "tests/fixtures/mini_nba.csv"
        grid_file = str(grid_path)
        apply_best = True
        apply_metric = "log_loss"
        allow_worse = False
        fail_fast = True
        sport = "nba"
        season = "2024-25"
        db = str(db_path)

    from cli.pipeline import _run_tuning

    _run_tuning(Args())

    assert (
        load_active_tuned_metric(
            db_path, sport="nba", season="2024-25", model="elo"
        )
        == "log_loss"
    )
    assert (
        load_active_tuned_metric(
            db_path, sport="nba", season="2024-25", model="gssd"
        )
        == "mae_margin"
    )

    elo_path = build_schedule_with_projections(
        db_path,
        sport="nba",
        season="2024-25",
        model="elo",
        output_path=tmp_path / "elo_schedule.csv",
    )
    gssd_path = build_schedule_with_projections(
        db_path,
        sport="nba",
        season="2024-25",
        model="gssd",
        output_path=tmp_path / "gssd_schedule.csv",
    )

    elo_df = pd.read_csv(elo_path)
    gssd_df = pd.read_csv(gssd_path)
    assert set(elo_df["params_source"]) == {"db_active"}
    assert set(elo_df["tuned_metric_used"]) == {"log_loss"}
    assert set(gssd_df["params_source"]) == {"db_active"}
    assert set(gssd_df["tuned_metric_used"]) == {"mae_margin"}
