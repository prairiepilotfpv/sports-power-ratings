from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from data.repository import (
    init_db,
    save_games,
    save_model_market_tuning_run,
    set_active_model_market_params,
)
from ingest.schema import GameResult
from pipelines.market_tuning import _resolve_market_metric
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


def test_schedule_uses_active_market_params(tmp_path: Path) -> None:
    db_path = tmp_path / "games.db"
    init_db(db_path)

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

    _, metric_optimized = _resolve_market_metric("ML", None)
    run_id = "run-ml-1"
    save_model_market_tuning_run(
        db_path,
        sport="nba",
        season="2024-25",
        model="elo",
        market="ML",
        metric_optimized=metric_optimized,
        run_id=run_id,
        best_score=1.0,
        best_params_json=json.dumps({"k_factor": 12}),
        summary_metrics_json=None,
        started_at=None,
        finished_at=None,
        notes=None,
    )
    set_active_model_market_params(
        db_path,
        sport="nba",
        season="2024-25",
        model="elo",
        market="ML",
        params={"k_factor": 12},
        source_run_id=run_id,
    )

    schedule_path = build_schedule_with_projections(
        db_path,
        sport="nba",
        season="2024-25",
        model="elo",
        output_path=tmp_path / "schedule.csv",
    )
    df = pd.read_csv(schedule_path)

    assert set(df["params_source"]) == {"db_market_active"}
    assert set(df["tuned_metric_used"]) == {"log_loss"}
    assert set(df["params_market"]) == {"ML"}
    assert set(df["tuning_run_id"]) == {run_id}



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

    def activate_best_run(model_name: str, market: str, params: dict[str, float], suffix: str) -> str:
        metric_name, metric_optimized = _resolve_market_metric(market, None)
        run_id = f"{model_name}-{suffix}"
        save_model_market_tuning_run(
            db_path,
            sport="nba",
            season="2024-25",
            model=model_name,
            market=market,
            metric_optimized=metric_optimized,
            run_id=run_id,
            best_score=1.0,
            best_params_json=json.dumps(params),
            summary_metrics_json=None,
            started_at=None,
            finished_at=None,
            notes=None,
        )
        set_active_model_market_params(
            db_path,
            sport="nba",
            season="2024-25",
            model=model_name,
            market=market,
            params=params,
            source_run_id=run_id,
        )
        return metric_name

    elo_metric = activate_best_run("elo", "ML", {"k_factor": 10.0}, "ml")
    gssd_metric = activate_best_run("gssd", "ML", {"recency_lambda": 0.5}, "ml")

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
    assert set(elo_df["params_source"]) == {"db_market_active"}
    assert set(elo_df["tuned_metric_used"]) == {elo_metric}
    assert set(elo_df["params_market"]) == {"ML"}
    assert set(gssd_df["params_source"]) == {"db_market_active"}
    assert set(gssd_df["tuned_metric_used"]) == {gssd_metric}
    assert set(gssd_df["params_market"]) == {"ML"}
