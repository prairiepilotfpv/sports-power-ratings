from __future__ import annotations

import json
from pathlib import Path

from cli.pipeline import _run_tuning


def test_tune_all_models_and_metrics_summary(tmp_path: Path, monkeypatch) -> None:
    grid_override = {
        "elo": {
            "k_factor": [10.0],
            "home_advantage": [0.0],
            "initial_rating": [1500.0],
            "min_rating": [1.0],
        },
        "toor": {
            "max_iter": [200],
            "tol": [1e-6],
        },
    }
    grid_path = tmp_path / "grid.json"
    grid_path.write_text(json.dumps(grid_override), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    csv_path = Path(__file__).resolve().parents[1] / "fixtures" / "mini_nba.csv"

    monkeypatch.setattr(
        "models.registry.list_backtest_models", lambda: ["elo", "toor"]
    )
    monkeypatch.setattr(
        "pipelines.tuning.list_metrics", lambda: ["log_loss", "brier_score"]
    )

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
        csv = str(csv_path)
        grid_file = str(grid_path)
        apply_best = False
        apply_metric = "log_loss"
        allow_worse = True
        fail_fast = False
        sport = "nba"
        season = "2024-25"
        db = None

    _run_tuning(Args())

    for model in ("elo", "toor"):
        for metric in ("log_loss", "brier_score"):
            assert (tmp_path / model / metric).exists()

    summary_dir = tmp_path / "outputs" / "tuning" / "nba" / "2024-25" / "_all"
    summaries = list(summary_dir.glob("tune_summary_*.csv"))
    assert summaries
