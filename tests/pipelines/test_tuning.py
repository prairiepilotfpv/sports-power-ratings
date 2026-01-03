from __future__ import annotations

from pathlib import Path

from pipelines.tuning import run_tuning_pipeline


def test_tuning_pipeline_selects_best_params(tmp_path: Path) -> None:
    csv_path = Path("tests/fixtures/mini_nba.csv")
    grid_override = {
        "k_factor": [10.0, 20.0],
        "home_advantage": [0.0],
        "initial_rating": [1500.0],
        "min_rating": [1.0],
    }

    outputs = run_tuning_pipeline(
        csv_path=csv_path,
        model="elo",
        start_date="2024-10-24",
        end_date="2024-11-05",
        metric="log_loss",
        output_dir=tmp_path,
        grid_override=grid_override,
        require_improvement=False,
    )

    assert outputs.results.shape[0] == 2
    assert outputs.best_params
    assert outputs.metric == "log_loss"
    run_id = outputs.results.iloc[0]["run_id"]
    assert (tmp_path / f"tuning_results_{run_id}.csv").exists()
