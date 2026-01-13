import json
from pathlib import Path
import pandas as pd
from src.pipelines.tuning import run_tuning_pipeline


def _make_small_games_csv(path: Path) -> Path:
    df = pd.DataFrame(
        [
            {"date": "2024-11-01", "home_team": "A", "away_team": "B", "home_score": 100, "away_score": 90},
            {"date": "2024-11-02", "home_team": "B", "away_team": "A", "home_score": 95, "away_score": 97},
            {"date": "2024-11-03", "home_team": "A", "away_team": "B", "home_score": 102, "away_score": 99},
            {"date": "2024-11-04", "home_team": "B", "away_team": "A", "home_score": 88, "away_score": 90},
        ]
    )
    p = path / "small_games.csv"
    df.to_csv(p, index=False)
    return p


def _cmp_results(a, b):
    # drop run_id and output_dir which embed timestamps/paths
    drop_cols = [c for c in ["run_id", "output_dir"] if c in a.columns]
    A = a.drop(columns=drop_cols, errors="ignore").sort_values(by=["params_json"]).reset_index(drop=True)
    B = b.drop(columns=drop_cols, errors="ignore").sort_values(by=["params_json"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(A, B, check_dtype=False)


def test_serial_vs_parallel_equivalence(tmp_path):
    csv_path = _make_small_games_csv(tmp_path)
    grid_override = {"k_factor": [10.0, 20.0], "home_advantage": [0.0]}

    out1 = run_tuning_pipeline(
        csv_path=csv_path,
        model="elo",
        start_date="2024-11-01",
        end_date="2024-11-04",
        metric="log_loss",
        grid_override=grid_override,
        output_dir=tmp_path / "out1",
        jobs=1,
    )

    out2 = run_tuning_pipeline(
        csv_path=csv_path,
        model="elo",
        start_date="2024-11-01",
        end_date="2024-11-04",
        metric="log_loss",
        grid_override=grid_override,
        output_dir=tmp_path / "out2",
        jobs=2,
    )

    assert out1.best_params == out2.best_params
    assert out1.best_score == out2.best_score
    _cmp_results(out1.results, out2.results)
