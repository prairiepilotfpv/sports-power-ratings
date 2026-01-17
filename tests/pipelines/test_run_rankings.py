from __future__ import annotations

from datetime import date
import math
from pathlib import Path

import pandas as pd

from data.repository import load_games, load_model_metrics, save_games
from ingest.schema import GameResult
from pipelines.run_rankings import run_rankings


def test_run_rankings_smoke(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    games = [
        GameResult(
            date=date(2024, 1, 1),
            home_team="B",
            away_team="A",
            home_score=70,
            away_score=80,
            sport="nba",
            season="2023-24",
        ),
        GameResult(
            date=date(2024, 1, 2),
            home_team="A",
            away_team="B",
            home_score=75,
            away_score=65,
            sport="nba",
            season="2023-24",
        ),
    ]
    saved = save_games(db_path, games)
    assert saved >= len(games)
    assert load_games(db_path, sport="nba", season="2023-24")

    output_dir = tmp_path / "processed"
    output_path = run_rankings(
        db_path,
        sport="nba",
        season="2023-24",
        model="bradley-terry",
        output_path=output_dir,
    )

    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert set(df.columns) == {
        "team",
        "rating",
        "games",
        "params_source",
        "tuned_metric_used",
    }
    assert {"A", "B"}.issubset(set(df["team"]))
    # legacy `points` column removed; ensure `rating` exists and is finite
    assert df["rating"].notna().all()

    metrics = load_model_metrics(
        db_path, sport="nba", season="2023-24", model="bradley-terry"
    )
    assert metrics is not None
