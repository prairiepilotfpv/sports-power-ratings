from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from data.repository import load_games, save_games
from ingest.schema import GameResult
from pipelines.run_rankings import run_rankings


def test_run_rankings_smoke(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    games = [
        GameResult(
            date=date(2024, 1, 1),
            visitor_team="A",
            visitor_pts=80,
            home_team="B",
            home_pts=70,
            sport="nba",
            season="2023-24",
        ),
        GameResult(
            date=date(2024, 1, 2),
            visitor_team="B",
            visitor_pts=65,
            home_team="A",
            home_pts=75,
            sport="nba",
            season="2023-24",
        ),
    ]
    saved = save_games(db_path, games)
    assert saved == 2
    assert load_games(db_path, sport="nba", season="2023-24")

    output_dir = tmp_path / "processed"
    output_path = run_rankings(
        db_path,
        sport="nba",
        season="2023-24",
        output_path=output_dir,
    )

    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert set(df.columns) == {"team", "rating", "games"}
    assert {"A", "B"}.issubset(set(df["team"]))
