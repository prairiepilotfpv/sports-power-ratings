from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from data.repository import save_games
from ingest.schema import GameResult
from pipelines.schedule import build_schedule_with_projections


def test_build_schedule_with_projections(tmp_path: Path) -> None:
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
            home_team="Team C",
            away_team="Team A",
            home_score=95,
            away_score=110,
            sport="nba",
            season="2024-25",
        ),
        GameResult(
            date=date(2024, 1, 5),
            home_team="Team B",
            away_team="Team C",
            home_score=None,
            away_score=None,
            sport="nba",
            season="2024-25",
        ),
    ]
    save_games(db_path, games)

    output_path = build_schedule_with_projections(
        db_path,
        sport="nba",
        season="2024-25",
        output_path=tmp_path / "schedule.csv",
    )

    df = pd.read_csv(output_path)
    assert len(df) == 3
    assert set(df["status"]) == {"final", "scheduled"}

    upcoming = df[df["status"] == "scheduled"].iloc[0]
    assert upcoming["home_team"] == "Team B"
    assert upcoming["away_team"] == "Team C"
    assert pd.notna(upcoming["projected_winner"])
    assert pd.notna(upcoming["projected_spread"])
    assert upcoming["projected_total"] > 0
