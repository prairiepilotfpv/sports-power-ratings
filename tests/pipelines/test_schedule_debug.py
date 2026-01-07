from __future__ import annotations

from datetime import date
from pathlib import Path

from ingest.schema import GameResult
from data.repository import save_games, load_games
from pipelines.schedule import build_schedule_with_projections
import pandas as pd


def test_debug_schedule_projection(tmp_path: Path) -> None:
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

    out_path = build_schedule_with_projections(db_path, sport="nba", season="2024-25", model="bradley-terry")
    df = pd.read_csv(out_path)
    scheduled = df[df["status"] == "scheduled"].iloc[0]
    print("PROJECTION ROW:", scheduled.to_dict())
    # Keep a trivial assertion so test doesn't just print
    assert "home_team" in scheduled
