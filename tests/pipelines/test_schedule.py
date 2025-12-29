from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from data.repository import load_games, save_games
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
    assert "notes" not in df.columns

    upcoming = df[df["status"] == "scheduled"].iloc[0]
    assert upcoming["home_team"] == "Team B"
    assert upcoming["away_team"] == "Team C"
    assert pd.notna(upcoming["projected_winner"])
    assert pd.notna(upcoming["projected_spread"])
    assert upcoming["projected_total"] > 0


def test_schedule_uses_latest_scores(tmp_path: Path) -> None:
    db_path = tmp_path / "games.db"
    game_id = "2024-01-03|Team A|Team B"
    initial_game = GameResult(
        date=date(2024, 1, 3),
        home_team="Team B",
        away_team="Team A",
        home_score=None,
        away_score=None,
        game_id=game_id,
        sport="nba",
        season="2024-25",
    )
    save_games(db_path, [initial_game])

    output_path = tmp_path / "schedule.csv"
    build_schedule_with_projections(
        db_path,
        sport="nba",
        season="2024-25",
        output_path=output_path,
    )
    initial_df = pd.read_csv(output_path)
    initial_row = initial_df.loc[initial_df["game_id"] == game_id].iloc[0]
    assert pd.isna(initial_row["home_score"])
    assert pd.isna(initial_row["away_score"])
    assert initial_row["status"] == "scheduled"

    updated_game = GameResult(
        date=date(2024, 1, 3),
        home_team="Team B",
        away_team="Team A",
        home_score=101,
        away_score=99,
        game_id=game_id,
        sport="nba",
        season="2024-25",
    )
    save_games(db_path, [updated_game])

    build_schedule_with_projections(
        db_path,
        sport="nba",
        season="2024-25",
        output_path=output_path,
    )
    updated_df = pd.read_csv(output_path)
    updated_row = updated_df.loc[updated_df["game_id"] == game_id].iloc[0]
    assert updated_row["home_score"] == 101
    assert updated_row["away_score"] == 99
    assert updated_row["status"] == "final"

    db_games = {game.game_id: game for game in load_games(db_path, sport="nba", season="2024-25")}
    assert db_games[game_id].home_score == updated_row["home_score"]
    assert db_games[game_id].away_score == updated_row["away_score"]
