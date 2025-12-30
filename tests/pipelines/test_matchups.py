from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from data.repository import save_games
from ingest.schema import GameResult
from pipelines.matchups import predict_matchup, team_home_advantages


def _seed_games(db_path: Path) -> None:
    games = [
        GameResult(
            date=date(2024, 1, 1),
            home_team="Alpha",
            away_team="Beta",
            home_score=100,
            away_score=90,
            sport="nba",
            season="2024-25",
        ),
        GameResult(
            date=date(2024, 1, 2),
            home_team="Beta",
            away_team="Alpha",
            home_score=95,
            away_score=102,
            sport="nba",
            season="2024-25",
        ),
        GameResult(
            date=date(2024, 1, 3),
            home_team="Gamma",
            away_team="Alpha",
            home_score=88,
            away_score=110,
            sport="nba",
            season="2024-25",
        ),
    ]
    save_games(db_path, games)


def test_predict_matchup_smoke(tmp_path: Path) -> None:
    db_path = tmp_path / "games.db"
    _seed_games(db_path)

    prediction = predict_matchup(
        db_path,
        sport="nba",
        season="2024-25",
        home_team="Alpha",
        away_team="Beta",
    )

    assert prediction.home_team == "Alpha"
    assert prediction.away_team == "Beta"
    assert prediction.winner in {"Alpha", "Beta"}
    assert prediction.total_points >= 0


def test_predict_matchup_unknown_team_raises(tmp_path: Path) -> None:
    db_path = tmp_path / "games.db"
    _seed_games(db_path)

    with pytest.raises(ValueError, match="Unknown team"):
        predict_matchup(
            db_path,
            sport="nba",
            season="2024-25",
            home_team="Alpha",
            away_team="Unknown",
        )


def test_team_home_advantages_skip_neutral(tmp_path: Path) -> None:
    games = [
        GameResult(
            date=date(2024, 1, 1),
            home_team="Alpha",
            away_team="Beta",
            home_score=100,
            away_score=90,
            neutral=True,
            sport="nba",
            season="2024-25",
        ),
        GameResult(
            date=date(2024, 1, 2),
            home_team="Alpha",
            away_team="Beta",
            home_score=98,
            away_score=95,
            neutral=False,
            sport="nba",
            season="2024-25",
        ),
    ]
    save_games(tmp_path / "games.db", games)
    ratings = {"Alpha": 2.0, "Beta": -1.0}

    advantages = team_home_advantages(
        df=pd.DataFrame([g.model_dump() for g in games]),
        ratings=ratings,
    )
    assert "Alpha" in advantages
