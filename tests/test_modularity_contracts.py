from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from data.repository import save_games
from ingest.schema import GameResult
from models.base import ModelMetadata
from models.registry import register_model, unregister_model
from pipelines.ingest import ingest_games
from pipelines.run_rankings import build_rankings
from pipelines.schedule import (
    build_schedule_excel_report,
    build_schedule_with_projections,
)


class DummyPowerRatingModel:
    def __init__(self) -> None:
        self.model_id = "dummy"
        self.model_version = "1.0"
        self.params = {"source": "dummy"}
        self.fitted = False

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=self.model_id,
            model_version=self.model_version,
            params=self.params,
            supports_margin=True,
            supports_total=False,
            supports_win_prob=True,
        )

    def fit(self, games: list[dict]) -> None:
        self.fitted = True

    def rankings(self) -> list[tuple[str, float]]:
        return [("Team A", 2.0), ("Team B", 1.0)]


class DummyIngestSource:
    name = "dummy"

    def __init__(self, games: list[GameResult]) -> None:
        self._games = games
        self.used: str | None = None

    def load_path(
        self,
        path: str | Path,
        *,
        sport: str | None = None,
        season: str | None = None,
        format_hint: str | None = None,
    ) -> list[GameResult]:
        self.used = "path"
        return self._games

    def load_text(
        self,
        text: str,
        *,
        sport: str | None = None,
        season: str | None = None,
    ) -> list[GameResult]:
        self.used = "text"
        return self._games


def test_dummy_model_can_run_through_rankings_pipeline() -> None:
    register_model("dummy", DummyPowerRatingModel)
    try:
        df = pd.DataFrame(
            [
                {
                    "date": "2024-01-01",
                    "home_team": "Team A",
                    "away_team": "Team B",
                    "home_score": 100,
                    "away_score": 90,
                }
            ]
        )
        rankings = build_rankings(df, model="dummy")
        assert "Team A" in set(rankings["team"])
        assert rankings.loc[rankings["team"] == "Team A", "rating"].iloc[0] == 2.0
    finally:
        unregister_model("dummy")


def test_output_writers_are_swappable(tmp_path: Path) -> None:
    db_path = tmp_path / "games.db"
    today = date.today()
    games = [
        GameResult(
            date=today - timedelta(days=1),
            home_team="Team A",
            away_team="Team B",
            home_score=100,
            away_score=90,
            sport="nba",
            season="2024-25",
        ),
        GameResult(
            date=today,
            home_team="Team A",
            away_team="Team B",
            home_score=None,
            away_score=None,
            sport="nba",
            season="2024-25",
        ),
    ]
    save_games(db_path, games)

    csv_path = build_schedule_with_projections(
        db_path,
        sport="nba",
        season="2024-25",
        model="elo",
        output_path=tmp_path / "schedule.csv",
    )
    xlsx_path = build_schedule_excel_report(
        db_path,
        sport="nba",
        season="2024-25",
        model="elo",
        output_path=tmp_path / "schedule.xlsx",
    )

    assert csv_path.exists()
    assert xlsx_path.exists()


def test_ingest_source_can_be_swapped() -> None:
    games = [
        GameResult(
            date=date.today(),
            home_team="Team A",
            away_team="Team B",
            home_score=100,
            away_score=90,
            sport="nba",
            season="2024-25",
        )
    ]
    source = DummyIngestSource(games)
    normalized = ingest_games(
        source,
        input_path=None,
        input_text="dummy",
        sport="nba",
        season="2024-25",
    )
    assert source.used == "text"
    assert len(normalized) == 1
    assert normalized[0].home_team == "Team A"
