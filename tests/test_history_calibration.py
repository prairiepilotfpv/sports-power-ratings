import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import pytest

from data import repository as games_repo
from data import betting_repository as bets_repo
from markets.base import Market
from pipelines.history_calibration import (
    build_history_calibration_dataset,
    calibrate_market_from_history,
)


def _setup_db(path: Path) -> None:
    games_repo.init_db(path)
    bets_repo.init_db(path)


def _insert_game(
    conn: sqlite3.Connection,
    *,
    game_id: str,
    date: str,
    home_score: int | None = None,
    away_score: int | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO games (date, home_team, away_team, sport, season, game_id, home_score, away_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (date, "HomeTeam", "AwayTeam", "nba", "2025-26", game_id, home_score, away_score),
    )


def _insert_prediction(
    conn: sqlite3.Connection,
    *,
    game_id: str,
    source: str,
    market_type: str,
    selection: str,
    line: float,
    model_prob: float,
    prediction_date: str,
) -> None:
    # bets_predictions stores the forecast data (not stake outcomes), which mirrors the
    # source history calibrators read for each market.
    conn.execute(
        """
        INSERT INTO bets_predictions (
            game_id, sport, season, prediction_date,
            home_win_prob, model_prob, market_type,
            selection, line, market_forecast_source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            game_id,
            "nba",
            "2025-26",
            prediction_date,
            model_prob,
            model_prob,
            market_type,
            selection,
            line,
            source,
        ),
    )


def test_build_history_calibration_dataset(tmp_path: Path) -> None:
    db_path = tmp_path / "bets.db"
    _setup_db(db_path)
    with sqlite3.connect(db_path) as conn:
        _insert_game(conn, game_id="game-1", date="2025-01-10", home_score=110, away_score=100)
        _insert_prediction(
            conn,
            game_id="game-1",
            source="spread_ensemble_v1",
            market_type="spread",
            selection="HomeTeam",
            line=-3.5,
            model_prob=0.56,
            prediction_date="2025-01-05",
        )
        conn.commit()

    dataset = build_history_calibration_dataset(
        db_path=db_path,
        sport="nba",
        season="2025-26",
        market=Market.SPREAD,
        source="spread_ensemble_v1",
        start_date="2025-01-01",
        end_date="2025-01-31",
    )
    assert len(dataset) == 1
    assert dataset["home_win"].iloc[0] == 1


def test_calibrate_market_from_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "bets.db"
    _setup_db(db_path)
    with sqlite3.connect(db_path) as conn:
        _insert_game(conn, game_id="game-2", date="2025-02-01", home_score=110, away_score=100)
        _insert_prediction(
            conn,
            game_id="game-2",
            source="total_ensemble_v1",
            market_type="total",
            selection="Over",
            line=210.5,
            model_prob=0.62,
            prediction_date="2025-02-01",
        )
        _insert_game(conn, game_id="game-3", date="2025-02-02", home_score=100, away_score=95)
        _insert_prediction(
            conn,
            game_id="game-3",
            source="total_ensemble_v1",
            market_type="total",
            selection="Under",
            line=210.5,
            model_prob=0.41,
            prediction_date="2025-02-02",
        )
        conn.commit()

    out_path = calibrate_market_from_history(
        db_path=db_path,
        sport="nba",
        season="2025-26",
        market=Market.TOTAL,
        source="total_ensemble_v1",
        start_date="2025-01-01",
        end_date="2025-12-31",
        method="platt",
    )
    assert out_path.exists()
