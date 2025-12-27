from datetime import date

from data.repository import list_seasons, list_sports, load_games, save_games
from ingest.schema import GameResult


def test_load_games_and_discovery_functions(tmp_path) -> None:
    db_path = tmp_path / "games.db"
    games = [
        GameResult(
            date=date(2024, 1, 1),
            home_team="Boston Celtics",
            away_team="Atlanta Hawks",
            home_score=120,
            away_score=110,
            overtime=False,
            game_id="202401010BOS",
            sport="nba",
            season="2023-24",
        ),
        GameResult(
            date=date(2024, 1, 2),
            home_team="Detroit Pistons",
            away_team="Chicago Bulls",
            home_score=101,
            away_score=99,
            overtime=True,
            game_id="202401020DET",
            sport="nba",
            season="2023-24",
        ),
        GameResult(
            date=date(2023, 9, 10),
            home_team="Los Angeles Chargers",
            away_team="Miami Dolphins",
            home_score=34,
            away_score=36,
            overtime=False,
            game_id="202309100LAC",
            sport="nfl",
            season="2023",
        ),
    ]

    save_games(db_path, games)

    assert list_sports(db_path) == ["nba", "nfl"]
    assert list_seasons(db_path, "nba") == ["2023-24"]
    assert list_seasons(db_path, "nfl") == ["2023"]

    all_games = load_games(db_path)
    assert [game.game_id for game in all_games] == [
        "202309100LAC",
        "202401010BOS",
        "202401020DET",
    ]

    nba_games = load_games(db_path, sport="nba")
    assert {game.game_id for game in nba_games} == {
        "202401010BOS",
        "202401020DET",
    }

    season_games = load_games(db_path, sport="nba", season="2023-24")
    assert len(season_games) == 2
