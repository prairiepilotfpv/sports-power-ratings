import sqlite3
from src.data.repository import SCHEMA
from src.data.migrations import apply_migrations
from src.utils.game_id import make_game_id


def test_migrate_game_ids_and_update_market_lines(tmp_path):
    db_path = tmp_path / "migrate.db"
    conn = sqlite3.connect(db_path)
    # create initial schema
    conn.executescript(SCHEMA)
    conn.commit()
    # insert a game with legacy game_id
    conn.execute(
        "INSERT INTO games (date, home_team, away_team, game_id, sport, season) VALUES (?, ?, ?, ?, ?, ?)",
        ("2024-11-01", "Boston Celtics", "L.A. Lakers", "legacy-123", "nba", "2024-25"),
    )
    # insert market line referencing legacy id
    conn.execute(
        "INSERT INTO market_lines (sport, season, game_id, game_date, market_type, odds) VALUES (?, ?, ?, ?, ?, ?)",
        ("nba", "2024-25", "legacy-123", "2024-11-01", "ML", 100),
    )
    conn.commit()

    # run migrations (this will apply new migration)
    apply_migrations(conn)

    # check the game's legacy_game_id and new game_id
    row = conn.execute("SELECT legacy_game_id, game_id, home_team, away_team FROM games WHERE id = 1").fetchone()
    assert row is not None
    legacy, new_gid, home, away = row
    assert legacy == "legacy-123"
    expected = make_game_id("nba", "2024-25", "2024-11-01", "L.A. Lakers", "Boston Celtics")
    assert new_gid == expected

    # check market_lines updated
    mrow = conn.execute("SELECT game_id FROM market_lines WHERE id = 1").fetchone()
    assert mrow and mrow[0] == expected
