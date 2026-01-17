import sqlite3
from pathlib import Path
from src.data.repository import SCHEMA
from src.data.migrations import apply_migrations


def test_backfill_game_ids(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    # create schema
    conn.executescript(SCHEMA)
    conn.commit()
    # insert game without game_id
    conn.execute(
        "INSERT INTO games (date, home_team, away_team, sport, season) VALUES (?, ?, ?, ?, ?)",
        ("2024-11-01", "Los Angeles Lakers", "Boston Celtics", "nba", "2024-25"),
    )
    conn.commit()
    # apply migrations (should perform backfill)
    apply_migrations(conn)
    row = conn.execute("SELECT game_id FROM games WHERE id = 1").fetchone()
    assert row and row[0] and row[0] != ""
    # ensure id is stable by recomputing
    from src.utils.game_id import make_game_id
    gid = make_game_id("nba", "2024-25", "2024-11-01", "Boston Celtics", "Los Angeles Lakers")
    # note make_game_id expects away then home; our inserted row used away=Boston, home=Los Angeles
    assert gid == row[0]
