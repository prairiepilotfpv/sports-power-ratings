import csv
import sqlite3
from datetime import date

from src.data import market_lines as ml
from src.data import repository as repo
from src.data import teams as team_repo


def test_normalize_game_date_variants():
    assert ml.normalize_game_date("1/15/2026") == "2026-01-15"
    assert ml.normalize_game_date("2026-01-15") == "2026-01-15"
    assert ml.normalize_game_date("2026/01/15T19:00:00") == "2026-01-15"


def test_team_aliases_populated_from_games(tmp_path):
    db_path = tmp_path / "test.db"
    repo.init_db(db_path)
    repo.save_games(
        db_path,
        [
            repo.GameResult(
                date=date(2025, 11, 10),
                home_team="Los Angeles Lakers",
                away_team="LA Clippers",
                home_score=None,
                away_score=None,
                neutral=False,
                overtime=False,
                decision_type=None,
                game_id="2025-11-10-lakers-clippers",
                sport="nba",
                season="2025-26",
                division=None,
                conference=None,
                notes=None,
            )
        ],
    )
    conn = sqlite3.connect(db_path)
    try:
        team_id = team_repo.resolve_team_id(conn, sport="nba", season="2025-26", raw_team_name="Lakers")
        assert team_id is not None
        team_name = team_repo.get_canonical_name(conn, team_id=team_id)
        assert team_name == "Los Angeles Lakers"
    finally:
        conn.close()


def test_import_market_lines_records_rows(tmp_path):
    db_path = tmp_path / "markets.db"
    repo.init_db(db_path)
    repo.save_games(
        db_path,
        [
            repo.GameResult(
                date=date(2026, 1, 15),
                home_team="Los Angeles Lakers",
                away_team="LA Clippers",
                home_score=None,
                away_score=None,
                neutral=False,
                overtime=False,
                decision_type=None,
                game_id="2026-01-15-lakers-clippers",
                sport="nba",
                season="2025-26",
                division=None,
                conference=None,
                notes=None,
            )
        ],
    )
    csv_path = tmp_path / "markets.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "team_home_raw",
                "team_away_raw",
                "game_date",
                "market_type",
                "selection",
                "line",
                "odds",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "team_home_raw": "Los Angeles Lakers",
                "team_away_raw": "LA Clippers",
                "game_date": "1/15/2026",
                "market_type": "ML",
                "selection": "Lakers",
                "line": "0",
                "odds": "+110",
            }
        )
        writer.writerow(
            {
                "team_home_raw": "Los Angeles Lakers",
                "team_away_raw": "LA Clippers",
                "game_date": "2026-01-15",
                "market_type": "spread",
                "selection": "LA Clippers",
                "line": "3.5",
                "odds": "-110",
            }
        )
        writer.writerow(
            {
                "team_home_raw": "Los Angeles Lakers",
                "team_away_raw": "LA Clippers",
                "game_date": "2026-01-15",
                "market_type": "total",
                "selection": "O",
                "line": "210.5",
                "odds": "-105",
            }
        )
        writer.writerow(
            {
                "team_home_raw": "Los Angeles Lakers",
                "team_away_raw": "LA Clippers",
                "game_date": "2026-01-15",
                "market_type": "ML",
                "selection": "Unknown Team",
                "line": "0",
                "odds": "+120",
            }
        )

    result = ml.import_market_csv(db_path, csv_path=csv_path, sport="nba", season="2025-26", default_book="DK")
    assert result["rows_loaded"] == 4
    assert result["inserted"] == 3
    assert result["unmatched"] == 1
    assert result["unmatched_reasons"].get("team_unmatched") == 1
    assert result["unmatched_examples"]
    assert result["date_filtered"] == 0

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT market_type, selection_team_id, selection, line, odds FROM market_lines"
        ).fetchall()
        assert len(rows) == 3
        ml_row = next(r for r in rows if r[0] == "ML")
        assert ml_row[2] == "Los Angeles Lakers"
        assert ml_row[3] is None
        assert ml_row[4] == 110
        spread_row = next(r for r in rows if r[0] == "spread")
        assert spread_row[3] == 3.5
        assert spread_row[4] == -110
        total_row = next(r for r in rows if r[0] == "total")
        assert total_row[1] is None
        assert total_row[2] == "Over"
        assert total_row[3] == 210.5
        assert total_row[4] == -105
        errors = conn.execute("SELECT failure_reason FROM market_line_import_errors").fetchall()
        assert errors
    finally:
        conn.close()
