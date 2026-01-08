import csv
from pathlib import Path
import tempfile
from datetime import date
import sqlite3

from src.data import betting_repository as br
from src.data import repository as repo


def test_import_market_csv_commits_and_stages():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        # seed games
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
        br.init_db(db_path)
        csv_path = Path(td) / "markets.csv"
        with open(csv_path, "w", newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=["market_type", "selection", "line", "odds", "team_home", "team_away", "game_date"])
            writer.writeheader()
            writer.writerow({"market_type": "ML", "selection": "Los Angeles Lakers", "line": 0.0, "odds": "+110", "team_home": "LA Lakers", "team_away": "Clippers", "game_date": "2025-11-10"})
            writer.writerow({"market_type": "ML", "selection": "LA Clippers", "line": 0.0, "odds": "O+120", "team_home": "Clippers", "team_away": "LA Lakers", "game_date": "2025-11-10"})
            writer.writerow({"market_type": "ML", "selection": "Unknown", "line": 0.0, "odds": "+++", "team_home": "Unknown", "team_away": "Foo", "game_date": "2025-11-10"})

        res = br.import_market_csv(db_path, csv_path=csv_path, snapshot_run_id="run-csv", sport="nba", season="2025-26")
        assert res["committed"] >= 1
        assert res["staged"] >= 1
        assert res["rejected"] >= 1

        conn = sqlite3.connect(db_path)
        try:
            ms = conn.execute("SELECT COUNT(*) FROM market_snapshots WHERE snapshot_run_id = ?", ("run-csv",)).fetchone()[0]
            assert ms >= 1
            staged = conn.execute("SELECT COUNT(*) FROM market_snapshot_staging WHERE match_status != 'committed'").fetchone()[0]
            assert staged >= 1
        finally:
            conn.close()
