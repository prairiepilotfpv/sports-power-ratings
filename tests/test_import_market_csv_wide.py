import csv
from pathlib import Path
import sqlite3
import tempfile
from datetime import date

from src.data import betting_repository as br
from src.data import repository as repo


def test_import_market_csv_wide_format():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
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
        csv_path = Path(td) / "markets_wide.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "game_date",
                    "home_team",
                    "away_team",
                    "home_ml",
                    "away_ml",
                    "home_spread",
                    "home_spread_odds",
                    "away_spread",
                    "away_spread_odds",
                    "total",
                    "over_odds",
                    "under_odds",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "game_date": "2025-11-10",
                    "home_team": "Los Angeles Lakers",
                    "away_team": "LA Clippers",
                    "home_ml": "+110",
                    "away_ml": "-130",
                    "home_spread": "-3.5",
                    "home_spread_odds": "-110",
                    "away_spread": "3.5",
                    "away_spread_odds": "-110",
                    "total": "210.5",
                    "over_odds": "-105",
                    "under_odds": "-115",
                }
            )

        res = br.import_market_csv(
            db_path,
            csv_path=csv_path,
            snapshot_run_id="run-wide",
            sport="nba",
            season="2025-26",
        )
        assert res["committed"] >= 6

        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                "SELECT market_type, selection FROM market_snapshots WHERE snapshot_run_id = ?",
                ("run-wide",),
            ).fetchall()
        finally:
            conn.close()
        types = [r[0] for r in rows]
        assert types.count("ML") == 2
        assert types.count("spread") == 2
        assert types.count("total") == 2
        selections = {r[1] for r in rows}
        assert {"Over", "Under"}.issubset(selections)
