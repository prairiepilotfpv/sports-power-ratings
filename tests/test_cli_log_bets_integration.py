import sys
from pathlib import Path
import tempfile
import sqlite3

from src.cli import pipeline as pl
from src.data import repository as repo
from src.data import betting_repository as br
import pandas as pd


def build_wb(path, review_run_id):
    df_meta = pd.DataFrame([{"key": "review_run_id", "value": review_run_id}])
    df_bets = pd.DataFrame(
        [
            {"game_id": "2025-11-10-lakers-clippers", "market_type": "ML", "selection": "Los Angeles Lakers", "line": 0.0, "odds": 110, "stake": 10.0, "book": "BookA"}
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_bets.to_excel(writer, sheet_name="BETS", index=False)
        df_meta.to_excel(writer, sheet_name="META", index=False)


def test_cli_log_bets_integration(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        repo.init_db(db_path)
        br.init_db(db_path)
        rid = br.create_review_run(db_path, sport="nba", season="2025-26", model="elo")
        # seed game
        repo.save_games(
            db_path,
            [
                repo.GameResult(
                    date=repo.date.fromisoformat("2025-11-10"),
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
        wb = Path(td) / "wb.xlsx"
        build_wb(wb, rid)
        monkeypatch.setattr(sys, "argv", ["prog", "betting", "log-bets", "--workbook", str(wb), "--db", str(db_path), "--writeback"])
        pl.main()
        # verify writeback
        df = pd.read_excel(wb, sheet_name="BETS")
        assert "bet_id" in df.columns
        assert not df.iloc[0]["bet_id"] is None
