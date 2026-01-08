from pathlib import Path
import sqlite3
import tempfile
from datetime import date

import pandas as pd

from src.pipelines.bets import log_bets, settle_bets
from src.data import repository as repo
from src.data import betting_repository as br


def build_workbook(path: Path, review_run_id: str):
    df_meta = pd.DataFrame([{"key": "review_run_id", "value": review_run_id}])
    df_bets = pd.DataFrame(
        [
            {
                "review_run_id": review_run_id,
                "game_id": "2025-11-10-lakers-clippers",
                "market_type": "ML",
                "selection": "Los Angeles Lakers",
                "line": 0.0,
                "odds": 110,
                "stake": 10.0,
                "book": "BookA",
                "price": None,
            },
            {
                "review_run_id": review_run_id,
                "game_id": "2025-11-10-lakers-clippers",
                "market_type": "spread",
                "selection": "Los Angeles Lakers",
                "line": -3.5,
                "odds": -110,
                "stake": 5.0,
                "book": "BookA",
                "price": None,
            },
            # PASS row
            {
                "review_run_id": review_run_id,
                "game_id": "2025-11-10-lakers-clippers",
                "market_type": "ML",
                "selection": "LA Clippers",
                "line": 0.0,
                "odds": 120,
                "stake": "",
                "book": "BookA",
                "price": None,
            },
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_bets.to_excel(writer, sheet_name="BETS", index=False)
        df_meta.to_excel(writer, sheet_name="META", index=False)


def test_log_bets_idempotent_and_writeback():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        repo.init_db(db_path)
        br.init_db(db_path)
        rid = br.create_review_run(db_path, sport="nba", season="2025-26", model="elo", notes="test")
        # seed game
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
        wb = Path(td) / "review.xlsx"
        build_workbook(wb, rid)
        # log bets with writeback
        inserted = log_bets(str(wb), review_run_id=rid, db_path=db_path, dry_run=False, writeback=True)
        assert inserted == 2
        # run again should be idempotent
        inserted2 = log_bets(str(wb), review_run_id=rid, db_path=db_path, dry_run=False, writeback=True)
        assert inserted2 == 2
        # workbook should have bet_id and logged_at in BETS sheet
        df = pd.read_excel(wb, sheet_name="BETS")
        assert "bet_id" in df.columns
        assert "logged_at" in df.columns
        assert not df[df.stake != ""].bet_id.isna().all()


def test_settle_bets_ml_and_spread_and_total():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        repo.init_db(db_path)
        br.init_db(db_path)
        # create a game with scores
        repo.save_games(
            db_path,
            [
                repo.GameResult(
                    date=date(2025, 11, 10),
                    home_team="Los Angeles Lakers",
                    away_team="LA Clippers",
                    home_score=110,
                    away_score=100,
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
        # insert bets
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "INSERT INTO bets (review_run_id, game_id, market_type, selection, line, odds, stake, book, logged_at, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), 'pending')",
                ("r1", "2025-11-10-lakers-clippers", "ML", "Los Angeles Lakers", 0.0, 110, 10.0, "BookA"),
            )
            conn.execute(
                "INSERT INTO bets (review_run_id, game_id, market_type, selection, line, odds, stake, book, logged_at, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), 'pending')",
                ("r1", "2025-11-10-lakers-clippers", "spread", "Los Angeles Lakers", -3.5, -110, 5.0, "BookA"),
            )
            conn.execute(
                "INSERT INTO bets (review_run_id, game_id, market_type, selection, line, odds, stake, book, logged_at, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), 'pending')",
                ("r1", "2025-11-10-lakers-clippers", "total", "Over", 205.0, -110, 20.0, "BookA"),
            )
            conn.commit()
        finally:
            conn.close()

        settled = settle_bets(sport="nba", season="2025-26", db_path=db_path)
        assert settled == 3
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute("SELECT outcome, profit FROM bets").fetchall()
            assert any(r[0] == "win" for r in rows)
            assert any(r[0] == "loss" or r[0] == "push" or r[0] == "win" for r in rows)
        finally:
            conn.close()
