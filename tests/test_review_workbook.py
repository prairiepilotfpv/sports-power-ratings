from pathlib import Path
import tempfile
from datetime import date
import sqlite3

import pandas as pd

from src.pipelines.review_runs import build_review_workbook, create_and_build_review
from src.data import betting_repository as br
from src.data import repository as repo


def test_build_review_workbook_creates_file_and_meta():
    td = Path(tempfile.mkdtemp())
    try:
        db_path = Path(td) / "test.db"
        # seed games and opportunities
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
        rid = br.create_review_run(db_path, sport="nba", season="2025-26", model="elo", notes="test")
        # insert an opportunity
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO opportunities (review_run_id, game_id, market_type, selection, line, odds, implied_prob, model_prob, edge, ev, source_market_snapshot_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
                (rid, "2025-11-10-lakers-clippers", "ML", "Los Angeles Lakers", 0.0, 110, 0.48, 0.55, 0.07, 0.08, None),
            )
            conn.commit()

        out = build_review_workbook(db_path, review_run_id=rid, sport="nba", season="2025-26")
        assert out.exists()
        # load metasheet
        meta = pd.read_excel(out, sheet_name="META")
        assert meta[meta.key == "review_run_id"].value.values[0] == rid
        # EV sheet has rows
        ev = pd.read_excel(out, sheet_name="EV")
        assert not ev.empty
        # BETS sheet exists and has stake column
        bets = pd.read_excel(out, sheet_name="BETS")
        assert "stake" in bets.columns
        # EV sheet is protected via openpyxl (load workbook to check)
        from openpyxl import load_workbook
        wb = load_workbook(out)
        try:
            assert wb["EV"].protection.sheet is True
            assert wb["BETS"].protection.sheet is False
            assert wb["META"].sheet_state == "hidden"
        finally:
            # close workbook to release file handles on Windows
            try:
                wb.close()
            except Exception:
                pass
        # best-effort: ensure any lingering DB handles are closed
        import gc

        gc.collect()
        try:
            sqlite3.connect(db_path).close()
        except Exception:
            pass
        # best-effort: attempt to delete DB file before temporary cleanup on Windows (handle locking)
        try:
            import os, time

            for _ in range(10):
                try:
                    os.unlink(db_path)
                    break
                except PermissionError:
                    time.sleep(0.1)
        except Exception:
            pass
    finally:
        # try best to remove temp dir but ignore permission errors on Windows
        import shutil, errno

        try:
            shutil.rmtree(td)
        except OSError as exc:
            if exc.errno != errno.EACCES:
                raise


def test_create_and_build_review_helper_returns_path():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        repo.init_db(db_path)
        br.init_db(db_path)
        out = create_and_build_review(db_path, sport="nba", season="2025-26", model="elo", notes="x")
        assert out.exists()
