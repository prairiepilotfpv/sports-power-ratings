from pathlib import Path
import tempfile
from datetime import date
import sqlite3

import pandas as pd

from src.pipelines.review_runs import build_review_workbook, create_and_build_review
from src.pipelines import opportunities as opportunities_pipeline
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


def test_review_workbook_includes_ocr_raw_sheet_when_available():
    td = Path(tempfile.mkdtemp())
    try:
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
        rid = br.create_review_run(db_path, sport="nba", season="2025-26", model="elo", notes="test")
        staging_id = br.add_staging_row(
            db_path,
            source="screenshot",
            captured_at="2025-11-10T12:00:00Z",
            image_path="img1.png",
            raw_text="Lakers -110 vs Clippers",
            book="dn",
            market_type="ML",
            selection="Los Angeles Lakers",
            line=0.0,
            odds=-110,
            team_home_raw="Lakers",
            team_away_raw="Clippers",
            game_date="2025-11-10",
            match_status="matched",
            match_confidence=0.91,
            game_id="2025-11-10-lakers-clippers",
        )
        br.tag_staging_hold(db_path, staging_id=staging_id, reason="duplicate_in_image")

        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO market_snapshots (
                    snapshot_run_id, captured_at, book, market_type, selection, line, odds, game_id, source_staging_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    "snap1",
                    "2025-11-10T12:00:00Z",
                    "dn",
                    "ML",
                    "Los Angeles Lakers",
                    0.0,
                    -110,
                    "2025-11-10-lakers-clippers",
                    staging_id,
                ),
            )
            snapshot_id = cur.lastrowid
            conn.execute(
                """
                INSERT INTO opportunities (
                    review_run_id, game_id, market_type, selection, line, odds,
                    implied_prob, model_prob, edge, ev, source_market_snapshot_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (rid, "2025-11-10-lakers-clippers", "ML", "Los Angeles Lakers", 0.0, -110, 0.52, 0.58, 0.06, 0.09, snapshot_id),
            )
            conn.commit()

        out = build_review_workbook(db_path, review_run_id=rid, sport="nba", season="2025-26")
        ocr = pd.read_excel(out, sheet_name="OCR_RAW")
        expected_cols = [
            "source_market_snapshot_id",
            "snapshot_run_id",
            "game_id",
            "source_staging_id",
            "staging_source",
            "image_path",
            "raw_text",
            "team_home_raw",
            "team_away_raw",
            "game_date",
            "match_status",
            "match_confidence",
            "hold_reason",
            "captured_at",
            "book",
            "market_type",
            "selection",
            "line",
            "odds",
        ]
        for col in expected_cols:
            assert col in ocr.columns
        assert not ocr.empty
        assert int(ocr["source_market_snapshot_id"].iloc[0]) == snapshot_id
        assert ocr["snapshot_run_id"].iloc[0] == "snap1"
        assert ocr["game_id"].iloc[0] == "2025-11-10-lakers-clippers"
        assert int(ocr["source_staging_id"].iloc[0]) == staging_id
        assert ocr["staging_source"].iloc[0] == "screenshot"
        assert ocr["image_path"].iloc[0] == "img1.png"
        assert ocr["game_date"].iloc[0] == "2025-11-10"
    finally:
        import shutil, errno

        try:
            shutil.rmtree(td)
        except OSError as exc:
            if exc.errno != errno.EACCES:
                raise


def test_review_workbook_includes_exclusions_sheet_when_guardrails_filter_predictions(monkeypatch):
    td = Path(tempfile.mkdtemp())
    try:
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
        rid = br.create_review_run(db_path, sport="nba", season="2025-26", model="elo", notes="test")
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO market_snapshots (
                    snapshot_run_id, captured_at, book, market_type, selection, line, odds, game_id, source_staging_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    "snap1",
                    "2025-11-10T12:00:00Z",
                    "dn",
                    "ML",
                    "Los Angeles Lakers",
                    0.0,
                    -110,
                    "2025-11-10-lakers-clippers",
                    None,
                ),
            )
            conn.commit()

        def _fake_predictions(*args, **kwargs):
            return pd.DataFrame(
                [
                    {
                        "game_id": "2025-11-10-lakers-clippers",
                        "model": "elo",
                        "sport": "nba",
                        "margin_mean": 4.0,
                        "margin_sd": 1.0,
                        "projected_home_score": 110.0,
                        "projected_away_score": 100.0,
                        "model_p_home_win": 0.6,
                        "margin_dist_assumption": "normal_approx",
                    }
                ]
            )

        monkeypatch.setattr(opportunities_pipeline, "load_schedule_predictions", _fake_predictions)
        opportunities_pipeline.build_opportunities(
            db_path,
            review_run_id=rid,
            sport="nba",
            season="2025-26",
            model="elo",
            snapshot_run_id="snap1",
        )

        out = build_review_workbook(db_path, review_run_id=rid, sport="nba", season="2025-26")
        exclusions = pd.read_excel(out, sheet_name="EXCLUSIONS")
        assert not exclusions.empty
        assert "game_id" in exclusions.columns
        assert "model" in exclusions.columns
        assert "excluded_reason" in exclusions.columns
    finally:
        import shutil, errno

        try:
            shutil.rmtree(td)
        except OSError as exc:
            if exc.errno != errno.EACCES:
                raise
