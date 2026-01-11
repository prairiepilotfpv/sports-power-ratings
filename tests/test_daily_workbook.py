from pathlib import Path
import tempfile
from datetime import date
import sqlite3

import pandas as pd

from src.pipelines.daily_workbook import build_daily_workbook
from src.data import betting_repository as br
from src.data import repository as repo


def test_daily_workbook_creates_sheets_and_rows():
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
        rid = br.create_review_run(
            db_path,
            sport="nba",
            season="2025-26",
            model="elo",
            notes="daily",
            created_at="2025-11-10T08:00:00Z",
        )

        staging_id = br.add_staging_row(
            db_path,
            source="screenshot",
            captured_at="2025-11-10T09:00:00Z",
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

        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO market_snapshots (
                    snapshot_run_id, captured_at, book, market_type, selection, line,
                    odds, game_id, source_staging_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    "snap1",
                    "2025-11-10T09:00:00Z",
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
                (
                    rid,
                    "2025-11-10-lakers-clippers",
                    "ML",
                    "Los Angeles Lakers",
                    0.0,
                    -110,
                    0.52,
                    0.58,
                    0.06,
                    0.09,
                    snapshot_id,
                ),
            )
            conn.commit()

        out = build_daily_workbook(
            db_path,
            sport="nba",
            season="2025-26",
            date="2025-11-10",
        )
        assert out.exists()

        projections = pd.read_excel(out, sheet_name="PROJECTIONS")
        market = pd.read_excel(out, sheet_name="MARKET_SNAPSHOTS")
        ocr = pd.read_excel(out, sheet_name="OCR_RAW")
        ev = pd.read_excel(out, sheet_name="EV")
        bets = pd.read_excel(out, sheet_name="BETS")

        assert not projections.empty
        assert not market.empty
        assert not ocr.empty
        assert not ev.empty
        assert not bets.empty
    finally:
        import shutil, errno

        try:
            shutil.rmtree(td)
        except OSError as exc:
            if exc.errno != errno.EACCES:
                raise
