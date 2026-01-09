import tempfile
from pathlib import Path
from datetime import date
import sqlite3

import pandas as pd

from src.data import repository as repo
from src.data import betting_repository as br
from src.pipelines.bets import log_bets
from src.data.reporting import write_full_report_xlsx, daily_report


def build_workbook_for_clv(path: Path, review_run_id: str):
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
            }
        ]
    )
    df_meta = pd.DataFrame([{"key": "review_run_id", "value": review_run_id}])
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_bets.to_excel(writer, sheet_name="BETS", index=False)
        df_meta.to_excel(writer, sheet_name="META", index=False)


def test_clv_attached_to_bet():
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
                )
            ],
        )
        # add CLV snapshot
        br.add_clv_snapshot(db_path, game_id="2025-11-10-lakers-clippers", market_type="ML", selection="Los Angeles Lakers", close_line=0.0, close_odds=115)

        wb = Path(td) / "review.xlsx"
        build_workbook_for_clv(wb, rid)
        inserted = log_bets(str(wb), review_run_id=rid, db_path=db_path, dry_run=False, writeback=True)
        assert inserted == 1

        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute("SELECT clv_close_odds, clv_close_line FROM bets").fetchone()
            assert row is not None
            assert row[0] == 115
            assert abs(row[1] - 0.0) < 1e-9
        finally:
            conn.close()


def test_write_full_report_pnl_sheet():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        out = Path(td) / "report.xlsx"
        repo.init_db(db_path)
        br.init_db(db_path)
        # seed game and opportunity
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
                )
            ],
        )
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "INSERT INTO opportunities (review_run_id, game_id, market_type, selection, line, odds, implied_prob, model_prob, edge, ev, source_market_snapshot_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
                ("r1", "2025-11-10-lakers-clippers", "ML", "Los Angeles Lakers", 0.0, 110, 0.476, 0.6, 0.12, 1.2, None),
            )
            op_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            # insert a bet referencing opportunity
            conn.execute(
                "INSERT INTO bets (review_run_id, game_id, market_type, selection, line, odds, stake, book, logged_at, status, source_opportunity_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?, ?)",
                ("r1", "2025-11-10-lakers-clippers", "ML", "Los Angeles Lakers", 0.0, 110, 10.0, "BookA", "pending", op_id),
            )
            conn.commit()
        finally:
            conn.close()

        rows = daily_report(db_path, sport="nba", season="2025-26")
        write_full_report_xlsx(db_path, sport="nba", season="2025-26", rpt_type="daily", rows=rows, output_path=out)
        with pd.ExcelFile(out) as wb:
            assert "PnL_Scenarios" in wb.sheet_names
        df = pd.read_excel(out, sheet_name="PnL_Scenarios")
        assert "kelly_fraction" in df.columns
        assert df["kelly_fraction"].notna().any()
