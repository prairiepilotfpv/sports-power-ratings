from pathlib import Path
import tempfile
import sqlite3
from datetime import date

from src.data import repository as repo
from src.data import betting_repository as br
from src.data import reporting as rpt


def seed_simple_db(db_path: Path):
    repo.init_db(db_path)
    br.init_db(db_path)
    # create two games on two dates
    repo.save_games(
        db_path,
        [
            repo.GameResult(
                date=date(2025, 11, 10),
                home_team="H1",
                away_team="A1",
                home_score=100,
                away_score=90,
                neutral=False,
                overtime=False,
                decision_type=None,
                game_id="g1",
                sport="nba",
                season="2025-26",
                division=None,
                conference=None,
                notes=None,
            ),
            repo.GameResult(
                date=date(2025, 11, 11),
                home_team="H2",
                away_team="A2",
                home_score=110,
                away_score=100,
                neutral=False,
                overtime=False,
                decision_type=None,
                game_id="g2",
                sport="nba",
                season="2025-26",
                division=None,
                conference=None,
                notes=None,
            ),
        ],
    )
    # insert opportunities and bets
    conn = sqlite3.connect(db_path)
    try:
        # opportunity with edge 0.06 and ev 0.1
        conn.execute("INSERT INTO opportunities (review_run_id, game_id, market_type, selection, line, odds, implied_prob, model_prob, edge, ev, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))", ("r1", "g1", "ML", "H1", 0.0, 110, 0.48, 0.54, 0.06, 0.1))
        # matched bet settled
        conn.execute("INSERT INTO bets (review_run_id, game_id, market_type, selection, line, odds, stake, book, logged_at, status, source_opportunity_id, profit) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), 'settled', ?, ?)", ("r1", "g1", "ML", "H1", 0.0, 110, 10.0, "b", 1, 10* (110/100)))
        # pending bet with ev
        conn.execute("INSERT INTO opportunities (review_run_id, game_id, market_type, selection, line, odds, implied_prob, model_prob, edge, ev, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))", ("r1", "g2", "ML", "H2", 0.0, 120, 0.45, 0.50, 0.05, 0.08))
        conn.execute("INSERT INTO bets (review_run_id, game_id, market_type, selection, line, odds, stake, book, logged_at, status, source_opportunity_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), 'pending', ?)", ("r1", "g2", "ML", "H2", 0.0, 120, 5.0, "b", 2))
        conn.commit()
    finally:
        conn.close()


def test_daily_report_and_edge_buckets_and_clv():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        seed_simple_db(db_path)
        daily = rpt.daily_report(db_path, sport="nba", season="2025-26")
        assert len(daily) >= 2
        # find g1 row
        d1 = [r for r in daily if r["date"] == date(2025, 11, 10)][0]
        assert d1["total_bets"] >= 1
        assert d1["realized_pl"] > 0

        buckets = rpt.edge_bucket_report(db_path, sport="nba", season="2025-26")
        assert any(b["edge_bucket"] == ">5%" for b in buckets)

        clv = rpt.clv_summary(db_path, sport="nba", season="2025-26")
        assert clv["avg_clv_close_odds"] is None
