from pathlib import Path
import sqlite3
import tempfile
from datetime import date

from src.pipelines.bets import settle_bets
from src.data import repository as repo
from src.data import betting_repository as br


def test_spread_push_and_away_push():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        repo.init_db(db_path)
        br.init_db(db_path)
        # home wins by 3, line -3 -> push for both home and away selections
        repo.save_games(
            db_path,
            [
                repo.GameResult(
                    date=date(2025, 11, 10),
                    home_team="A",
                    away_team="B",
                    home_score=100,
                    away_score=97,
                    neutral=False,
                    overtime=False,
                    decision_type=None,
                    game_id="g1",
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
            conn.execute("INSERT INTO bets (review_run_id, game_id, market_type, selection, line, odds, stake, book, logged_at, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), 'pending')", ("r", "g1", "spread", "A", -3.0, -110, 10.0, "b"))
            conn.execute("INSERT INTO bets (review_run_id, game_id, market_type, selection, line, odds, stake, book, logged_at, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), 'pending')", ("r", "g1", "spread", "B", -3.0, -110, 10.0, "b"))
            conn.commit()
        finally:
            conn.close()

        settled = settle_bets(sport="nba", season="2025-26", db_path=db_path)
        assert settled == 2
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute("SELECT selection, outcome, profit FROM bets ORDER BY selection").fetchall()
            # Both should be push
            assert rows[0][1] == "push" and rows[1][1] == "push"
            assert rows[0][2] == 0.0 and rows[1][2] == 0.0
        finally:
            conn.close()


def test_total_push_and_win_loss():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        repo.init_db(db_path)
        br.init_db(db_path)
        # total = 210, over selection with total 210 -> push; over when total>line win
        repo.save_games(
            db_path,
            [
                repo.GameResult(
                    date=date(2025, 11, 11),
                    home_team="X",
                    away_team="Y",
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
                repo.GameResult(
                    date=date(2025, 11, 12),
                    home_team="P",
                    away_team="Q",
                    home_score=111,
                    away_score=100,
                    neutral=False,
                    overtime=False,
                    decision_type=None,
                    game_id="g3",
                    sport="nba",
                    season="2025-26",
                    division=None,
                    conference=None,
                    notes=None,
                ),
                repo.GameResult(
                    date=date(2025, 11, 13),
                    home_team="M",
                    away_team="N",
                    home_score=100,
                    away_score=100,
                    neutral=False,
                    overtime=False,
                    decision_type=None,
                    game_id="g4",
                    sport="nba",
                    season="2025-26",
                    division=None,
                    conference=None,
                    notes=None,
                ),
            ],
        )
        conn = sqlite3.connect(db_path)
        try:
            # g2 total 210 -> push for Over 210
            conn.execute("INSERT INTO bets (review_run_id, game_id, market_type, selection, line, odds, stake, book, logged_at, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), 'pending')", ("r", "g2", "total", "Over", 210.0, -110, 10.0, "b"))
            # g3 total 211 -> Over 210 -> win
            conn.execute("INSERT INTO bets (review_run_id, game_id, market_type, selection, line, odds, stake, book, logged_at, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), 'pending')", ("r", "g3", "total", "Over", 210.0, -110, 10.0, "b"))
            # g4 total 200 -> Over 210 -> loss
            conn.execute("INSERT INTO bets (review_run_id, game_id, market_type, selection, line, odds, stake, book, logged_at, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), 'pending')", ("r", "g4", "total", "Over", 210.0, -110, 10.0, "b"))
            conn.commit()
        finally:
            conn.close()

        settled = settle_bets(sport="nba", season="2025-26", db_path=db_path)
        assert settled == 3
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute("SELECT game_id, outcome, profit FROM bets ORDER BY game_id").fetchall()
            # g2 push, g3 win, g4 loss
            d = {r[0]: (r[1], r[2]) for r in rows}
            assert d['g2'][0] == 'push'
            assert d['g3'][0] == 'win' and d['g3'][1] > 0
            assert d['g4'][0] == 'loss' and d['g4'][1] < 0
        finally:
            conn.close()


def test_settle_idempotent():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        repo.init_db(db_path)
        br.init_db(db_path)
        repo.save_games(
            db_path,
            [
                repo.GameResult(
                    date=date(2025, 11, 14),
                    home_team="H",
                    away_team="A",
                    home_score=110,
                    away_score=100,
                    neutral=False,
                    overtime=False,
                    decision_type=None,
                    game_id="g5",
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
            conn.execute("INSERT INTO bets (review_run_id, game_id, market_type, selection, line, odds, stake, book, logged_at, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), 'pending')", ("r", "g5", "ML", "H", 0.0, 110, 10.0, "b"))
            conn.commit()
        finally:
            conn.close()

        first = settle_bets(sport="nba", season="2025-26", db_path=db_path)
        second = settle_bets(sport="nba", season="2025-26", db_path=db_path)
        assert first == 1
        assert second == 0
