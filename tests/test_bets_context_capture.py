"""Tests for bet context capture: prediction fields, odds, probabilities, ensemble details."""

from pathlib import Path
import sqlite3
import tempfile
from datetime import date

import pandas as pd

from src.pipelines.bets import log_bets
from src.data import repository as repo
from src.data import betting_repository as br


def build_workbook_with_context(path: Path, review_run_id: str):
    """Build a BETS workbook with full prediction context columns."""
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
                # Prediction context
                "home_win_prob": 0.65,
                "away_win_prob": 0.35,
                "model_prob": 0.62,
                "edge": 0.03,  # 3% edge
                "ev": 0.30,  # $0.30 expected value
                "market_forecast_source": "elo_ml_ensemble",
                "ml_ensemble_components_json": '{"elo": 0.6, "bradley_terry": 0.4}',
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
                # Prediction context for spread
                "margin_mean": -3.0,
                "margin_sd": 1.2,
                "model_prob": 0.58,
                "edge": 0.08,
                "ev": 0.40,
                "market_forecast_source": "spread_ensemble",
                "spread_ensemble_components_json": '{"gaussian": 0.7, "bradley_terry": 0.3}',
            },
            {
                "review_run_id": review_run_id,
                "game_id": "2025-11-10-lakers-clippers",
                "market_type": "total",
                "selection": "over",
                "line": 225.5,
                "odds": -110,
                "stake": 3.0,
                "book": "BookA",
                "price": None,
                # Prediction context for total
                "total": 224.0,
                "total_sd": 8.5,
                "model_prob": 0.55,
                "edge": 0.05,
                "ev": 0.15,
                "market_forecast_source": "total_ensemble",
                "total_ensemble_components_json": '{"poisson": 1.0}',
            },
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_bets.to_excel(writer, sheet_name="BETS", index=False)
        df_meta.to_excel(writer, sheet_name="META", index=False)


def test_log_bets_captures_prediction_context():
    """Verify that log_bets extracts and stores all prediction context fields."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        repo.init_db(db_path)
        br.init_db(db_path)
        rid = br.create_review_run(db_path, sport="nba", season="2025-26", model="elo", notes="test")
        
        # Seed game
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
        
        wb = Path(td) / "bets_with_context.xlsx"
        build_workbook_with_context(wb, rid)
        
        # Log bets with context
        inserted = log_bets(str(wb), review_run_id=rid, db_path=db_path, dry_run=False, writeback=False)
        assert inserted == 3, f"Expected 3 bets inserted, got {inserted}"
        
        # Query the DB and verify context fields were captured
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        # Check ML bet
        ml_row = cur.execute(
            "SELECT * FROM bets WHERE game_id = ? AND market_type = ? AND selection = ?",
            ("2025-11-10-lakers-clippers", "ML", "Los Angeles Lakers"),
        ).fetchone()
        assert ml_row is not None
        assert ml_row["home_win_prob"] == 0.65
        assert ml_row["away_win_prob"] == 0.35
        assert ml_row["model_prob"] == 0.62
        assert ml_row["edge"] == 0.03
        assert ml_row["ev"] == 0.30
        assert ml_row["market_forecast_source"] == "elo_ml_ensemble"
        assert "elo" in ml_row["ensemble_components_json"]
        
        # Check spread bet
        spread_row = cur.execute(
            "SELECT * FROM bets WHERE game_id = ? AND market_type = ? AND selection = ?",
            ("2025-11-10-lakers-clippers", "spread", "Los Angeles Lakers"),
        ).fetchone()
        assert spread_row is not None
        assert spread_row["margin_mean"] == -3.0
        assert spread_row["margin_sd"] == 1.2
        assert spread_row["edge"] == 0.08
        
        # Check total bet
        total_row = cur.execute(
            "SELECT * FROM bets WHERE game_id = ? AND market_type = ? AND selection = ?",
            ("2025-11-10-lakers-clippers", "total", "over"),
        ).fetchone()
        assert total_row is not None
        assert total_row["total"] == 224.0
        assert total_row["total_sd"] == 8.5
        
        conn.close()


def test_import_bets_csv_with_history_export():
    """Verify that import_bets_csv parses and imports bets with outcomes from history export."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        repo.init_db(db_path)
        br.init_db(db_path)

        # Seed games
        repo.save_games(
            db_path,
            [
                repo.GameResult(
                    date=date(2026, 1, 8),
                    home_team="Philadelphia 76ers",
                    away_team="Washington Wizards",
                    home_score=101,
                    away_score=98,
                    neutral=False,
                    overtime=False,
                    decision_type=None,
                    game_id="test-game-1",
                    sport="nba",
                    season="2025-26",
                    division=None,
                    conference=None,
                    notes=None,
                ),
                repo.GameResult(
                    date=date(2026, 1, 8),
                    home_team="Charlotte Hornets",
                    away_team="Toronto Raptors",
                    home_score=108,
                    away_score=112,
                    neutral=False,
                    overtime=False,
                    decision_type=None,
                    game_id="test-game-2",
                    sport="nba",
                    season="2025-26",
                    division=None,
                    conference=None,
                    notes=None,
                ),
            ],
        )

        # Create a CSV with explicit game_ids (simpler format for testing)
        csv_path = Path(td) / "history_sample.csv"
        csv_data = """league,start_time,game,game_id,pick_desc,type,period,odds,odds_spread_total,result,units_wagered,units_net,money_wagered,money_net,tag
nba,2026-01-08T00:00:00.000Z,Washington @ Philadelphia,test-game-1,WAS +12.5 -110,spread_away,game,-110,12.5,loss,1.1,-1.1,1.1,-1.1,
nba,2026-01-08T00:00:00.000Z,Toronto @ Charlotte,test-game-2,TOR -135,ml_away,game,-135,-135,win,1.35,1,1.35,1,"""
        csv_path.write_text(csv_data)

        # Import the CSV
        result = br.import_bets_csv(
            db_path,
            csv_path=csv_path,
            sport="nba",
            season="2025-26",
            dry_run=False,
        )
        
        assert result["inserted"] == 2, f"Expected 2 inserted, got {result['inserted']}. Errors: {result.get('errors')}"
        assert result["updated"] == 0
        assert result["rejected"] == 0
        assert result["skipped"] == 0
        
        # Verify the bets were inserted with outcomes
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        # Check spread bet (should be a loss)
        spread_bet = cur.execute(
            "SELECT * FROM bets WHERE market_type = ? AND selection = ?",
            ("spread", "away"),
        ).fetchone()
        assert spread_bet is not None
        assert spread_bet["outcome"] == "loss"
        assert spread_bet["profit"] == -1.1
        assert spread_bet["stake"] == 1.1
        
        # Check ML bet (should be a win)
        ml_bet = cur.execute(
            "SELECT * FROM bets WHERE market_type = ? AND selection = ?",
            ("ML", "away"),
        ).fetchone()
        assert ml_bet is not None
        assert ml_bet["outcome"] == "win"
        assert ml_bet["stake"] == 1.35
        
        conn.close()


def test_import_bets_csv_idempotent():
    """Verify that importing the same CSV twice is idempotent."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        repo.init_db(db_path)
        br.init_db(db_path)
        
        # Seed a game
        repo.save_games(
            db_path,
            [
                repo.GameResult(
                    date=date(2026, 1, 8),
                    home_team="Philadelphia 76ers",
                    away_team="Washington Wizards",
                    home_score=101,
                    away_score=98,
                    neutral=False,
                    overtime=False,
                    decision_type=None,
                    game_id="test-game-1",
                    sport="nba",
                    season="2025-26",
                    division=None,
                    conference=None,
                    notes=None,
                ),
            ],
        )

        csv_path = Path(td) / "history.csv"
        csv_data = """league,start_time,game,game_id,pick_desc,type,period,odds,odds_spread_total,result,units_wagered,units_net,money_wagered,money_net,tag
nba,2026-01-08T00:00:00.000Z,WAS @ PHI,test-game-1,WAS +12.5 -110,spread_away,game,-110,12.5,loss,1.1,-1.1,1.1,-1.1,"""
        csv_path.write_text(csv_data)

        # First import
        result1 = br.import_bets_csv(
            db_path,
            csv_path=csv_path,
            sport="nba",
            season="2025-26",
            dry_run=False,
        )
        assert result1["inserted"] == 1
        assert result1["updated"] == 0
        
        # Second import (same CSV)
        result2 = br.import_bets_csv(
            db_path,
            csv_path=csv_path,
            sport="nba",
            season="2025-26",
            dry_run=False,
        )
        
        # Verify only one bet exists in the DB
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0]
        assert count == 1
        conn.close()


def test_import_bets_csv_dry_run():
    """Verify that dry-run mode doesn't write to the database."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        repo.init_db(db_path)
        br.init_db(db_path)
        
        # Seed a game
        repo.save_games(
            db_path,
            [
                repo.GameResult(
                    date=date(2026, 1, 8),
                    home_team="Philadelphia 76ers",
                    away_team="Washington Wizards",
                    home_score=None,
                    away_score=None,
                    neutral=False,
                    overtime=False,
                    decision_type=None,
                    game_id="test-game-1",
                    sport="nba",
                    season="2025-26",
                    division=None,
                    conference=None,
                    notes=None,
                ),
            ],
        )

        csv_path = Path(td) / "history.csv"
        csv_data = """league,start_time,game,game_id,pick_desc,type,period,odds,odds_spread_total,result,units_wagered,units_net,money_wagered,money_net,tag
nba,2026-01-08T00:00:00.000Z,WAS @ PHI,test-game-1,WAS +12.5 -110,spread_away,game,-110,12.5,loss,1.1,-1.1,1.1,-1.1,"""
