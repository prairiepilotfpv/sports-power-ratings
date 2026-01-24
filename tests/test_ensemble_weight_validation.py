"""Test ensemble weight validation with database backend."""

import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import pytest

from pipelines.ensemble_weight_validation import validate_ensemble_ml_weights
from data.repository import init_db
from data.bets_repository import save_bets_predictions


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    with NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    # Initialize schema
    init_db(db_path)
    
    # Create and populate test data
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Insert test games with scores (completed)
    test_games = [
        ("2026-01-23|Team A|Team B", "2026-01-23", "Team A", "Team B", 110, 100, "nba", "2025-26"),
        ("2026-01-22|Team C|Team D", "2026-01-22", "Team C", "Team D", 95, 100, "nba", "2025-26"),
        ("2026-01-21|Team E|Team F", "2026-01-21", "Team E", "Team F", 85, 90, "nba", "2025-26"),
    ]
    
    for game_id, date, home, away, hscore, ascore, sport, season in test_games:
        cursor.execute(
            """
            INSERT INTO games
            (date, home_team, away_team, home_score, away_score, game_id, sport, season)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (date, home, away, hscore, ascore, game_id, sport, season),
        )
    
    conn.commit()
    conn.close()
    
    yield db_path
    
    # Cleanup
    Path(db_path).unlink()


def test_validate_with_database(temp_db):
    """Test validation using database backend."""
    # Save BETS predictions
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    day_before = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
    
    # Predictions from two days
    bets_df_yesterday = pd.DataFrame({
        "game_id": ["2026-01-23|Team A|Team B", "2026-01-22|Team C|Team D"],
        "home_win_prob": [0.65, 0.55],
    })
    
    bets_df_day_before = pd.DataFrame({
        "game_id": ["2026-01-21|Team E|Team F"],
        "home_win_prob": [0.45],
    })
    
    # Save predictions
    save_bets_predictions(
        temp_db,
        bets_df=bets_df_yesterday,
        sport="nba",
        season="2025-26",
        prediction_date=yesterday,
    )
    
    save_bets_predictions(
        temp_db,
        bets_df=bets_df_day_before,
        sport="nba",
        season="2025-26",
        prediction_date=day_before,
    )
    
    # Run validation (7-day window from yesterday)
    result = validate_ensemble_ml_weights(
        db_path=temp_db,
        sport="nba",
        season="2025-26",
        market="ML",
        days_back=7,
    )
    
    assert result is not None
    assert result["n_games"] == 3  # All three games
    assert result["n_prediction_dates"] >= 1
    assert result["tuned_log_loss"] is not None
    assert result["tuned_brier_score"] is not None
    assert "by_date" in result
    assert len(result["by_date"]) >= 1


def test_validate_no_predictions(temp_db):
    """Test validation when no predictions are stored."""
    result = validate_ensemble_ml_weights(
        db_path=temp_db,
        sport="nba",
        season="2025-26",
        market="ML",
    )
    
    assert result is None


def test_validate_non_ml_market(temp_db):
    """Test validation rejects non-ML markets."""
    result = validate_ensemble_ml_weights(
        db_path=temp_db,
        sport="nba",
        season="2025-26",
        market="SPREAD",  # Only ML supported
    )
    
    assert result is None


def test_validate_rolling_window(temp_db):
    """Test that validation respects rolling window (days_back)."""
    # Add predictions for 3 different dates
    base_date = datetime.now(timezone.utc) - timedelta(days=1)
    
    for i in range(3):
        pred_date = (base_date - timedelta(days=i)).strftime("%Y-%m-%d")
        game_date = (datetime.fromisoformat(pred_date).date() + timedelta(days=1)).strftime("%Y-%m-%d")
        game_id = f"2026-01-{25-i}|Team X{i}|Team Y{i}"
        
        # Create a game for this date
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO games
            (date, home_team, away_team, home_score, away_score, game_id, sport, season)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (game_date, f"Team X{i}", f"Team Y{i}", 100 + i*10, 90 + i*10, game_id, "nba", "2025-26"),
        )
        conn.commit()
        conn.close()
        
        # Save prediction
        bets_df = pd.DataFrame({
            "game_id": [game_id],
            "home_win_prob": [0.55],
        })
        
        save_bets_predictions(
            temp_db,
            bets_df=bets_df,
            sport="nba",
            season="2025-26",
            prediction_date=pred_date,
        )
    
    # Validate with days_back=2 (should get 2 days)
    result = validate_ensemble_ml_weights(
        db_path=temp_db,
        sport="nba",
        season="2025-26",
        market="ML",
        days_back=2,
    )
    
    assert result is not None
    assert result["n_prediction_dates"] <= 2  # At most 2 prediction dates
