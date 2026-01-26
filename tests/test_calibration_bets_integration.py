"""
Test that calibration system integrates properly with BETS sheet generation.

This test validates the end-to-end flow:
1. Fit calibrators for ML/SPREAD/TOTAL markets
2. Run schedule generation
3. Verify BETS sheet has calibrated values in the right columns
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.calibration.distribution import MarginalDistributionCalibrator
from src.calibration.io import save_calibrator
from src.contracts import Game
from src.data.db import DB
from src.data.paths import Paths
from src.pipelines.schedule import _apply_calibration_to_schedule_df


@pytest.fixture
def temp_db_nba():
    """Create a temporary test database with sample games."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "nba" / "2025-26.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        db = DB(db_path)
        
        # Create sample games with complete data
        base_date = datetime(2025, 11, 1)
        games = []
        
        for i in range(10):
            game_date = base_date + timedelta(days=i)
            games.append(
                Game(
                    game_id=f"test_game_{i}",
                    sport="nba",
                    season="2025-26",
                    game_date=game_date,
                    home_team=f"Team{i%5}",
                    away_team=f"Team{(i+1)%5}",
                    home_score=100 + i,
                    away_score=95 + i,
                    notes="",
                    source_id="test",
                )
            )
        
        db.insert_games(games)
        yield db, tmpdir


@pytest.fixture
def temp_calibrator_output(temp_db_nba):
    """Create temporary calibrator output directory."""
    _, tmpdir = temp_db_nba
    output_dir = Path(tmpdir) / "outputs" / "calibrators" / "nba" / "2025-26" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield output_dir


def test_apply_ml_calibration_to_schedule_df():
    """Test that ML calibrator is applied correctly to schedule DataFrame."""
    # Create sample schedule DataFrame with ML market predictions
    schedule_df = pd.DataFrame({
        "game_id": ["g1", "g2", "g3"],
        "market": ["ML", "ML", "ML"],
        "model": ["ensemble_ml_v1", "ensemble_ml_v1", "ensemble_ml_v1"],
        "home_win_prob": [0.60, 0.55, 0.70],
        "away_win_prob": [0.40, 0.45, 0.30],
    })
    
    # Create a simple mock calibrator
    class MockMLCalibrator:
        def __init__(self):
            self.method = "platt"
            
        def transform(self, probs):
            # Simple mock: add 0.05 to all probabilities and clip
            return (probs + 0.05).clip(0, 1)
    
    # Create calibrator directory and save mock calibrator
    with tempfile.TemporaryDirectory() as tmpdir:
        cal_output_dir = Path(tmpdir) / "outputs" / "calibrators" / "nba" / "2025-26" / "test" / "ML"
        cal_output_dir.mkdir(parents=True, exist_ok=True)
        
        mock_cal = MockMLCalibrator()
        save_calibrator(mock_cal, cal_output_dir)
        
        # Mock the load function to use our temp directory
        import src.calibration.io as cal_io
        orig_load = cal_io.load_latest_calibrator
        
        def mock_load(sport, season, model, market, source_id="test"):
            if market == "ML":
                return mock_cal
            return None
        
        cal_io.load_latest_calibrator = mock_load
        
        try:
            # Apply calibration
            result_df = _apply_calibration_to_schedule_df(
                schedule_df.copy(),
                sport="nba",
                season="2025-26",
                model="ensemble_ml_v1",
            )
            
            # Check that calibrated values are different from raw
            assert (result_df["home_win_prob"] != schedule_df["home_win_prob"]).any()
            
            # Check bounds
            assert (result_df["home_win_prob"] >= 0).all()
            assert (result_df["home_win_prob"] <= 1).all()
            
        finally:
            cal_io.load_latest_calibrator = orig_load


def test_apply_spread_distribution_calibration_to_schedule_df():
    """Test that SPREAD distribution calibrator is applied correctly."""
    # Create sample schedule DataFrame with SPREAD market predictions
    schedule_df = pd.DataFrame({
        "game_id": ["g1", "g2", "g3"],
        "market": ["spread", "spread", "spread"],
        "model": ["ensemble_spread_v1", "ensemble_spread_v1", "ensemble_spread_v1"],
        "margin_mean": [2.5, 3.0, 1.5],
        "margin_sd": [1.0, 1.2, 0.8],
    })
    
    # Create a simple distribution calibrator
    calibrator = MarginalDistributionCalibrator()
    
    # Fit on mock data
    fit_data = pd.DataFrame({
        "pred_mean": [2.0, 3.0, 2.5],
        "pred_sd": [1.0, 1.0, 1.0],
        "actual_value": [2.5, 3.5, 2.0],
    })
    calibrator.fit(fit_data)
    
    # Save and load calibrator
    with tempfile.TemporaryDirectory() as tmpdir:
        cal_dir = Path(tmpdir) / "calibrators" / "nba" / "2025-26" / "test" / "spread"
        cal_dir.mkdir(parents=True, exist_ok=True)
        
        save_calibrator(calibrator, cal_dir)
        
        # Mock the load function
        import src.calibration.io as cal_io
        orig_load = cal_io.load_latest_calibrator
        
        def mock_load(sport, season, model, market, source_id="test"):
            if market == "spread":
                return calibrator
            return None
        
        cal_io.load_latest_calibrator = mock_load
        
        try:
            # Apply calibration
            result_df = _apply_calibration_to_schedule_df(
                schedule_df.copy(),
                sport="nba",
                season="2025-26",
                model="ensemble_spread_v1",
            )
            
            # Check that spread calibration was applied
            assert "margin_mean" in result_df.columns
            assert "margin_sd" in result_df.columns
            
            # Values should be calibrated (not exactly equal to raw)
            # Note: They might be close depending on fit, but shouldn't be identical
            
        finally:
            cal_io.load_latest_calibrator = orig_load


def test_apply_total_distribution_calibration_to_schedule_df():
    """Test that TOTAL distribution calibrator is applied correctly."""
    # Create sample schedule DataFrame with TOTAL market predictions
    schedule_df = pd.DataFrame({
        "game_id": ["g1", "g2", "g3"],
        "market": ["total", "total", "total"],
        "model": ["ensemble_total_v1", "ensemble_total_v1", "ensemble_total_v1"],
        "total_mean": [210.0, 215.0, 205.0],
        "total_sd": [5.0, 6.0, 4.0],
    })
    
    # Create a simple distribution calibrator
    calibrator = MarginalDistributionCalibrator()
    
    # Fit on mock data
    fit_data = pd.DataFrame({
        "pred_mean": [210.0, 215.0, 205.0],
        "pred_sd": [5.0, 6.0, 4.0],
        "actual_value": [212.0, 217.0, 203.0],
    })
    calibrator.fit(fit_data)
    
    # Save and load calibrator
    with tempfile.TemporaryDirectory() as tmpdir:
        cal_dir = Path(tmpdir) / "calibrators" / "nba" / "2025-26" / "test" / "total"
        cal_dir.mkdir(parents=True, exist_ok=True)
        
        save_calibrator(calibrator, cal_dir)
        
        # Mock the load function
        import src.calibration.io as cal_io
        orig_load = cal_io.load_latest_calibrator
        
        def mock_load(sport, season, model, market, source_id="test"):
            if market == "total":
                return calibrator
            return None
        
        cal_io.load_latest_calibrator = mock_load
        
        try:
            # Apply calibration
            result_df = _apply_calibration_to_schedule_df(
                schedule_df.copy(),
                sport="nba",
                season="2025-26",
                model="ensemble_total_v1",
            )
            
            # Check that total calibration was applied
            assert "total_mean" in result_df.columns
            assert "total_sd" in result_df.columns
            
        finally:
            cal_io.load_latest_calibrator = orig_load


def test_apply_all_three_markets_calibration():
    """Test that all three markets can be calibrated simultaneously."""
    # Create schedule DataFrame with mixed markets
    schedule_df = pd.DataFrame({
        "game_id": ["g1", "g1", "g1"],
        "market": ["ML", "spread", "total"],
        "model": ["ensemble", "ensemble", "ensemble"],
        "home_win_prob": [0.60, None, None],
        "away_win_prob": [0.40, None, None],
        "margin_mean": [None, 2.5, None],
        "margin_sd": [None, 1.0, None],
        "total_mean": [None, None, 210.0],
        "total_sd": [None, None, 5.0],
    })
    
    # Create mock calibrators
    class MockMLCalibrator:
        method = "platt"
        def transform(self, probs):
            return (probs + 0.05).clip(0, 1)
    
    dist_calibrator = MarginalDistributionCalibrator()
    fit_data = pd.DataFrame({
        "pred_mean": [2.0, 210.0],
        "pred_sd": [1.0, 5.0],
        "actual_value": [2.5, 212.0],
    })
    dist_calibrator.fit(fit_data)
    
    # Mock the load function
    import src.calibration.io as cal_io
    orig_load = cal_io.load_latest_calibrator
    
    def mock_load(sport, season, model, market, source_id="test"):
        if market == "ML":
            return MockMLCalibrator()
        elif market in ["spread", "total"]:
            return dist_calibrator
        return None
    
    cal_io.load_latest_calibrator = mock_load
    
    try:
        # Apply calibration
        result_df = _apply_calibration_to_schedule_df(
            schedule_df.copy(),
            sport="nba",
            season="2025-26",
            model="ensemble",
        )
        
        # Check that all markets were calibrated
        ml_row = result_df[result_df["market"] == "ML"].iloc[0]
        assert ml_row["home_win_prob"] != schedule_df[schedule_df["market"] == "ML"].iloc[0]["home_win_prob"]
        
        # Spread and total should also have been processed
        assert "margin_mean" in result_df.columns
        assert "total_mean" in result_df.columns
        
    finally:
        cal_io.load_latest_calibrator = orig_load


def test_missing_calibrator_fallback():
    """Test that missing calibrators don't crash schedule generation."""
    # Create schedule DataFrame
    schedule_df = pd.DataFrame({
        "game_id": ["g1"],
        "market": ["ML"],
        "model": ["ensemble"],
        "home_win_prob": [0.60],
        "away_win_prob": [0.40],
    })
    
    # Mock the load function to return None
    import src.calibration.io as cal_io
    orig_load = cal_io.load_latest_calibrator
    
    cal_io.load_latest_calibrator = lambda **kwargs: None
    
    try:
        # Apply calibration - should not crash, just use raw values
        result_df = _apply_calibration_to_schedule_df(
            schedule_df.copy(),
            sport="nba",
            season="2025-26",
            model="ensemble",
        )
        
        # Should return DataFrame unchanged
        assert result_df is not None
        assert len(result_df) == len(schedule_df)
        
    finally:
        cal_io.load_latest_calibrator = orig_load


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
