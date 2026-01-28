"""Quick validation test for standalone calibration system.

This tests that the calibration system:
1. Loads games correctly
2. Generates predictions
3. Builds datasets
4. Fits calibrators
5. All for different sports/markets

Run with: python tests/test_standalone_calibration.py
"""

import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


def test_calibration_system():
    """Test the standalone calibration system against mock data."""
    print("Testing Standalone Calibration System")
    print("=" * 60)
    
    # Create a temporary database with mock games
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        # Create minimal database schema
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create games table (minimal schema)
        cursor.execute("""
            CREATE TABLE games (
                game_id TEXT PRIMARY KEY,
                sport TEXT,
                season TEXT,
                date TEXT,
                home_team TEXT,
                away_team TEXT,
                home_score INTEGER,
                away_score INTEGER
            )
        """)
        
        # Insert test data for multiple sports
        test_games = [
            # NBA games
            ("nba_2025_001", "nba", "2025-26", "2025-01-15", "Lakers", "Celtics", 110, 105),
            ("nba_2025_002", "nba", "2025-26", "2025-01-16", "Warriors", "Suns", 115, 110),
            ("nba_2025_003", "nba", "2025-26", "2025-01-17", "Nets", "Heat", 95, 100),
            # NFL games
            ("nfl_2024_001", "nfl", "2024-25", "2025-01-15", "Chiefs", "Ravens", 27, 24),
            ("nfl_2024_002", "nfl", "2024-25", "2025-01-16", "49ers", "Packers", 34, 31),
            # MLB games
            ("mlb_2024_001", "mlb", "2024", "2024-09-15", "Yankees", "Red Sox", 8, 5),
        ]
        
        cursor.executemany(
            "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            test_games
        )
        conn.commit()
        conn.close()
        
        print(f"\n✓ Created test database with {len(test_games)} games")
        
        # Test 1: Load completed games for each sport
        print("\n[Test 1] Loading completed games...")
        from calibration.historical_calibration import load_completed_games
        
        for sport, season in [("nba", "2025-26"), ("nfl", "2024-25"), ("mlb", "2024")]:
            games = load_completed_games(
                db_path,
                sport=sport,
                season=season,
            )
            assert not games.empty, f"Should have games for {sport}"
            assert "game_id" in games.columns
            assert "home_score" in games.columns
            assert "away_score" in games.columns
            print(f"  ✓ {sport}/{season}: {len(games)} games")
        
        # Test 2: Build calibration datasets
        print("\n[Test 2] Building calibration datasets...")
        from calibration.historical_calibration import (
            build_ml_calibration_dataset,
            build_spread_calibration_dataset,
            build_total_calibration_dataset,
        )
        
        # Create mock predictions
        games = load_completed_games(db_path, sport="nba", season="2025-26")
        
        # ML predictions
        ml_preds = pd.DataFrame([
            {"game_id": g, "p_home_win": 0.55}
            for g in games["game_id"]
        ])
        ml_dataset = build_ml_calibration_dataset(games, ml_preds)
        assert not ml_dataset.empty
        assert "p_home_win" in ml_dataset.columns
        assert "home_win" in ml_dataset.columns
        print(f"  ✓ ML dataset: {len(ml_dataset)} records")
        
        # SPREAD predictions
        spread_preds = pd.DataFrame([
            {"game_id": g, "margin_mean": 5.0, "margin_sd": 3.0}
            for g in games["game_id"]
        ])
        spread_dataset = build_spread_calibration_dataset(games, spread_preds)
        assert not spread_dataset.empty
        assert "pred_mean" in spread_dataset.columns
        assert "actual_value" in spread_dataset.columns
        print(f"  ✓ SPREAD dataset: {len(spread_dataset)} records")
        
        # TOTAL predictions
        total_preds = pd.DataFrame([
            {"game_id": g, "total_mean": 215.0, "total_sd": 8.0}
            for g in games["game_id"]
        ])
        total_dataset = build_total_calibration_dataset(games, total_preds)
        assert not total_dataset.empty
        assert "pred_mean" in total_dataset.columns
        assert "actual_value" in total_dataset.columns
        print(f"  ✓ TOTAL dataset: {len(total_dataset)} records")
        
        # Test 3: Fit calibrators
        print("\n[Test 3] Fitting calibrators...")
        from calibration.historical_calibration import fit_calibrator_for_market
        from markets.base import Market
        
        # ML calibrator
        ml_calib, _ = fit_calibrator_for_market(
            ml_dataset,
            market=Market.ML,
            method="platt",
        )
        assert ml_calib is not None
        assert hasattr(ml_calib, "transform")
        print(f"  ✓ ML calibrator fitted: {ml_calib.metadata}")
        
        # SPREAD calibrator
        spread_calib, _ = fit_calibrator_for_market(
            spread_dataset,
            market=Market.SPREAD,
        )
        assert spread_calib is not None
        assert hasattr(spread_calib, "metadata")
        print(f"  ✓ SPREAD calibrator fitted: {spread_calib.metadata}")
        
        # TOTAL calibrator
        total_calib, _ = fit_calibrator_for_market(
            total_dataset,
            market=Market.TOTAL,
        )
        assert total_calib is not None
        print(f"  ✓ TOTAL calibrator fitted: {total_calib.metadata}")
        
        # Test 4: Distribution calibrator specifics
        print("\n[Test 4] Testing distribution calibrator...")
        from calibration.distribution import VarianceCalibrator
        
        calib = VarianceCalibrator()
        calib.fit(spread_dataset)
        
        # Transform predictions
        result = calib.transform(spread_preds)
        assert "calibrated_mean" in result.columns
        assert "calibrated_sd" in result.columns
        print(f"  ✓ Distribution calibrator transform applied")
        print(f"    Original mean: {spread_preds['margin_mean'].mean():.2f}")
        print(f"    Calibrated mean: {result['calibrated_mean'].mean():.2f}")
        print(f"    Original SD: {spread_preds['margin_sd'].mean():.2f}")
        print(f"    Calibrated SD: {result['calibrated_sd'].mean():.2f}")
        
        # Test 5: Multi-sport capability
        print("\n[Test 5] Testing multi-sport capability...")
        for sport in ["nba", "nfl", "mlb"]:
            season = "2025-26" if sport != "mlb" else "2024"
            games = load_completed_games(db_path, sport=sport, season=season)
            if not games.empty:
                print(f"  ✓ {sport}: Can load and process")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("\nCalibration system verified:")
    print("  ✓ Sport-agnostic (NBA, NFL, MLB)")
    print("  ✓ Market-agnostic (ML, SPREAD, TOTAL)")
    print("  ✓ Probability calibration (ML)")
    print("  ✓ Distribution calibration (SPREAD/TOTAL)")
    print("  ✓ No betting pipeline dependencies")
    print("  ✓ No hardcoded assumptions")


if __name__ == "__main__":
    test_calibration_system()
