#!/usr/bin/env python
"""
Quick canonization health check - run this daily to verify system integrity.

Usage:
    python scripts/verify_canonization.py

Checks:
1. Ensemble imports work
2. No duplicate games in databases
3. Core models are registered
4. Game ID format is consistent
5. Database schema is intact
"""

import sys
import sqlite3
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def check_ensemble_imports() -> bool:
    """Verify all ensemble classes are importable."""
    try:
        from ensemble.ml_v1 import MLWeightedAverageEnsemble
        from ensemble.spread_v1 import SpreadWeightedAverageEnsemble
        from ensemble.total_v1 import TotalWeightedAverageEnsemble
        print("✅ Ensemble imports: OK")
        return True
    except ImportError as e:
        print(f"❌ Ensemble imports: FAILED - {e}")
        return False


def check_duplicate_games(db_path: Path) -> bool:
    """Check for duplicate games in database."""
    if not db_path.exists():
        print(f"⚠️  Database not found: {db_path}")
        return True
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) FROM (
            SELECT date, home_team, away_team, COUNT(*) as cnt
            FROM games
            GROUP BY date, home_team, away_team
            HAVING cnt > 1
        )
    """)
    duplicates = cursor.fetchone()[0]
    conn.close()
    
    if duplicates == 0:
        print(f"✅ Duplicate games ({db_path.name}): None")
        return True
    else:
        print(f"❌ Duplicate games ({db_path.name}): {duplicates} found")
        return False


def check_model_registry() -> bool:
    """Verify core models are registered."""
    try:
        from models.registry import list_models, list_backtest_models
        
        all_models = set(list_models())
        backtest_models = set(list_backtest_models())
        required_models = {"elo", "bradley-terry", "poisson"}
        
        if required_models.issubset(all_models) and required_models.issubset(backtest_models):
            print(f"✅ Model registry: OK ({len(all_models)} models)")
            return True
        else:
            missing = required_models - all_models
            print(f"❌ Model registry: Missing {missing}")
            return False
    except Exception as e:
        print(f"❌ Model registry: FAILED - {e}")
        return False


def check_game_id_format(db_path: Path) -> bool:
    """Verify game IDs use canonical format."""
    if not db_path.exists():
        return True
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Sample some game IDs
    cursor.execute("SELECT game_id FROM games LIMIT 100")
    game_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    if not game_ids:
        return True
    
    # Check for pipe format (deprecated)
    pipe_format_count = sum(1 for gid in game_ids if "|" in str(gid))
    if pipe_format_count > 0:
        print(f"❌ Game ID format ({db_path.name}): {pipe_format_count} pipe-format IDs found")
        return False
    
    # Check for hash format (canonical)
    hash_format_count = sum(1 for gid in game_ids if ":" in str(gid) and len(str(gid).split(":")[-1]) == 12)
    if hash_format_count == len(game_ids):
        print(f"✅ Game ID format ({db_path.name}): All canonical")
        return True
    else:
        print(f"⚠️  Game ID format ({db_path.name}): {hash_format_count}/{len(game_ids)} canonical")
        return True


def check_db_schema(db_path: Path) -> bool:
    """Verify database has required tables."""
    if not db_path.exists():
        return True
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    conn.close()
    
    required_tables = {
        "games",
        "model_metrics",
        "model_market_tuning_runs",
        "model_market_active_params",
    }
    
    if required_tables.issubset(tables):
        print(f"✅ Database schema ({db_path.name}): OK ({len(tables)} tables)")
        return True
    else:
        missing = required_tables - tables
        print(f"❌ Database schema ({db_path.name}): Missing {missing}")
        return False


def main():
    """Run all canonization checks."""
    print("=" * 60)
    print("CANONIZATION HEALTH CHECK")
    print("=" * 60)
    
    results = []
    
    # Check ensemble imports
    results.append(check_ensemble_imports())
    
    # Check model registry
    results.append(check_model_registry())
    
    # Check databases
    db_dir = Path(__file__).parent.parent / "data" / "db"
    if db_dir.exists():
        for sport_dir in db_dir.iterdir():
            if sport_dir.is_dir():
                for db_file in sport_dir.glob("*.db"):
                    results.append(check_duplicate_games(db_file))
                    results.append(check_game_id_format(db_file))
                    results.append(check_db_schema(db_file))
    
    print("=" * 60)
    if all(results):
        print("✅ CANONIZATION: HEALTHY")
        print("=" * 60)
        return 0
    else:
        failed = len([r for r in results if not r])
        print(f"❌ CANONIZATION: {failed} CHECKS FAILED")
        print("=" * 60)
        print("\nRecommended actions:")
        print("1. Run: pytest tests/test_pipeline_canonization.py -v")
        print("2. Check docs/CANONIZATION_CHECKLIST.md for troubleshooting")
        print("3. Review recent changes that may have broken invariants")
        return 1


if __name__ == "__main__":
    sys.exit(main())
