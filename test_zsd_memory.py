#!/usr/bin/env python
"""Test script to verify ZSD model memory cleanup during tuning."""

import psutil
import os
import gc
import sys
from pathlib import Path

# Get current process
process = psutil.Process(os.getpid())

def get_memory_mb():
    """Get current memory usage in MB."""
    return process.memory_info().rss / 1024 / 1024

print("=" * 60)
print("ZSD Memory Cleanup Test")
print("=" * 60)

# Initial memory
initial_mem = get_memory_mb()
print(f"Initial memory: {initial_mem:.1f} MB")

# Test imports
try:
    import numpy as np
    import pandas as pd
    from src.models.zsd import ZSDModel
    print("✓ Imports successful")
except ImportError as e:
    # Try alternative import path
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        import numpy as np
        import pandas as pd
        from models.zsd import ZSDModel
        print("✓ Imports successful (via src path)")
    except ImportError as e2:
        print(f"✗ Import error: {e} then {e2}")
        sys.exit(1)

mem_after_import = get_memory_mb()
print(f"Memory after imports: {mem_after_import:.1f} MB (Δ {mem_after_import - initial_mem:.1f} MB)")

# Create a small dataset for testing
np.random.seed(42)
n_games = 500
n_teams = 20

teams = [f"Team{i}" for i in range(n_teams)]
games = []
for _ in range(n_games):
    home = np.random.choice(teams)
    away = np.random.choice([t for t in teams if t != home])
    home_score = np.random.poisson(105)
    away_score = np.random.poisson(100)
    games.append({
        "date": "2025-01-01",
        "home_team": home,
        "away_team": away,
        "home_score": home_score,
        "away_score": away_score,
    })

games_df = pd.DataFrame(games)
print(f"\nTest dataset: {len(games)} games, {len(teams)} teams")

# Test ZSD fit
print("\n" + "=" * 60)
print("Testing ZSD fit with new memory cleanup...")
print("=" * 60)

mem_before_fit = get_memory_mb()
print(f"Memory before fit: {mem_before_fit:.1f} MB")

model = ZSDModel(max_iter=5000, optimizer="slsqp")
model.fit(games_df)

mem_after_fit = get_memory_mb()
fit_delta = mem_after_fit - mem_before_fit
print(f"Memory after fit: {mem_after_fit:.1f} MB (Δ {fit_delta:.1f} MB)")

# Delete model and force garbage collection
del model
gc.collect()

mem_after_cleanup = get_memory_mb()
cleanup_delta = mem_before_fit - mem_after_cleanup
print(f"Memory after cleanup: {mem_after_cleanup:.1f} MB")
print(f"Memory recovered: {cleanup_delta:.1f} MB" + (" ✓" if cleanup_delta > 0 else " (no recovery)"))

# Test multiple fits (simulating tuning)
print("\n" + "=" * 60)
print("Testing multiple ZSD fits (simulating tuning)...")
print("=" * 60)

mem_before_multi = get_memory_mb()
print(f"Memory before multiple fits: {mem_before_multi:.1f} MB")

models = []
for i in range(3):
    print(f"  Fit {i+1}/3...", end="")
    m = ZSDModel(max_iter=3000, optimizer="slsqp")
    m.fit(games_df)
    models.append(m)
    mem_iter = get_memory_mb()
    print(f" {mem_iter:.1f} MB")

mem_with_models = get_memory_mb()
print(f"Memory with 3 models in memory: {mem_with_models:.1f} MB (Δ {mem_with_models - mem_before_multi:.1f} MB)")

# Clear models
del models
gc.collect()

mem_final = get_memory_mb()
final_recovery = mem_with_models - mem_final
print(f"Memory after clearing all models: {mem_final:.1f} MB")
print(f"Memory recovered: {final_recovery:.1f} MB" + (" ✓" if final_recovery > 50 else " (limited recovery)"))

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
peak_mem = max(mem_with_models, mem_after_fit)
print(f"Peak memory: {peak_mem:.1f} MB")
print(f"Memory overhead vs initial: {peak_mem - initial_mem:.1f} MB")
print("\nIf memory usage is reasonable and cleanup is effective,")
print("the test passed. If memory stays high (>500 MB delta),")
print("there may still be a leak.")
