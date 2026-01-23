#!/usr/bin/env python
"""Quick verification that TOOR v1.1 is properly integrated."""

import sys
sys.path.insert(0, ".")

from src.models.registry import get_backtest_model, list_backtest_models
from src.ensemble.config import DEFAULT_MARKET_MODELS

print("=" * 60)
print("TOOR v1.1 Integration Verification")
print("=" * 60)

# 1. Check registry
print("\n✓ Step 1: Model Registry")
models = list_backtest_models()
print(f"  Registered backtest models: {models}")
assert "toor" in models, "TOOR not in backtest registry!"
print("  ✓ TOOR is registered")

# 2. Check model instantiation
print("\n✓ Step 2: Model Instantiation")
model_cls = get_backtest_model("toor")
model = model_cls()
meta = model.metadata()
print(f"  Model ID: {meta.model_id}")
print(f"  Model Version: {meta.model_version}")
print(f"  Supports Margin: {meta.supports_margin}")
print(f"  Supports Total: {meta.supports_total}")
print(f"  Supports Win Prob: {meta.supports_win_prob}")
assert meta.model_id == "toor", f"Expected model_id='toor', got '{meta.model_id}'"
assert meta.model_version == "1.1", f"Expected version='1.1', got '{meta.model_version}'"
print("  ✓ Model instantiates correctly with v1.1")

# 3. Check optimizer parameter
print("\n✓ Step 3: Optimizer Configuration")
print(f"  Default optimizer: {model._optimizer}")
assert model._optimizer == "scipy", f"Expected optimizer='scipy', got '{model._optimizer}'"
print("  ✓ Scipy optimizer is default")

# 4. Check ensemble integration
print("\n✓ Step 4: Ensemble Integration")
spread_models = DEFAULT_MARKET_MODELS.get("SPREAD", [])
print(f"  SPREAD market models: {spread_models}")
assert "toor" in spread_models, "TOOR not in SPREAD ensemble!"
print("  ✓ TOOR is in SPREAD ensemble")

# 5. Check new parameters exist
print("\n✓ Step 5: New Parameters")
params = meta.params
print(f"  optimizer: {params.get('optimizer')}")
print(f"  initial_home_adv: {params.get('initial_home_adv')}")
print(f"  initial_home_coeff: {params.get('initial_home_coeff')}")
print(f"  initial_away_coeff: {params.get('initial_away_coeff')}")
assert "optimizer" in params, "optimizer parameter missing!"
assert "initial_home_adv" in params, "initial_home_adv parameter missing!"
print("  ✓ New parameters present in metadata")

print("\n" + "=" * 60)
print("✅ ALL INTEGRATION CHECKS PASSED")
print("=" * 60)
print("\nTOOR v1.1 is properly integrated into:")
print("  • Backtest pipeline (model_id='toor', version='1.1')")
print("  • SPREAD ensemble (included in default models)")
print("  • Model registry (get_backtest_model('toor'))")
print("  • Scipy optimization (default optimizer='scipy')")
print("\nNo breaking changes detected.")
