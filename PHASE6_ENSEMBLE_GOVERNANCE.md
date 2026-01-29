# Phase 6: Ensemble Governance Hardening

## Overview

Phase 6 hardened the ensemble system to be explicit, deterministic, and auditable. The core problem: weight collapse and silent fallbacks were hiding issues, making it impossible to know when ensembles were degrading.

**Solutions implemented:**
1. **Explicit per-market model allowlists** (single source of truth in `src/config.py`)
2. **EnsembleAudit dataclass** that tracks dropped models, coverage, and Neff
3. **Weight governance policy** with clamping and Neff thresholds
4. **Strict mode** that fails fast instead of silently degrading

## Key Components

### 1. Market Model Allowlists (src/config.py)

Single source of truth for which models are used per market:

```python
MARKET_MODEL_ALLOWLISTS: dict[str, list[str]] = {
    "ML": ["bradley-terry", "elo", "gssd", "poisson", "toor"],
    "SPREAD": ["bradley-terry", "elo", "gssd", "poisson", "toor"],
    "TOTAL": ["bradley-terry", "elo", "gssd", "poisson", "toor"],
}
```

Used by:
- Forecast generation (which models are built)
- Ensemble weight resolution (which models are considered)
- Validator/audit reporting

### 2. Weight Governance Constants (src/config.py)

```python
# MIN_WEIGHT_EPS: weights below this are clamped to 0 then renormalized
MIN_WEIGHT_EPS: float = 0.01

# MIN_NEFF: minimum effective model count (Neff = 1 / sum(w_i^2))
# If Neff < MIN_NEFF in strict mode, raise RuntimeError
MIN_NEFF: float = 1.5

# ENSEMBLE_STRICT_MODE: when True, fail instead of silently degrading
ENSEMBLE_STRICT_MODE: bool = False
```

### 3. EnsembleAudit Dataclass (src/ensemble/audit.py)

Comprehensive audit object returned by weight resolution:

```python
@dataclass(frozen=False)
class EnsembleAudit:
    """Audit object from weight resolution and governance."""
    
    market: str  # Market name (ML, SPREAD, TOTAL)
    final_models: list[str]  # Models in final ensemble (after filtering)
    final_weights: dict[str, float]  # Post-clamp, post-governance weights
    weight_source: str  # Source (db_best_run, config, file, fallback, etc.)
    dropped_models: dict[str, str]  # Model -> reason_for_drop
    coverage_summary: dict[str, dict]  # Model -> coverage details
    neff: float  # Effective model count (1 / sum(w_i^2))
    neff_threshold_met: bool  # Whether Neff >= MIN_NEFF
    weight_run_id: Optional[str]  # Optional tuning run ID
    selection_run_id: Optional[str]  # Optional selection run ID
    weight_clamped: bool  # Whether any weights were clamped to 0
    fallback_applied: bool  # Whether fallback to uniform was used
    warnings: list[str]  # List of warnings emitted
```

#### Key Methods

- `calculate_neff()`: Compute Neff from final_weights
- `to_dict()`: Serialize to dict (for logging, persistence)
- `emit_log()`: Emit single INFO log with full audit trail

### 4. Weight Resolution with Audit (src/pipelines/schedule.py)

New function `_resolve_ensemble_weights_with_audit()` replaces pure filtering:

```python
def _resolve_ensemble_weights_with_audit(
    *,
    weights: dict[str, float] | None,
    forecast_df: pd.DataFrame,
    market: str | Market,
    weight_source: str,
    weight_run_id: str | None = None,
    selection_run_id: str | None = None,
) -> EnsembleAudit:
    """Resolve weights with full audit, governance, and strict mode."""
```

Steps:
1. **Identify forecast models** and their validity (required columns present)
2. **Build candidate weights** from provided weights or uniform
3. **Filter** based on validity and required columns
4. **Clamp** weights below MIN_WEIGHT_EPS to 0
5. **Handle fallbacks** (weight collapse, no positive weights)
6. **Normalize** weights to sum to 1.0
7. **Calculate Neff** and check threshold
8. **Populate audit** with all details
9. **Emit log** with full audit trail

## Governance Policies

### Weight Clamping

Weights below `MIN_WEIGHT_EPS` (default 0.01) are set to 0 and renormalized:

```python
# Before: {"model1": 0.5, "model2": 0.004}
# After clamping: {"model1": 0.5}
# After renormalization: {"model1": 1.0}

# Audit tracks:
# - weight_clamped = True
# - dropped_models["model2"] = "weight=0.004 < MIN_WEIGHT_EPS=0.01"
```

### Neff Threshold Enforcement

Effective model count must be >= `MIN_NEFF` (default 1.5):

- **Strict mode (ENSEMBLE_STRICT_MODE=True)**: If Neff < MIN_NEFF, raise `ValueError`
- **Non-strict mode (default)**: Log warning, allow to proceed

```python
# Example: weights = {"model1": 0.99, "model2": 0.01}
# Neff = 1 / (0.99^2 + 0.01^2) ≈ 1.02 < 1.5
# 
# Strict: raises ValueError
# Non-strict: logs warning, audit.warnings includes Neff message
```

### Fallback Behavior

When weights collapse to 1 model or no positive weights remain:

1. **If valid models ≥ 2**: Fall back to uniform weights
2. **If valid models < 2**: In strict mode, raise error; in non-strict, allow empty
3. **Always log loudly**: `audit.warnings` and `emit_log()` ensure visibility

Example:
```python
# Tuned weights: {"elo": 0.4, "gssd": 0.3, "poisson": 0.3}
# Available in forecast: only {"elo"}
# After filtering: {"elo": 1.0}
# Neff = 1.0 < MIN_NEFF=1.5
#
# Audit warnings:
#   "Tuned weights for ML collapsed to 1 model(s) after filtering"
# 
# Strict mode: raises RuntimeError with details
# Non-strict: logs warning, allows ensemble with 1 model
```

## Per-Game Coverage Tracking (Heads Mode)

`coverage_summary` in audit object tracks per-model forecast coverage:

```python
coverage_summary = {
    "elo": {
        "games_with_forecasts": 150,
        "required_columns": ["p_home_win"],
        "missing_columns": [],
    },
    "bradley-terry": {
        "games_with_forecasts": 120,
        "required_columns": ["p_home_win"],
        "missing_columns": [],
    },
}
```

In heads mode, ensures consistent coverage across models:
- If a model is in `final_models` but missing rows for any target games, drop it with reason "missing forecast rows"
- In strict mode, if dropping causes insufficient model count or Neff < MIN_NEFF, raise

## Audit Logging

Single comprehensive INFO log per market:

```
[ensemble audit][ML] source=db_best_run Neff=2.8 (threshold=met) 
models=['elo', 'bradley-terry'] 
weights={'elo': 0.6, 'bradley-terry': 0.4} 
dropped={'gssd': 'weight=0.0'} weight_clamped=false fallback_applied=false
```

Warnings emitted separately at WARNING level:
```
[ensemble audit][ML] Neff=1.2 < MIN_NEFF=1.5 (non-strict mode, allowing)
```

## Integration Points

### Existing Code Changes

- **src/config.py**: Added market allowlists and governance constants
- **src/pipelines/schedule.py**: 
  - Added imports for audit module and config constants
  - Created new function `_resolve_ensemble_weights_with_audit()`
  - Old `_filter_market_weights_for_forecast()` remains for backward compatibility

### Where to Integrate

Future integration points (not yet modified to preserve existing behavior):
- ML/SPREAD/TOTAL ensemble sections in `schedule()` function (~lines 3400-4200)
- Should call `_resolve_ensemble_weights_with_audit()` instead of `_filter_market_weights_for_forecast()`
- Use `audit.final_models` and `audit.final_weights` instead of separate filter results

## Testing

### Phase 6 Tests (tests/test_phase6_ensemble_governance.py)

25 comprehensive tests covering:
- **Neff calculation** (uniform, skewed, single-model, empty)
- **EnsembleAudit dataclass** (creation, serialization, logging)
- **Weight clamping** (below-threshold, at-threshold, with audit)
- **Neff threshold** (strict/non-strict modes, above/below threshold)
- **Fallback scenarios** (weight collapse, no silent fallback)
- **Dropped model reasoning** (missing forecasts, zero weight, below-eps, missing columns)
- **Regression test** (weight collapse scenario; fails in strict, logs in non-strict)
- **Coverage tracking** (structure, missing columns)

Run with:
```bash
python -m pytest tests/test_phase6_ensemble_governance.py -v
```

### Expected Behavior

**No regressions**: Phase 6 adds new functionality without changing existing `_filter_market_weights_for_forecast()` behavior.

**Audit is optional**: Existing code continues to work; new code can opt-in to audit by calling `_resolve_ensemble_weights_with_audit()`.

## Configuration

### Strict Mode Toggle

Enable strict mode to fail fast on degradation:

```python
# In src/config.py
ENSEMBLE_STRICT_MODE: bool = True  # Enable strict mode
```

Or override at runtime:
```python
import config
config.ENSEMBLE_STRICT_MODE = True
```

### Tuning Constants

Adjust weight clamping and Neff thresholds:

```python
# In src/config.py
MIN_WEIGHT_EPS: float = 0.01  # Clamp weights below this
MIN_NEFF: float = 1.5  # Require at least this many "effective models"
```

## Assumptions & Trade-offs

1. **Backward compatibility**: Old filter function remains; audit is additive
2. **Determinism**: Same inputs → same audit output (no randomness)
3. **Non-silent failures**: Warnings always logged, never suppressed
4. **Coverage tracking**: Assumes `game_id` column exists in forecast_df for game counting

## Future Work

1. **Integration**: Update schedule() ensemble sections to use audit function
2. **Persistence**: Store audit objects in BETS sheet or logs for traceability
3. **Metrics**: Dashboard showing Neff, weight clamping frequency, fallback rate
4. **Per-sport tuning**: Allow per-sport overrides of MIN_NEFF, MIN_WEIGHT_EPS

## References

- **Implementation**: `src/ensemble/audit.py`, `src/pipelines/schedule.py::_resolve_ensemble_weights_with_audit()`
- **Config**: `src/config.py::MARKET_MODEL_ALLOWLISTS`, `MIN_WEIGHT_EPS`, `MIN_NEFF`, `ENSEMBLE_STRICT_MODE`
- **Tests**: `tests/test_phase6_ensemble_governance.py`
