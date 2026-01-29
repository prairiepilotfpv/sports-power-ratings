# Phase 6: Ensemble System Hardening & Governance

## Executive Summary

Phase 6 hardens the ensemble system to be explicit, deterministic, and auditable. Previous phases (1-5) completed heads migration. Phase 6 tackles the orthogonal problem: **ensemble weight governance and visibility**.

**Problem solved:**
- Weight collapse (e.g., tuned weights for 3 models → only 1 available) silently fell back to uniform
- Multiple fallback paths with inconsistent logging made troubleshooting hard
- Neff (effective model count) had no threshold; degraded ensembles were treated as valid
- No visibility into why models were dropped or weights were clamped

**Solution:**
- **EnsembleAudit** dataclass tracks all decisions: dropped models, coverage, Neff
- **Weight governance** with configurable clamping and Neff thresholds
- **Strict mode** fails fast on degradation instead of silent fallback
- **Single INFO log** per market with full audit trail (never silent)

## What Changed

### 1. Configuration (src/config.py)

New constants define ensemble governance:

```python
# Market model allowlists (single source of truth)
MARKET_MODEL_ALLOWLISTS: dict[str, list[str]] = {
    "ML": ["bradley-terry", "elo", "gssd", "poisson", "toor"],
    "SPREAD": ["bradley-terry", "elo", "gssd", "poisson", "toor"],
    "TOTAL": ["bradley-terry", "elo", "gssd", "poisson", "toor"],
}

# Weight governance policy
MIN_WEIGHT_EPS: float = 0.01  # Clamp weights < 0.01 to 0
MIN_NEFF: float = 1.5  # Minimum effective model count
ENSEMBLE_STRICT_MODE: bool = False  # Fail vs allow fallback
```

### 2. Audit Module (src/ensemble/audit.py)

New module with `EnsembleAudit` dataclass:

```python
@dataclass(frozen=False)
class EnsembleAudit:
    """Tracks weight resolution decisions."""
    market: str
    final_models: list[str]
    final_weights: dict[str, float]
    weight_source: str  # db_best_run, config, file, fallback, etc.
    dropped_models: dict[str, str]  # model -> reason
    coverage_summary: dict[str, dict]  # per-model coverage
    neff: float  # Effective model count
    neff_threshold_met: bool
    weight_clamped: bool
    fallback_applied: bool
    warnings: list[str]
```

### 3. Weight Resolution Function (src/pipelines/schedule.py)

New function `_resolve_ensemble_weights_with_audit()` that:
1. Filters weights based on forecast availability
2. Clamps weights below `MIN_WEIGHT_EPS` to 0
3. Normalizes weights to sum to 1.0
4. Calculates Neff and checks threshold
5. Returns comprehensive `EnsembleAudit` object
6. **Always logs** (never silent)

### 4. Test Suite (tests/test_phase6_ensemble_governance.py)

25 comprehensive tests covering:
- Neff calculation for uniform, skewed, and single-model ensembles
- Weight clamping behavior
- Neff threshold enforcement (strict vs non-strict)
- Fallback detection and logging
- Dropped model reasoning
- Regression test: weight collapse scenario
- Coverage tracking

All tests passing ✓

## How It Works

### Weight Governance Flow

```
Candidate Weights: {"elo": 0.4, "gssd": 0.3, "poisson": 0.3}
Available Models: {"elo", "gssd"}  (poisson has no margin_sd)

1. Filter by availability:
   {"elo": 0.4, "gssd": 0.3}

2. Clamp (MIN_WEIGHT_EPS = 0.01):
   Both above 0.01, no clamping

3. Normalize to sum to 1.0:
   {"elo": 0.4/0.7 ≈ 0.57, "gssd": 0.3/0.7 ≈ 0.43}

4. Calculate Neff = 1 / sum(w_i^2):
   1 / (0.57^2 + 0.43^2) ≈ 1.98

5. Check threshold (MIN_NEFF = 1.5):
   1.98 >= 1.5 ✓ threshold met

6. Emit audit log:
   [ensemble audit][SPREAD] source=db_best_run Neff=1.98
   models=['elo', 'gssd'] weights={'elo': 0.57, 'gssd': 0.43}
```

### Fallback Detection (No Silent Collapse)

When weights collapse to 1 model:

```
Tuned Weights: {"elo": 0.4, "gssd": 0.3, "poisson": 0.3}
Available: {"elo"}  (gssd & poisson missing)

After filtering: {"elo": 1.0}
Neff = 1.0 < MIN_NEFF = 1.5 ← threshold not met

Audit output:
  final_models = ["elo"]
  fallback_applied = True
  warnings = ["Tuned weights collapsed to 1 model"]
  neff_threshold_met = False

Strict mode (ENSEMBLE_STRICT_MODE=True):
  → Raises ValueError with details

Non-strict mode (default):
  → Logs warning, allows to proceed
```

## Configuration Examples

### Disable Strict Mode (Default - Allow Fallback)

```python
# In src/config.py
ENSEMBLE_STRICT_MODE = False  # Graceful fallback with warnings
```

### Enable Strict Mode (Fail Fast)

```python
# In src/config.py
ENSEMBLE_STRICT_MODE = True  # Fail on Neff < threshold or weight collapse
```

### Adjust Thresholds

```python
# In src/config.py
MIN_WEIGHT_EPS = 0.02  # More aggressive clamping
MIN_NEFF = 2.0  # Require higher effective model count
```

## Audit Output Example

```
[ensemble audit][ML] source=db_best_run Neff=2.8 (threshold=met) 
models=['elo', 'bradley-terry'] 
weights={'elo': 0.6, 'bradley-terry': 0.4} 
dropped={'gssd': 'weight=0.0'} 
weight_clamped=false fallback_applied=false

[ensemble audit][ML] Neff=1.2 < MIN_NEFF=1.5 (non-strict mode, allowing)
```

## Running Tests

```bash
# Run Phase 6 tests
python -m pytest tests/test_phase6_ensemble_governance.py -v

# Run Phase 4/5 tests (verify no regression)
python -m pytest tests/test_phase4_heads_contract.py tests/test_elo_heads_equivalence.py -q

# Run all tests
python -m pytest tests/ -q
```

## Integration Checklist

- [x] Audit dataclass created
- [x] Weight governance constants added
- [x] Weight resolution function with audit implemented
- [x] Comprehensive test suite created (all passing)
- [x] Documentation (PHASE6_ENSEMBLE_GOVERNANCE.md)
- [ ] Integrate audit function into schedule() ML/SPREAD/TOTAL sections
- [ ] Store audit objects for traceability
- [ ] Create metrics dashboard

## Backward Compatibility

✓ **No breaking changes**
- Old `_filter_market_weights_for_forecast()` unchanged
- New audit function is opt-in
- Existing code continues to work as-is
- Phase 4/5 tests still passing

## Key Files

| File | Purpose |
|------|---------|
| [src/ensemble/audit.py](src/ensemble/audit.py) | EnsembleAudit dataclass, governance utilities |
| [src/config.py](src/config.py) | Market allowlists, governance constants |
| [src/pipelines/schedule.py](src/pipelines/schedule.py) | `_resolve_ensemble_weights_with_audit()` function |
| [tests/test_phase6_ensemble_governance.py](tests/test_phase6_ensemble_governance.py) | 25 comprehensive tests |
| [PHASE6_ENSEMBLE_GOVERNANCE.md](PHASE6_ENSEMBLE_GOVERNANCE.md) | Detailed architecture guide |
| [docs/ensembles.md](docs/ensembles.md) | Updated with Phase 6 reference |

## Next Steps

1. **Integration**: Update schedule() ensemble sections to use audit function
2. **Metrics**: Build dashboard showing Neff, clamping, fallback rates
3. **Traceability**: Store audit objects in BETS sheet or logs
4. **Per-sport tuning**: Allow sport-specific overrides of MIN_NEFF, MIN_WEIGHT_EPS

## References

- **Architecture**: [PHASE6_ENSEMBLE_GOVERNANCE.md](PHASE6_ENSEMBLE_GOVERNANCE.md)
- **Implementation summary**: [PHASE6_IMPLEMENTATION_SUMMARY.md](PHASE6_IMPLEMENTATION_SUMMARY.md)
- **Ensemble guide**: [docs/ensembles.md](docs/ensembles.md)
