# Phase 6: Ensemble Governance - Implementation Summary

## Completed Work

### 1. Market Model Allowlists (src/config.py)
- Added `MARKET_MODEL_ALLOWLISTS` dict with per-market model allowlists (ML, SPREAD, TOTAL)
- Single source of truth for which models are used per market
- All 5 models included: bradley-terry, elo, gssd, poisson, toor

### 2. Weight Governance Constants (src/config.py)
- `MIN_WEIGHT_EPS = 0.01`: Weights below this are clamped to 0
- `MIN_NEFF = 1.5`: Minimum effective model count threshold
- `ENSEMBLE_STRICT_MODE = False`: Fail vs allow fallback on threshold violations

### 3. EnsembleAudit Dataclass (src/ensemble/audit.py - NEW)
- Comprehensive audit object tracking:
  - `final_models`: Models in final ensemble
  - `final_weights`: Post-governance weights
  - `weight_source`: Where weights came from
  - `dropped_models`: Dict of model → drop reason
  - `coverage_summary`: Per-model forecast coverage
  - `neff`: Effective model count (1 / sum(w_i^2))
  - `neff_threshold_met`: Boolean for threshold check
  - `weight_clamped`: Whether clamping occurred
  - `fallback_applied`: Whether uniform fallback used
  - `warnings`: List of all warnings
- Methods:
  - `calculate_neff()`: Compute Neff from weights
  - `to_dict()`: Serialize to dict
  - `emit_log()`: Single INFO log with full audit trail
- Utility function: `compute_neff(weights)` for standalone calculation

### 4. Weight Resolution Function (src/pipelines/schedule.py)
- New function: `_resolve_ensemble_weights_with_audit()`
- Replaces simple filtering with comprehensive governance:
  1. Identify forecast models and validity
  2. Build candidate weights
  3. Filter based on validity
  4. **Clamp** weights below MIN_WEIGHT_EPS
  5. Handle fallbacks (weight collapse, no positive weights)
  6. Normalize weights to sum to 1.0
  7. Calculate Neff and check threshold
  8. Populate audit object
  9. **Emit audit log** (never silent)
- Returns: `EnsembleAudit` object with full details
- Strict mode: Raises `ValueError` if Neff < MIN_NEFF
- Non-strict mode: Logs warning, allows to proceed

### 5. Updated Imports (src/pipelines/schedule.py)
- Added: `from ensemble.audit import EnsembleAudit, compute_neff`
- Added imports for governance constants: `MIN_WEIGHT_EPS, MIN_NEFF, ENSEMBLE_STRICT_MODE, MARKET_MODEL_ALLOWLISTS`

### 6. Comprehensive Test Suite (tests/test_phase6_ensemble_governance.py - NEW)
25 tests covering:
- **Neff calculation**: uniform, skewed, single-model, empty weights
- **EnsembleAudit**: creation, serialization, logging
- **Weight clamping**: below-threshold, at-threshold, audit tracking
- **Neff threshold**: strict/non-strict modes, above/below threshold
- **Fallback scenarios**: weight collapse, no silent fallback
- **Dropped models**: missing forecasts, zero weight, below-eps, missing columns
- **Regression test**: weight collapse scenario (fails strict, logs non-strict)
- **Coverage tracking**: structure, missing columns

All 25 tests passing ✓

### 7. Documentation
- **PHASE6_ENSEMBLE_GOVERNANCE.md**: Complete governance architecture guide
  - Overview of problems solved
  - Component descriptions with code examples
  - Governance policies (clamping, Neff, fallback)
  - Integration points
  - Configuration options
  - Assumptions and trade-offs
  - Future work items
- **docs/ensembles.md**: Updated with Phase 6 reference
  - Added section linking to Phase 6 docs
  - Quick reference for config constants
  - Example audit log output

## Backward Compatibility

✓ **No breaking changes**
- Old `_filter_market_weights_for_forecast()` function remains unchanged
- New `_resolve_ensemble_weights_with_audit()` is additive
- Existing code continues to work
- New audit system is opt-in for now

## Key Design Decisions

1. **Single source of truth**: Market model allowlists in config.py (not scattered in code)
2. **Comprehensive audit**: All dropped models, coverage, and Neff tracked in one object
3. **No silent failures**: Always log warnings; never suppress feedback
4. **Deterministic**: Same inputs always produce same audit output
5. **Configurable governance**: Constants allow tuning without code changes
6. **Strict mode optional**: Default non-strict allows fallback with logging; strict fails fast

## Integration Checklist (For Future)

- [ ] Update schedule() ML ensemble section to use audit function (~line 3400)
- [ ] Update schedule() SPREAD ensemble section to use audit function (~line 3750)
- [ ] Update schedule() TOTAL ensemble section to use audit function (~line 3930)
- [ ] Store audit objects for traceability (logs, BETS sheet, or DB)
- [ ] Create metrics dashboard showing Neff, clamping, fallback rates
- [ ] Add per-sport overrides for MIN_NEFF and MIN_WEIGHT_EPS

## Test Results

```
tests/test_phase6_ensemble_governance.py::TestComputeNeff::test_uniform_weights_neff PASSED
tests/test_phase6_ensemble_governance.py::TestComputeNeff::test_one_model_neff PASSED
tests/test_phase6_ensemble_governance.py::TestComputeNeff::test_skewed_weights_neff PASSED
tests/test_phase6_ensemble_governance.py::TestComputeNeff::test_empty_weights_neff PASSED
tests/test_phase6_ensemble_governance.py::TestComputeNeff::test_zero_weights_neff PASSED
tests/test_phase6_ensemble_governance.py::TestEnsembleAudit::test_basic_audit_creation PASSED
tests/test_phase6_ensemble_governance.py::TestEnsembleAudit::test_audit_to_dict PASSED
tests/test_phase6_ensemble_governance.py::TestEnsembleAudit::test_calculate_neff PASSED
tests/test_phase6_ensemble_governance.py::TestEnsembleAudit::test_emit_log PASSED
tests/test_phase6_ensemble_governance.py::TestWeightClamping::test_weights_below_eps_clamped PASSED
tests/test_phase6_ensemble_governance.py::TestWeightClamping::test_weights_at_eps_threshold PASSED
tests/test_phase6_ensemble_governance.py::TestWeightClamping::test_clamping_with_audit PASSED
tests/test_phase6_ensemble_governance.py::TestNeffThreshold::test_neff_below_threshold_strict_mode PASSED
tests/test_phase6_ensemble_governance.py::TestNeffThreshold::test_neff_below_threshold_non_strict_mode PASSED
tests/test_phase6_ensemble_governance.py::TestNeffThreshold::test_neff_above_threshold PASSED
tests/test_phase6_ensemble_governance.py::TestFallbackScenarios::test_weight_collapse_triggers_fallback PASSED
tests/test_phase6_ensemble_governance.py::TestFallbackScenarios::test_no_silent_fallback PASSED
tests/test_phase6_ensemble_governance.py::TestDroppedModelsReasoning::test_dropped_missing_forecasts PASSED
tests/test_phase6_ensemble_governance.py::TestDroppedModelsReasoning::test_dropped_zero_weight PASSED
tests/test_phase6_ensemble_governance.py::TestDroppedModelsReasoning::test_dropped_below_eps PASSED
tests/test_phase6_ensemble_governance.py::TestDroppedModelsReasoning::test_dropped_missing_columns PASSED
tests/test_phase6_ensemble_governance.py::TestRegressionWeightCollapse::test_collapse_scenario_strict_mode PASSED
tests/test_phase6_ensemble_governance.py::TestRegressionWeightCollapse::test_collapse_scenario_non_strict_mode PASSED
tests/test_phase6_ensemble_governance.py::TestCoverageTracking::test_coverage_summary_structure PASSED
tests/test_phase6_ensemble_governance.py::TestCoverageTracking::test_coverage_with_missing_columns PASSED

======================== 25 passed in 0.08s ========================
```

## Files Modified/Created

### Created:
- `src/ensemble/audit.py` - EnsembleAudit dataclass and governance
- `tests/test_phase6_ensemble_governance.py` - Comprehensive test suite
- `PHASE6_ENSEMBLE_GOVERNANCE.md` - Full architecture documentation

### Modified:
- `src/config.py` - Added market allowlists and governance constants
- `src/pipelines/schedule.py` - Added audit function and imports
- `docs/ensembles.md` - Added Phase 6 reference section

## Usage Examples

### Basic Audit Creation
```python
from ensemble.audit import EnsembleAudit

audit = EnsembleAudit(
    market="ML",
    weight_source="db_best_run",
    final_models=["elo", "bradley-terry"],
    final_weights={"elo": 0.6, "bradley-terry": 0.4},
)
audit.neff = audit.calculate_neff()
audit.emit_log()  # Logs: [ensemble audit][ML] source=db_best_run Neff=2.0 ...
```

### Using Weight Resolution Function
```python
from pipelines.schedule import _resolve_ensemble_weights_with_audit

audit = _resolve_ensemble_weights_with_audit(
    weights={"elo": 0.5, "gssd": 0.5},
    forecast_df=forecast_df,
    market="SPREAD",
    weight_source="config",
)

print(f"Final models: {audit.final_models}")
print(f"Neff: {audit.neff:.2f}")
print(f"Dropped: {audit.dropped_models}")
audit.emit_log()
```

### Strict Mode Failure Handling
```python
from config import ENSEMBLE_STRICT_MODE

if ENSEMBLE_STRICT_MODE:
    # Will raise ValueError if Neff < MIN_NEFF
    try:
        audit = _resolve_ensemble_weights_with_audit(...)
    except ValueError as e:
        # Handle ensemble degradation
        log.error(f"Ensemble failed: {e}")
```

## Notes

- Existing test failures (12 failures pre-existed before Phase 6 work)
- Phase 6 adds ~400 lines of new code (audit module, tests, docs)
- No changes to model math, head math, calibration, or probability formulas
- Backward compatible: old filter function untouched
- Ready for integration into main ensemble pipeline sections
