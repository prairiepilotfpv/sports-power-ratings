# Phase 4: Heads Mode Contract Enforcement - Implementation Summary

**Completion Date**: January 28, 2025  
**Status**: ✓ COMPLETE

## Executive Summary

Phase 4 eliminates remaining layered derivation paths and enforces a single, explicit market-neutral contract across the entire heads suite. Three new modules enforce:

1. **Derivation lockout** — projection engine cannot derive canonical fields in heads mode
2. **Model support matrix** — explicit ML/SPREAD/TOTAL capability per model
3. **Producer ID normalization** — standardized naming without "direct" terminology

**Test Results**: 56 new tests, all passing. 0 regressions in existing heads tests.

---

## Deliverables

### 1. Projection Engine Derivation Lockout ✓

**Module**: `src/pipelines/projection_engines.py` (+37 lines)

**What it does**:
- Defines canonical fields that projection engine must not derive in heads mode
- Raises `RuntimeError` if projection engine attempts to create/modify any canonical field when heads mode is enabled
- Keeps legacy behavior completely unchanged when heads mode is disabled

**Canonical Fields Protected**:
- `p_home_win`, `model_p_home_win` (ML)
- `margin_mean`, `margin_sd` (SPREAD)
- `total_mean`, `total_sd` (TOTAL)
- `projected_home_score`, `projected_away_score`, `projected_total` (derived scores)

**Implementation**:
```python
_CANONICAL_DERIVABLE_FIELDS = {
    "p_home_win", "model_p_home_win",
    "margin_mean", "margin_sd",
    "total_mean", "total_sd",
    "projected_home_score", "projected_away_score", "projected_total",
}

def _assert_derivation_locked(field: str, heads_mode: bool, context: ProjectionContext | None = None) -> None:
    """Enforce: if attempting canonical field derivation in heads mode, raise RuntimeError."""
```

**Testing**: 4 dedicated tests + integration with 33 producer tests

---

### 2. Model Support Matrix ✓

**Module**: `src/forecasting/model_support.py` (149 lines)

**What it does**:
- Defines which markets (ML, SPREAD, TOTAL) each model supports
- Provides filtering by capability
- Enables future optimization and validation

**Current Registry** (all Phase 1-3 heads models):
```
bradley-terry  ✓ ML  ✓ SPREAD  ✓ TOTAL
elo            ✓ ML  ✓ SPREAD  ✓ TOTAL
toor           ✓ ML  ✓ SPREAD  ✓ TOTAL
gssd           ✓ ML  ✓ SPREAD  ✓ TOTAL
poisson        ✓ ML  ✓ SPREAD  ✓ TOTAL
```

**Key Functions**:
```python
get_model_support(model_id: str) -> ModelSupport | None
get_supported_markets(model_id: str) -> list[str]
filter_models_for_market(model_ids: list[str], market: str) -> tuple[list[str], list[str]]
```

**Usage Example**:
```python
from forecasting.model_support import filter_models_for_market

models = ["elo", "bradley-terry", "unknown_model"]
supported, unsupported = filter_models_for_market(models, "SPREAD")
# supported = ["elo", "bradley-terry"]
# unsupported = ["unknown_model"]
```

**Testing**: 11 dedicated tests + 6 integration tests

---

### 3. Producer ID Normalization ✓

**Module**: `src/forecasting/producer_id.py` (174 lines)

**What it does**:
- Normalizes producer labels to canonical form
- Enforces market-specific naming contracts
- Removes "direct" terminology in heads mode
- Validates producer IDs for compliance

**Contract**:
- **ML market**: `ensemble_ml_v1` OR model name (e.g., `elo`)
- **SPREAD market**: `ensemble_spread_v1` ONLY
- **TOTAL market**: `ensemble_total_v1` ONLY
- **Calibration tags**: appended with `+` (e.g., `elo+calibrated_ml+calibrated_spread`)
- **No "direct"**: forbidden in heads mode

**Key Functions**:
```python
normalize_win_prob_source(source: str | None, market: Market, *, heads_mode: bool = False) -> str | None
get_ensemble_producer_id(market: Market) -> str
is_ensemble_producer(source: str | None) -> bool
is_valid_producer_in_market(source: str | None, market: Market) -> bool
validate_producer_in_heads_mode(source: str | None, market: Market) -> tuple[bool, str | None]
```

**Change Example** (Bradley-Terry projection engine):
```python
# Before
win_prob_source: "direct" if model_p is not None else "bt_margin_normal"

# After (normalized)
win_prob_source: "bradley-terry" if model_p is not None else "bt_margin_normal"
```

**Testing**: 33 dedicated tests covering normalization, validation, and market-specific rules

---

## Test Coverage

### Phase 4 Tests (56 total)

**File**: `tests/test_phase4_heads_contract.py` (23 tests)
- ModelSupportMatrix: 11 tests
  - Registry lookup, filtering, case-insensitivity, immutability
- ProjectionEngineDerivationLockout: 4 tests
  - Heads-enabled/disabled guards, context inclusion, field completeness
- ModelSupportIntegration: 6 tests
  - All models support all markets, filtering with duplicates
- CanonicalFieldEnforcement: 2 tests
  - Protected fields list, non-canonical field allow-list

**File**: `tests/test_producer_id_normalization.py` (33 tests)
- ProducerIDNormalization: 8 tests
  - Source normalization, calibration tag preservation, whitespace handling
- GetEnsembleProducerID: 4 tests
  - Market-specific ensemble IDs, case-insensitivity
- IsEnsembleProducer: 5 tests
  - Ensemble detection, model vs ensemble distinction
- IsValidProducerInMarket: 7 tests
  - Market-specific validation (ML flexible, SPREAD/TOTAL strict)
- ValidateProducerInHeadsMode: 9 tests
  - Heads mode compliance, "direct" rejection, informative errors

### Regression Testing (14 additional tests)

- `tests/test_elo_heads_equivalence.py`: 11 tests ✓
- `tests/test_bradley_terry_heads.py`: 3 tests ✓

**Total**: 70 tests passing, 0 failures

---

## Files Changed

### Created
1. **src/forecasting/model_support.py** (149 lines)
   - ModelSupport dataclass (frozen)
   - Registry of all models and their capabilities
   - Filtering and lookup utilities

2. **src/forecasting/producer_id.py** (174 lines)
   - Producer ID normalization
   - Market-specific validation
   - Heads mode compliance checking
   - Constants for canonical ensemble IDs

3. **tests/test_phase4_heads_contract.py** (288 lines)
   - 23 comprehensive tests for model support and derivation lockout

4. **tests/test_producer_id_normalization.py** (362 lines)
   - 33 comprehensive tests for producer ID handling

5. **PHASE4_HEADS_ENFORCEMENT.md** (comprehensive implementation details)

6. **PHASE4_README.md** (user-facing reference guide)

### Modified
1. **src/pipelines/projection_engines.py**
   - Added `_CANONICAL_DERIVABLE_FIELDS` set
   - Added `_assert_derivation_locked()` function
   - Enhanced `_rating_projection_engine()` with heads mode guards
   - Updated `_bt_projection_engine()` to use "bradley-terry" instead of "direct"

---

## Design Decisions

### ✓ Did NOT Change
- Ensemble pooling math
- Calibration math
- BETS builder math
- Model fitting/training logic
- Any functional behavior in legacy mode

### ✓ Changed
- Producer ID value for Bradley-Terry ("direct" → "bradley-terry")
- Error handling in projection engine (fail-fast on contract violation)

### ✓ Added
- Model support matrix with extensible registry
- Producer ID normalization and validation utilities
- Derivation lockout guards in projection engine
- 56 comprehensive tests

---

## Integration Ready (Phase 4b)

The following utilities are **complete and ready for schedule pipeline integration**:

1. **Model Support Filtering**
   ```python
   from forecasting.model_support import filter_models_for_market
   
   supported, unsupported = filter_models_for_market(ensemble_models, market="SPREAD")
   if len(supported) < 2:
       raise RuntimeError(f"Ensemble {market} requires >=2 models")
   ```

2. **Producer ID Validation**
   ```python
   from forecasting.producer_id import validate_producer_in_heads_mode
   
   is_valid, error = validate_producer_in_heads_mode(win_prob_source, market)
   if not is_valid:
       raise ValueError(error)
   ```

3. **Ensemble Config Alignment** (validators)
   - Compare configured vs supported vs present models
   - Raise on unsupported references in strict mode
   - Log diagnostics per market

---

## Usage Examples

### Check Model Capabilities
```python
from forecasting.model_support import get_supported_markets

markets = get_supported_markets("elo")
# ['ML', 'SPREAD', 'TOTAL']
```

### Filter Models by Market
```python
from forecasting.model_support import filter_models_for_market

models = ["elo", "bradley-terry", "unknown"]
supported, _ = filter_models_for_market(models, "SPREAD")
# ['elo', 'bradley-terry']
```

### Normalize Producer ID
```python
from forecasting.producer_id import normalize_win_prob_source

source = normalize_win_prob_source("ensemble_ml", "ML")
# 'ensemble_ml_v1'

source = normalize_win_prob_source("elo+calibrated_ml", "ML")
# 'elo+calibrated_ml'
```

### Validate in Heads Mode
```python
from forecasting.producer_id import validate_producer_in_heads_mode

is_valid, error = validate_producer_in_heads_mode("direct", "ML")
# (False, "Producer ID 'direct' uses forbidden 'direct' terminology...")

is_valid, error = validate_producer_in_heads_mode("elo", "SPREAD")
# (False, "Producer ID 'elo' invalid for SPREAD. Must be 'ensemble_spread_v1'...")

is_valid, error = validate_producer_in_heads_mode("ensemble_spread_v1", "SPREAD")
# (True, None)
```

---

## Testing Commands

```bash
# All Phase 4 tests (56 total)
pytest tests/test_phase4_heads_contract.py tests/test_producer_id_normalization.py -v

# Specific category
pytest tests/test_phase4_heads_contract.py::TestModelSupportMatrix -v
pytest tests/test_producer_id_normalization.py::TestValidateProducerInHeadsMode -v

# Regression check (existing heads tests)
pytest tests/test_elo_heads_equivalence.py tests/test_bradley_terry_heads.py -q

# All together (70 tests)
pytest tests/test_phase4_heads_contract.py tests/test_producer_id_normalization.py tests/test_elo_heads_equivalence.py tests/test_bradley_terry_heads.py -q
```

**Result**: ✓ All 70 tests pass

---

## Assumptions

1. **All Phase 1-3 models support all markets** — Bradley-Terry, Elo, TOOR, GSSD, Poisson
2. **Ensemble producers are market-specific** — SPREAD/TOTAL must use ensemble; ML can use model or ensemble
3. **"direct" is legacy terminology** — Replaced with model names in heads mode
4. **Calibration tags are optional** — Appended with `+`; normalization preserves them
5. **Derivation lockout is fail-fast** — RuntimeError on first canonical field derivation attempt

---

## Next Steps (Phase 4b)

1. Integrate model support filtering into schedule pipeline
2. Add ensemble config alignment check with logging
3. Validate producer IDs in schedule export
4. Update validator contracts for heads mode
5. Add integration tests for schedule pipeline

---

## Documentation

- **PHASE4_HEADS_ENFORCEMENT.md** — Detailed implementation notes
- **PHASE4_README.md** — User-facing reference guide
- **docs/model_canonization_playbook.md** — Heads model specification (existing)
- **docs/ENSEMBLE_ARCHITECTURE.md** — Ensemble design (existing)

---

## Verification Checklist

✓ Model support matrix implemented and tested (11 tests)  
✓ Producer ID normalization implemented and tested (8 tests)  
✓ Producer ID validation implemented and tested (9 tests)  
✓ Projection engine derivation lockout implemented (4 tests)  
✓ Canonical field enforcement implemented (2 tests)  
✓ Integration tests for all utilities (6 tests)  
✓ Zero regressions in existing heads tests (14 tests pass)  
✓ Comprehensive documentation provided  
✓ All code follows project conventions  
✓ No changes to ensemble, calibration, or model math  

**Total**: 56 new tests + 14 regression tests = 70 tests, all passing ✓

---

## Contacts & Questions

Phase 4 is complete and ready for production. See PHASE4_README.md for API reference and usage examples.

For Phase 4b integration: use the provided utilities (model_support.filter_models_for_market, producer_id.validate_producer_in_heads_mode) in the schedule pipeline.
