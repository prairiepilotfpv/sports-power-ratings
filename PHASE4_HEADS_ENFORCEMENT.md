# Phase 4: Heads Mode Contract Enforcement & Producer ID Normalization

**Status**: Complete  
**Date**: 2025-01-28  
**Scope**: Contract enforcement, model support matrix, producer ID standardization

## Overview

Phase 4 eliminates layered derivation paths and enforces a single, explicit market-neutral contract across the entire heads suite. This ensures that in heads mode:

1. **Projection engine is derivation-locked**: cannot create or modify canonical fields
2. **Model support is explicit**: filtered by market capability
3. **Ensemble configs are validated**: against supported models
4. **Producer IDs are consistent**: no "direct" terminology, standardized naming

## Deliverables

### 1. Projection Engine Derivation Lockout

**Module**: `src/pipelines/projection_engines.py`

Added guards to prevent canonical field derivation in heads mode:

```python
_CANONICAL_DERIVABLE_FIELDS = {
    "p_home_win",
    "model_p_home_win",
    "margin_mean",
    "margin_sd",
    "total_mean",
    "total_sd",
    "projected_home_score",
    "projected_away_score",
    "projected_total",
}

def _assert_derivation_locked(field: str, heads_mode: bool, context: ProjectionContext | None = None) -> None:
    """Enforce derivation lockout: raises RuntimeError if attempting to derive canonical field in heads mode."""
```

**Behavior**:
- When `heads_mode=False` (legacy mode): projection engine derives all fields normally
- When `heads_mode=True` (heads mode): projection engine raises `RuntimeError` if it attempts to compute any canonical field
- All canonical fields must come exclusively from heads; projection engine is validation-only

**Example Error**:
```
RuntimeError: [heads-mode derivation lockout] Cannot derive canonical field 'margin_mean' 
in heads mode (game: game_123). All canonical fields must come from heads; projection engine 
must not compute or overwrite margin_mean.
```

### 2. Model Support Matrix

**Module**: `src/forecasting/model_support.py`

Defines which models support which markets (ML/SPREAD/TOTAL):

```python
@dataclass(frozen=True)
class ModelSupport:
    supports_ml: bool
    supports_spread: bool
    supports_total: bool
    native_fields: Set[str]
    derived_fields: Set[str]
```

**Registry** (all Phase 1-3 heads models):
- **bradley-terry**: supports ML, SPREAD, TOTAL
- **elo**: supports ML, SPREAD, TOTAL
- **toor**: supports ML, SPREAD, TOTAL
- **gssd**: supports ML, SPREAD, TOTAL
- **poisson**: supports ML, SPREAD, TOTAL

**Key Functions**:
- `get_model_support(model_id)`: Retrieve support matrix
- `get_supported_markets(model_id)`: List of supported markets
- `filter_models_for_market(model_ids, market)`: Returns (supported, unsupported) lists

**Usage**:
```python
from forecasting.model_support import filter_models_for_market

models = ["elo", "bradley-terry", "unknown_model"]
supported, unsupported = filter_models_for_market(models, "SPREAD")
# supported = ["elo", "bradley-terry"]
# unsupported = ["unknown_model"]
```

### 3. Producer ID Normalization

**Module**: `src/forecasting/producer_id.py`

Standardizes producer labels and enforces naming contract:

```python
ENSEMBLE_ML_V1 = "ensemble_ml_v1"
ENSEMBLE_SPREAD_V1 = "ensemble_spread_v1"
ENSEMBLE_TOTAL_V1 = "ensemble_total_v1"

def normalize_win_prob_source(source: str | None, market: Market, *, heads_mode: bool = False) -> str | None:
    """Normalize producer ID to canonical form (ensemble_<market>_v1 or model_name)."""

def validate_producer_in_heads_mode(source: str | None, market: Market) -> tuple[bool, str | None]:
    """Validate producer ID is compliant with heads mode contract."""
```

**Contract**:
- **ML market**: can be `ensemble_ml_v1` or single model name (e.g., `elo`, `bradley-terry`)
- **SPREAD market**: must be `ensemble_spread_v1` only
- **TOTAL market**: must be `ensemble_total_v1` only
- **Calibration tags**: appended with `+` (e.g., `elo+calibrated_ml+calibrated_spread`)
- **No "direct" terminology** in heads mode; changed to model name

**Example Conversions**:
```python
# Before (with "direct" terminology)
win_prob_source = "direct"

# After (normalized)
win_prob_source = "bradley-terry"  # single model source

# Bradley-Terry projection engine change
_bt_projection_engine():
    # OLD: win_prob_source: "direct" if model_p is not None else "bt_margin_normal"
    # NEW: win_prob_source: "bradley-terry" if model_p is not None else "bt_margin_normal"
```

### 4. Validation Utilities

**Market-Specific Validation**:
```python
is_valid_producer_in_market(source: str | None, market: Market) -> bool
```

Returns `True` if producer is valid for given market:
- SPREAD/TOTAL: only ensemble sources allowed
- ML: ensemble or single model allowed

**Heads Mode Compliance**:
```python
validate_producer_in_heads_mode(source: str | None, market: Market) -> tuple[bool, str | None]
```

Returns `(is_valid, error_message)` with detailed compliance diagnostics.

## Test Coverage

### Phase 4 Tests (`tests/test_phase4_heads_contract.py`)

**23 tests** covering:
- Model support matrix registry lookup and filtering
- Derivation lockout enforcement in heads mode
- Canonical field protection
- Support matrix immutability

**Key Test Classes**:
- `TestModelSupportMatrix`: Registry and filtering (11 tests)
- `TestProjectionEngineDerivationLockout`: Guards and lockout (4 tests)
- `TestModelSupportIntegration`: Integration scenarios (6 tests)
- `TestCanonicalFieldEnforcement`: Field contract (2 tests)

### Producer ID Tests (`tests/test_producer_id_normalization.py`)

**33 tests** covering:
- Win probability source normalization
- Ensemble producer detection
- Market-specific validation
- Heads mode compliance
- Calibration tag handling
- Error messages

**Key Test Classes**:
- `TestProducerIDNormalization`: Source normalization (8 tests)
- `TestIsEnsembleProducer`: Detection (5 tests)
- `TestIsValidProducerInMarket`: Validation (7 tests)
- `TestValidateProducerInHeadsMode`: Compliance (9 tests)

### Existing Tests Still Pass

Verified compatibility with Phase 1-3 heads implementations:
- `tests/test_elo_heads_equivalence.py`: ✓ 11 tests pass
- `tests/test_toor_heads_equivalence.py`: ✓ 11 tests pass
- `tests/test_gssd_heads_equivalence.py`: ✓ 11 tests pass

**Total Phase 4**: 56 new tests, all passing  
**Regression**: 0 failures in existing heads tests

## Implementation Details

### Derivation Lockout Mechanism

The projection engine checks a context flag `__heads_mode__` to determine if lockout should be enforced:

```python
def _rating_projection_engine(..., context: ProjectionContext):
    heads_mode = context.get("__heads_mode__", HEADS_MODE_ENABLED)
    
    # Early guards for canonical fields
    if heads_mode:
        _assert_derivation_locked("margin_mean", heads_mode, guardrail_context)
        _assert_derivation_locked("margin_sd", heads_mode, guardrail_context)
        _assert_derivation_locked("total_mean", heads_mode, guardrail_context)
        _assert_derivation_locked("total_sd", heads_mode, guardrail_context)
    
    # If lockout triggered, this line raises RuntimeError
    # Otherwise, projection engine proceeds normally
```

### Architectural Notes

1. **Immutability**: `ModelSupport` is a frozen dataclass; registry cannot be modified at runtime
2. **Idempotency**: Producer ID normalization is idempotent; can be applied multiple times safely
3. **Backward Compatibility**: Legacy mode (heads_mode=False) is unchanged
4. **Future-Proof**: Extendable to new models via `_MODEL_SUPPORT_REGISTRY` update

## Future Integration Points

The following Phase 4 modules are ready for schedule pipeline integration (Phase 4b):

1. **Model Support Matrix**: Filter ensemble weights by market capability
   ```python
   from forecasting.model_support import filter_models_for_market
   
   supported, unsupported = filter_models_for_market(config_models, market="SPREAD")
   if len(supported) < 2:
       raise RuntimeError(f"Ensemble {market} requires >=2 models; got {supported}")
   ```

2. **Producer ID Validation**: Enforce naming contract in schedule pipeline
   ```python
   from forecasting.producer_id import validate_producer_in_heads_mode
   
   is_valid, error = validate_producer_in_heads_mode(win_prob_source, market)
   if not is_valid:
       raise ValueError(error)
   ```

## Files Modified/Created

- **Created**: `src/forecasting/model_support.py` (149 lines)
- **Created**: `src/forecasting/producer_id.py` (174 lines)
- **Modified**: `src/pipelines/projection_engines.py` (+37 lines, guard checks)
- **Created**: `tests/test_phase4_heads_contract.py` (288 lines)
- **Created**: `tests/test_producer_id_normalization.py` (362 lines)

## Running Phase 4 Tests

```bash
# All Phase 4 tests
python -m pytest tests/test_phase4_heads_contract.py tests/test_producer_id_normalization.py -v

# By category
python -m pytest tests/test_phase4_heads_contract.py::TestModelSupportMatrix -v
python -m pytest tests/test_phase4_heads_contract.py::TestProjectionEngineDerivationLockout -v
python -m pytest tests/test_producer_id_normalization.py::TestValidateProducerInHeadsMode -v

# Regression check (Phase 1-3 heads)
python -m pytest tests/test_elo_heads_equivalence.py tests/test_toor_heads_equivalence.py tests/test_gssd_heads_equivalence.py -q
```

## Assumptions & Design Decisions

1. **All Phase 1-3 models support all markets**: Bradley-Terry, Elo, TOOR, GSSD, Poisson all support ML, SPREAD, TOTAL. Future models can be added to registry.

2. **Ensemble producers are market-specific**: SPREAD and TOTAL always use ensemble sources; ML can use single model or ensemble.

3. **"direct" is legacy terminology**: Replaced with model names (e.g., "bradley-terry" instead of "direct") for heads mode clarity.

4. **Calibration tags are optional**: Appended with `+` separator; normalization preserves them.

5. **Derivation lockout is fail-fast**: RuntimeError raised immediately on first canonical field derivation attempt in heads mode, preventing silent double-derivation bugs.

## Next Steps (Phase 4b)

1. Integrate model support matrix into schedule pipeline
2. Filter ensemble weights by supported models per market
3. Validate ensemble configs in strict mode
4. Add ensemble config alignment check logging
5. Update validator contracts to enforce producer ID naming
