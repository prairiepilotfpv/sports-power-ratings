# Phase 4: Heads Mode Contract Enforcement

## Quick Start

Phase 4 introduces three new modules to enforce heads mode contracts:

1. **Model Support Matrix** (`src/forecasting/model_support.py`)
   - Defines which markets each model supports
   - Enables filtering by capability

2. **Producer ID Normalization** (`src/forecasting/producer_id.py`)
   - Standardizes producer labels (no "direct" terminology)
   - Validates market-specific naming contracts

3. **Projection Engine Derivation Lockout** (enhanced `src/pipelines/projection_engines.py`)
   - Guards against canonical field derivation in heads mode
   - Fail-fast on contract violations

## Key Concepts

### Derivation Lockout (✓ Implemented)

In heads mode, the projection engine **cannot** create or modify:
- `p_home_win`, `model_p_home_win` (ML head)
- `margin_mean`, `margin_sd` (SPREAD head)
- `total_mean`, `total_sd` (TOTAL head)
- `projected_home_score`, `projected_away_score`, `projected_total` (derived scores)

All canonical fields must come **exclusively** from heads. The projection engine becomes validation-only.

**Behavior**:
```python
# Legacy mode (heads_mode=False): projection engine derives normally
# Heads mode (heads_mode=True): projection engine raises RuntimeError on any canonical field derivation
```

### Model Support Matrix (✓ Implemented)

All Phase 1-3 heads models support all markets:

```
Model               ML    SPREAD    TOTAL
─────────────────────────────────────────
bradley-terry      ✓      ✓        ✓
elo                ✓      ✓        ✓
toor               ✓      ✓        ✓
gssd               ✓      ✓        ✓
poisson            ✓      ✓        ✓
```

**Usage**:
```python
from forecasting.model_support import filter_models_for_market

models = ["elo", "bradley-terry", "unknown"]
supported, unsupported = filter_models_for_market(models, "SPREAD")
# supported = ["elo", "bradley-terry"]
# unsupported = ["unknown"]
```

### Producer ID Naming (✓ Implemented)

**Contract**:
- **ML**: `ensemble_ml_v1` OR model name (e.g., `elo`)
- **SPREAD**: `ensemble_spread_v1` ONLY
- **TOTAL**: `ensemble_total_v1` ONLY
- **Tags**: appended with `+` (e.g., `elo+calibrated_ml`)
- **No "direct"**: removed; use model name instead

**Example**:
```python
from forecasting.producer_id import normalize_win_prob_source

# Before
source = "direct"

# After (normalized in heads mode)
source = normalize_win_prob_source("direct", "ML", heads_mode=True)
# Result: "model_direct" (fallback marker; prefer model_name)

# Bradley-Terry projection engine changed
# OLD: win_prob_source = "direct" if model_p is not None else "bt_margin_normal"
# NEW: win_prob_source = "bradley-terry" if model_p is not None else "bt_margin_normal"
```

## Testing

### Test Files

```
tests/test_phase4_heads_contract.py          # 23 tests
  ├── ModelSupportMatrix (11 tests)
  ├── ProjectionEngineDerivationLockout (4 tests)
  ├── ModelSupportIntegration (6 tests)
  └── CanonicalFieldEnforcement (2 tests)

tests/test_producer_id_normalization.py      # 33 tests
  ├── ProducerIDNormalization (8 tests)
  ├── GetEnsembleProducerID (4 tests)
  ├── IsEnsembleProducer (5 tests)
  ├── IsValidProducerInMarket (7 tests)
  └── ValidateProducerInHeadsMode (9 tests)
```

### Running Tests

```bash
# All Phase 4 tests (56 total)
pytest tests/test_phase4_heads_contract.py tests/test_producer_id_normalization.py -v

# Specific test class
pytest tests/test_phase4_heads_contract.py::TestProjectionEngineDerivationLockout -v

# Existing heads tests (regression check)
pytest tests/test_elo_heads_equivalence.py tests/test_bradley_terry_heads.py -q

# All together (70 tests)
pytest tests/test_phase4_heads_contract.py tests/test_producer_id_normalization.py tests/test_elo_heads_equivalence.py tests/test_bradley_terry_heads.py -q
```

**Result**: ✓ 70 tests pass

## Architecture

### Module Relationships

```
src/pipelines/projection_engines.py
  ├─ Uses: forecasting/model_support.py (when filtering)
  └─ Uses: forecasting/producer_id.py (when normalizing)

src/forecasting/model_support.py
  └─ Independent; provides registry

src/forecasting/producer_id.py
  └─ Independent; provides normalization & validation
```

### Integration Points (Ready for Phase 4b)

The following integrations are **designed but not yet implemented** (Phase 4b):

1. **Schedule Pipeline** → Model Support Matrix
   ```python
   # Filter ensemble weights by supported models
   from forecasting.model_support import filter_models_for_market
   
   supported, _ = filter_models_for_market(ensemble_models, market="SPREAD")
   if len(supported) < 2:
       raise RuntimeError(f"{market} ensemble requires >=2 models")
   ```

2. **Schedule Pipeline** → Producer ID Validation
   ```python
   # Validate producer IDs in schedule export
   from forecasting.producer_id import validate_producer_in_heads_mode
   
   is_valid, error = validate_producer_in_heads_mode(source, market)
   if not is_valid:
       raise ValueError(error)
   ```

3. **Validator Contracts** → Producer ID Standardization
   - SPREAD rows must have `ensemble_spread_v1` source
   - TOTAL rows must have `ensemble_total_v1` source
   - ML rows must have `ensemble_ml_v1` or model name (not "direct")

## Files

### Created
- `src/forecasting/model_support.py` (149 lines) — Model support matrix
- `src/forecasting/producer_id.py` (174 lines) — Producer ID normalization
- `tests/test_phase4_heads_contract.py` (288 lines) — Contract tests
- `tests/test_producer_id_normalization.py` (362 lines) — Producer ID tests

### Modified
- `src/pipelines/projection_engines.py` (+37 lines)
  - Added `_CANONICAL_DERIVABLE_FIELDS` set
  - Added `_assert_derivation_locked()` function
  - Added heads mode checks in `_rating_projection_engine()`
  - Changed Bradley-Terry producer ID from "direct" to "bradley-terry"

### Documentation
- `PHASE4_HEADS_ENFORCEMENT.md` (detailed implementation notes)
- This file (`PHASE4_README.md` - this is README structure)

## API Reference

### model_support.py

```python
class ModelSupport:
    supports_ml: bool
    supports_spread: bool
    supports_total: bool
    native_fields: Set[str]
    derived_fields: Set[str]
    
    def supports_market(self, market: str) -> bool: ...
    def all_supported_markets(self) -> list[str]: ...

def get_model_support(model_id: str) -> ModelSupport | None: ...
def get_supported_markets(model_id: str) -> list[str]: ...
def filter_models_for_market(model_ids: list[str], market: str) -> tuple[list[str], list[str]]: ...
```

### producer_id.py

```python
def normalize_win_prob_source(source: str | None, market: Market, *, heads_mode: bool = False) -> str | None: ...
def get_ensemble_producer_id(market: Market) -> str: ...
def is_ensemble_producer(source: str | None) -> bool: ...
def is_valid_producer_in_market(source: str | None, market: Market) -> bool: ...
def validate_producer_in_heads_mode(source: str | None, market: Market) -> tuple[bool, str | None]: ...
```

### projection_engines.py (enhanced)

```python
_CANONICAL_DERIVABLE_FIELDS: set[str] = {...}

def _assert_derivation_locked(field: str, heads_mode: bool, context: ProjectionContext | None = None) -> None: ...
def _rating_projection_engine(...) -> ProjectionOutput: ...  # enhanced with guards
def _bt_projection_engine(...) -> ProjectionOutput: ...      # producer ID changed
```

## Constraints & Decisions

✓ **DO NOT change**:
- Ensemble pooling math
- Calibration math
- BETS builder math
- Fitted model math
- Any model-specific derivation (only guards added)

✓ **Changed**:
- Producer ID terminology ("direct" → model name)
- Projection engine guards (fail-fast on canonical field derivation)
- Bradley-Terry win_prob_source value

✓ **Added**:
- Model support matrix with registry
- Producer ID normalization utilities
- Derivation lockout enforcement
- 56 comprehensive tests

## Future Work (Phase 4b)

1. **Ensemble Weight Filtering**
   - Filter ensemble weights by model support per market
   - Raise if final ensemble has <2 models

2. **Config Validation**
   - Log ensemble config diagnostics per market
   - Raise in strict mode for unsupported models

3. **Validator Updates**
   - Enforce producer ID naming in schedules
   - Prevent "direct" in heads mode exports

4. **Documentation**
   - Update CLI.md with heads mode contract details
   - Add troubleshooting guide for derivation lockout errors

## See Also

- `PHASE4_HEADS_ENFORCEMENT.md` — Detailed implementation notes
- `docs/model_canonization_playbook.md` — Heads model specification
- `docs/ENSEMBLE_ARCHITECTURE.md` — Ensemble design
