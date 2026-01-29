# Heads Migration Plan

**Version**: 1.0  
**Date**: 2026-01-28  
**Status**: PHASE 3 (GSSD heads - IN PROGRESS)

---

## Overview

This document describes the migration from an implicit multi-headed projection engine system to an explicit **Heads Framework** that clearly separates model derivations into composable, testable components.

### Problem Statement

Currently, the codebase has **three overlapping systems** that derive forecast outputs:

1. **Model predict() layer**: Some models (Bradley-Terry, Poisson) emit native outputs; others (Elo, TOOR, GSSD) emit none.
2. **Projection engine layer**: Implicitly derives missing margin/total/probability outputs for 3 of 5 models.
3. **Calibration layer**: Transforms (but does not create) distribution parameters after schedule building.

This creates **multiple implicit "heads"** with unclear responsibilities and difficult-to-debug derivation chains.

### Solution

Implement a **formal Heads Framework**:
- Explicit `Head` protocol defining what each component produces/requires
- Per-model `HeadSequence` registries for transparent composition
- Feature flag (`HEADS_MODE_ENABLED`) for safe rollout
- Projection engine **frozen** (validation-only) when heads mode enabled
- Never run both systems simultaneously (mutual exclusion enforced)

---

## Architecture

### Heads Framework Components

#### 1. Base Protocol (`src/forecasting/heads/base.py`)

```python
class Head(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    
    @abstractmethod
    def produces(self) -> set[str]: ...
    
    @abstractmethod
    def requires(self) -> set[str]: ...
    
    @abstractmethod
    def apply(df: pd.DataFrame, context: dict) -> None: ...
```

**Design**:
- Each head is a **composable unit** that fills specific output columns
- `produces()` and `requires()` are validated for dependency ordering
- `apply()` modifies DataFrame in-place (efficient for bulk operations)
- Logging at DEBUG level tracks which heads were applied and what they filled

#### 2. Registry (`src/forecasting/heads/registry.py`)

```python
_HEAD_REGISTRY: dict[str, callable] = {}

def register_model_heads(model_id: str, factory: callable) -> None:
    _HEAD_REGISTRY[model_id] = factory

def apply_heads(model_id: str, df, context) -> dict | None:
    # Returns applied_heads list or None if no heads registered
```

**Design**:
- Global registry mapping `model_id` → `HeadSequence` factory
- Factory pattern allows lazy instantiation
- `apply_heads()` is the entry point; handles missing registrations gracefully

#### 3. Elo Heads (`src/forecasting/heads/elo_heads.py`)

Four heads, applied in sequence:

1. **EloMarginHead**
   - Produces: `margin_mean`, `margin_sd`
   - Requires: `home_team`, `away_team`, context ratings
   - Logic: Computes margin from rating difference + HFA, applies conditional SD model with guardrails

2. **EloTotalHead**
   - Produces: `total_mean`, `total_sd`
   - Requires: `home_team`, `away_team`
   - Logic: Uses model formula (if available) → scoring averages → base_total fallback

3. **EloWinProbHead**
   - Produces: `p_home_win`, `projected_win_prob`, `model_p_home_win`, `logistic_home_win_prob`, `normal_p_home_win`, `win_prob_source`, `margin_dist_assumption`
   - Requires: `margin_mean`, `margin_sd`
   - Logic: Computes normal CDF (what margin implies) + logistic (what Elo uses); logistic is primary for Elo

4. **EloProjectedScoresHead**
   - Produces: `projected_home_score`, `projected_away_score`, `projected_total`
   - Requires: `margin_mean`, `total_mean`
   - Logic: Arithmetic derivation from margin and total

**Key Property**: These heads **reproduce legacy projection engine outputs exactly** within numerical tolerance (0.01).

---

## Feature Flag & Rollout

### Configuration

**File**: `src/config.py`

```python
# Formal heads system feature flag.
# When enabled, uses explicit heads framework for model derivations (Elo, TOOR, GSSD).
# When disabled (default), uses legacy projection engine derivation.
HEADS_MODE_ENABLED = False
```

### Mutual Exclusion

**In `src/pipelines/projection_engines.py`:**

```python
def get_projection_engine(model: Any, heads_mode: bool | None = None) -> ProjectionEngine:
    if heads_mode is None:
        heads_mode = HEADS_MODE_ENABLED
    
    if heads_mode:
        # Check if heads available for this model
        if get_model_heads(model_id) is not None:
            return _validation_only_engine  # No derivation; heads do the work
    
    return _ENGINES.get(model_key, _ENGINES[_DEFAULT_ENGINE_KEY])  # Legacy engine
```

**Effect**:
- When `heads_mode=True` and Elo heads exist → projection engine returns no-op values
- When `heads_mode=False` (default) → legacy projection engine derivations used
- **Cannot run both simultaneously** (prevents double-derivation bugs)

---

## Phase Plan

### PHASE 1: Elo Heads (COMPLETED ✅)

**Goals**:
- ✅ Implement heads framework (base protocol + registry)
- ✅ Implement Elo heads (4 components)
- ✅ Add feature flag and mutual exclusion
- ✅ Add equivalence tests (synthetic matchup scenarios)
- ✅ Document design and phase plan

**Deliverables**:
- `src/forecasting/heads/{base,registry,elo_heads,__init__}.py`
- `tests/test_elo_heads_equivalence.py` (11 tests, all passing)
- `docs/HEADS_MIGRATION_PLAN.md` (this document)
- `HEADS_MODE_ENABLED` flag in config

**Testing**:
```bash
pytest -xvs tests/test_elo_heads_equivalence.py
```

**Key Test Cases**:
- Margin/total/win_prob derivation correctness
- Equivalence to legacy projection engine (within tolerance)
- Missing ratings handled gracefully
- Conditional SD model integration
- Validation errors for missing dependencies
- Heads mode flag in config
- Projection engine no-op in heads mode

### PHASE 2: TOOR Heads (COMPLETED ✅)

**Scope**: Migrate TOOR derivations to explicit heads
- ✅ Implement `ToorMarginHead`, `ToorTotalHead`, `ToorWinProbHead`, `ToorProjectedScoresHead`
- ✅ Update registry: `register_model_heads("toor", create_toor_head_sequence)`
- ✅ Add equivalence tests mirroring Elo tests
- ✅ Verify `_toor_projection_engine` logic is reproduced exactly
- ✅ Strict per-model/per-market contract validation (Option B alignment)

**Deliverables**:
- `src/forecasting/heads/toor_heads.py`
- `tests/test_toor_heads_equivalence.py` (10+ tests, all passing)
- `tests/test_interface_contract.py` updated for TOOR heads

**Acceptance Criteria**:
- ✅ Toor heads produce outputs within 0.01 tolerance of legacy engine
- ✅ All tests pass
- ✅ No behavioral change in legacy mode (flag disabled)
- ✅ Contract validation enforces Option B universe alignment

### PHASE 3: GSSD Heads (IN PROGRESS)

**Scope**: Migrate GSSD derivations to explicit heads + eliminate NaN collapse
- ✅ Implement `GssdMarginHead`, `GssdTotalHead`, `GssdWinProbHead`, `GssdScoresHead`
- ✅ Update registry: `register_model_heads("gssd", create_gssd_head_sequence)`
- ✅ Add equivalence tests mirroring TOOR tests
- ✅ Strict guards: raise RuntimeError if margin_sd or p_home_win non-finite (never silent NaN)
- ✅ Account for per-team scoring stats (pfh, pah, pfa, paa) in context
- ✅ Support optional conditional SD model for margin prediction
- ✅ Ensure GSSD team_stats + calibration coefficients available in projection context

**Key Changes from Phase 2**:
- **Team Stats Context**: `team_stats` dict added to projection context (from model._gssd._team_stats)
- **GSSD Coefficients**: All calibration coefficients (intercept, beta_*, home_advantage_points, error_term) added to context
- **Total Computation**: Derives total from team stats: `(pfh + paa + pfa + pah) / 2`, not league average
- **NaN Elimination**: Strict invariants — if margin_sd or p_home_win become non-finite, raise RuntimeError immediately (no silent fallback)
- **Logistic Win Prob**: GSSD uses logistic curve (like TOOR), not normal CDF

**Deliverables**:
- `src/forecasting/heads/gssd_heads.py`
- `tests/test_gssd_heads_equivalence.py` (10+ tests covering equivalence, NaN prevention, contract validation, Option B)
- Projection context enhancement in `src/forecasting/forecast_service.py` (GSSD-specific params)

**Acceptance Criteria**:
- ✅ GSSD heads produce outputs equivalent to legacy `GSSDModel.project_matchup()`
- ✅ All tests pass (equivalence + NaN prevention + contract validation + Option B)
- ✅ No behavioral change in legacy mode
- ✅ Strict guards prevent NaN collapse (fail loudly instead)
- ✅ Historical SD/p_home_win NaN failure scenario is eliminated at the source

---

## Design Constraints

### DO NOT

- ❌ Change ensemble math (weights, aggregation logic stays the same)
- ❌ Change calibration math (transformations are independent of heads)
- ❌ Add betting/EV logic to heads (heads derive forecasts only)
- ❌ Invent new statistical models (just make implicit explicit)
- ❌ Break backward compatibility in legacy mode (heads_mode=False)
- ❌ Refactor unrelated modules
- ❌ Run both derivation systems simultaneously

### DO

- ✅ Keep heads framework minimal and composable
- ✅ Ensure exact numerical equivalence (tolerance ≤ 0.01)
- ✅ Add comprehensive tests for each phase
- ✅ Document assumptions in code comments
- ✅ Use feature flag for safe rollout
- ✅ Enforce mutual exclusion (validation-only engine in heads mode)
- ✅ Log all head applications at DEBUG level

---

## Integration Points

### Current Callers of Projection Engine

**File**: `src/pipelines/schedule.py` → `build_forecasts_df()`

**Before Phase 1**:
```python
projection_engine = get_projection_engine(model)
output = projection_engine(home, away, model, context)
```

**After Phase 1 (when heads_mode=True)**:
- If Elo + heads registered:
  ```python
  # Heads apply first (outside schedule.py, in forecasting service)
  result = apply_heads("elo", df, context)
  
  # Projection engine is still called but returns no-ops
  projection_engine = get_projection_engine(model, heads_mode=True)
  output = projection_engine(home, away, model, context)  # Returns all None
  ```
- DataFrame already has all canonical fields from heads; projection engine output ignored

**No changes required to schedule.py** in this phase (heads application managed elsewhere).

### Calibration

**File**: `src/pipelines/schedule.py` → `_apply_calibration_to_schedule_df()`

**Status**: UNCHANGED
- Calibration operates on DataFrames after heads/projection engine have populated fields
- Calibration transforms (doesn't create) outputs
- Works identically whether fields came from heads or projection engine

### Bradley-Terry & Poisson

**Status**: NOT AFFECTED IN PHASE 1
- Bradley-Terry: Already uses native `project_matchup()` method
- Poisson: Already emits all outputs natively
- No heads needed; projection engine currently bypasses them correctly

---

## Testing Strategy

### Unit Tests (`tests/test_elo_heads_equivalence.py`)

| Test | Purpose | Status |
|------|---------|--------|
| `test_elo_heads_margin_derivation` | Margin computation | ✅ |
| `test_elo_heads_total_derivation` | Total computation | ✅ |
| `test_elo_heads_win_prob_derivation_with_logistic` | Win prob logistic | ✅ |
| `test_elo_heads_projected_scores` | Projected score arithmetic | ✅ |
| `test_elo_heads_missing_rating` | Error handling | ✅ |
| `test_elo_heads_with_conditional_sd_model` | SD model integration | ✅ |
| `test_elo_heads_equivalence_to_projection_engine` | **Equivalence** | ✅ |
| `test_heads_mode_disallows_projection_derivation` | Mutual exclusion | ✅ |
| `test_heads_mode_flag_in_config` | Config flag | ✅ |
| `test_head_sequence_missing_dependency` | Validation | ✅ |
| `test_head_factory_produces_sequence` | Factory pattern | ✅ |

### Integration Tests

**TODO** (phase 2):
- Test full schedule pipeline with heads mode enabled
- Verify BETS sheet outputs match legacy mode
- Verify ensemble aggregation works with head-derived outputs
- Test with real game data (not synthetic)

---

## Rollout Timeline

### Pre-Rollout Checklist
- ✅ Phase 1 implementation complete
- ✅ Phase 1 tests all passing (11/11)
- ⏳ Integration tests pass with real data
- ⏳ Code review + approval
- ⏳ Benchmark: legacy vs heads mode performance (optional)

### Rollout Steps
1. **Week 1**: Code review, finalize docs
2. **Week 2**: Integration testing, flag disabled (default)
3. **Week 3**: Enable flag selectively (testing environment)
4. **Week 4**: Enable flag in production, monitor
5. **Later**: Phase 2 (TOOR heads), Phase 3 (GSSD heads)

---

## Known Limitations & Future Work

### Phase 1 Scope
- Only Elo heads implemented; projection engine still used for TOOR/GSSD
- Feature flag manual (not CLI-wired in this phase)
- No performance benchmarking vs projection engine

### Phase 2 Dependencies
- Must finalize Elo heads before starting TOOR (registry interference possible)
- Requires review of `_toor_projection_engine` logic (different margin formula?)
- May need separate conditional SD model for TOOR

### Opportunities
- Once all models migrated, could remove `projection_engines.py` entirely
- Could add real-time streaming derivations (heads are DataFrame-agnostic)
- Could partition heads across processes (pure functions)

---

## References

- **Audit**: `docs/CURRENT_HEAD_BEHAVIOR_AUDIT.md`
- **Framework**: `src/forecasting/heads/`
- **Config**: `src/config.py` (HEADS_MODE_ENABLED)
- **Projection Engine**: `src/pipelines/projection_engines.py` (get_projection_engine, _validation_only_engine)
- **Tests**: `tests/test_elo_heads_equivalence.py`
