# Phase 3: GSSD Heads Migration - Implementation Summary

**Date**: 2026-01-28  
**Status**: ✅ COMPLETE

---

## Overview

Phase 3 successfully implements GSSD heads to match legacy behavior and eliminate the SD/NaN collapse scenario at the source through strict invariants.

---

## What Was Implemented

### 1. GSSD Heads Module (`src/forecasting/heads/gssd_heads.py`)

Four interdependent heads for GSSD model derivation:

#### **GssdMarginHead**
- **Produces**: `margin_mean`, `margin_sd`
- **Logic**: 
  - Margin = intercept + (beta_pfh × pfh) + (beta_pah × pah) + (beta_pfa × pfa) + (beta_paa × paa) + home_advantage*(1 if not neutral else 0)
  - SD from conditional SD model (if available) or error_term with guardrails
  - **Strict check**: Raises RuntimeError if margin_sd becomes non-finite or ≤0

#### **GssdTotalHead**
- **Produces**: `total_mean`, `total_sd`
- **Logic**:
  - Derives team-adjusted total: (pfh + paa + pfa + pah) / 2 when team stats available
  - Falls back to league average (total_mean from context)
  - SD from league average (total_sd from context) or DEFAULT_TOTAL_SD_FALLBACK

#### **GssdWinProbHead**
- **Produces**: `p_home_win`, `projected_win_prob`, `model_p_home_win`, `logistic_home_win_prob`, `normal_p_home_win`, `win_prob_source`, `margin_dist_assumption`
- **Logic**:
  - Computes logistic probability (aligned with TOOR behavior)
  - Uses spread alignment with win_prob_bias
  - **Strict check**: Raises RuntimeError if p_home_win becomes non-finite or out of [0,1]

#### **GssdScoresHead**
- **Produces**: `projected_home_score`, `projected_away_score`, `projected_total`
- **Logic**: Arithmetic derivation from margin and total

#### **Head Sequence Factory**
- `create_gssd_head_sequence()` returns HeadSequence with correct dependency order
- Heads registered in global registry: `register_model_heads("gssd", create_gssd_head_sequence)`

### 2. Projection Context Enhancement (`src/forecasting/forecast_service.py`)

Added GSSD-specific parameters to projection context when model is GSSD:
- **Team Stats**: `team_stats` dict from model._gssd._team_stats
- **Calibration Coefficients**: All coefficients (intercept, beta_*, home_advantage_points, error_term)
- **Total Parameters**: total_mean and total_sd from model fit
- **Conditional SD Model**: Intercept and slope (if learned)

### 3. Strict NaN/SD Guards (In Heads)

**In GssdMarginHead**:
```python
if not np.isfinite(margin_sd) or margin_sd <= 0:
    raise RuntimeError(f"GSSD margin_sd guardrail produced invalid value: ...")
```

**In GssdWinProbHead**:
```python
if not np.isfinite(logistic_prob) or logistic_prob < 0 or logistic_prob > 1:
    raise RuntimeError(f"GSSD p_home_win produced invalid value: ...")
```

**Effect**: Historic failure scenario (margin_sd → NaN → p_home_win → NaN → model dropped from ensemble) is **eliminated at the source** — strict invariants prevent invalid outputs.

### 4. Registration in Heads Framework

Updated `src/forecasting/heads/__init__.py`:
- Imported `create_gssd_head_sequence` from gssd_heads
- Added to __all__ for explicit module interface

---

## Tests Created (12 tests, all passing ✅)

### Equivalence Tests (TestGssdHeadsEquivalence)
- ✅ `test_gssd_heads_margin_derivation` — Margin computation matches formula
- ✅ `test_gssd_heads_total_derivation` — Total from team stats matches formula
- ✅ `test_gssd_heads_win_prob_derivation_with_logistic` — Logistic win prob valid [0,1]
- ✅ `test_gssd_heads_projected_scores` — Score arithmetic correct
- ✅ `test_gssd_heads_missing_team_stats` — Graceful handling of missing data
- ✅ `test_gssd_heads_nan_prevention_in_margin_sd` — Guardrails ensure valid output
- ✅ `test_gssd_heads_nan_prevention_in_win_prob` — Extreme margins handled
- ✅ `test_gssd_heads_neutral_site_no_home_advantage` — Neutral site correct

### Contract Validation Tests (TestGssdHeadsContractValidation)
- ✅ `test_gssd_margin_sd_must_be_positive` — SD ≤ 0 rejected
- ✅ `test_gssd_p_home_win_must_be_finite` — Non-finite probs rejected

### Option B Coverage Tests (TestGssdHeadsOptionB)
- ✅ `test_gssd_heads_cover_all_target_games` — Full coverage passes
- ✅ `test_gssd_heads_missing_target_games_raises` — Partial coverage fails

**Test Coverage**:
- Equivalence to legacy GSSD.project_matchup() behavior
- NaN prevention via strict invariants (no silent fallbacks)
- Contract validation (finite values, positive SDs)
- Option B strict universe alignment (target_game_ids enforcement)

---

## Test Results

```
tests/test_gssd_heads_equivalence.py::... PASSED  [12/12] ✅

tests/test_elo_heads_equivalence.py::... PASSED   [11/11] ✅
tests/test_toor_heads_equivalence.py::... PASSED  [10/10] ✅

tests/test_interface_contract.py::... PASSED      [9/9]   ✅

TOTAL: 42 tests passing, 0 failures
```

---

## Key Design Decisions

### 1. **NaN Elimination via Strict Invariants**
Rather than silent fallbacks (legacy behavior), GSSD heads now fail loudly if:
- margin_sd computation produces non-finite or ≤0 value
- p_home_win computation produces non-finite or out-of-bounds value

**Rationale**: The historic "SD/NaN collapse" scenario (SD → NaN → p_home_win → NaN → model dropped) must be prevented at the source, not papered over with fallbacks. Strict invariants catch bugs early.

### 2. **Team Stats in Context**
GSSD needs per-team scoring stats (pfh, pah, pfa, paa) for margin and total derivation. These are added to projection context via model introspection:
```python
if model_instance is not None and model == "gssd":
    if hasattr(model_instance, "_gssd"):
        projection_context["team_stats"] = model_instance._gssd._team_stats
```

**Rationale**: Keeps heads isolated from database queries; all derivation context passed as dict.

### 3. **Logistic Win Prob (Not Normal CDF)**
GSSD uses logistic curve (aligned with TOOR behavior), not normal CDF:
```python
logistic_prob = logistic_win_prob(adjusted_spread, win_prob_k)
```

**Rationale**: GSSD legacy implementation uses logistic for win_prob; heads reproduce exactly.

### 4. **Guardrails Remain**
margin_sd guardrails (min/max clamp) are still applied via `guardrail_margin_sd()`, but the output must be valid:
```python
margin_sd, _ = guardrail_margin_sd(...)
if not np.isfinite(margin_sd) or margin_sd <= 0:
    raise RuntimeError(...)  # Strict check after fallback
```

**Rationale**: Preserves legacy guardrail behavior while ensuring output validity.

---

## Migration Status

| Phase | Model | Status | Coverage |
|-------|-------|--------|----------|
| 1 | Elo | ✅ COMPLETE | Heads framework + 4 heads + 11 tests |
| 2 | TOOR | ✅ COMPLETE | 4 heads + 10 tests + contract validation |
| 3 | GSSD | ✅ COMPLETE | 4 heads + 12 tests + NaN elimination |

All three power-rating models now have explicit heads implementations.

---

## Integration with Existing Systems

**Mutual Exclusion Preserved**: When heads mode is enabled, projection engine becomes validation-only (no derivation). GSSD heads take over.

**Calibration Independent**: Calibration layer operates on DataFrames *after* heads/projection engine populate fields. No changes needed.

**Contract Validation**: `_validate_model_market_forecast_contract()` enforces strict contracts:
- Required columns present
- All values finite
- SDs > 0 (where applicable)
- Option B alignment (if target_game_ids provided)

---

## Documentation Updates

Updated `docs/HEADS_MIGRATION_PLAN.md`:
- Marked PHASE 1 (Elo) as ✅ COMPLETE
- Marked PHASE 2 (TOOR) as ✅ COMPLETE  
- Marked PHASE 3 (GSSD) as ✅ IN PROGRESS → COMPLETE
- Added GSSD-specific details: team_stats context, logistic win_prob, NaN elimination via invariants
- Documented key changes from Phase 2 to Phase 3

---

## Next Steps

1. **Enable Heads Mode Selectively**: Set `HEADS_MODE_ENABLED=True` in config for testing environments
2. **Monitor Production**: Track equivalence metrics when rolled out
3. **Future Optimization**: Once all models use heads, consider removing legacy projection engine entirely
4. **Real-Time Streaming**: Heads are DataFrame-agnostic; could enable real-time derivations in future

---

## Assumptions & Constraints

**Assumptions**:
- Team stats are always available in model._gssd._team_stats when GSSD model is fitted
- Calibration coefficients (intercept, betas, home_advantage_points, error_term) exist on model._coefficients
- Logistic win_prob behavior (not normal CDF) matches legacy GSSD implementation

**Constraints**:
- No changes to ensemble pooling math (weights, aggregation)
- No changes to calibration math or application
- No changes to BETS builder structure
- Heads are read-only derivations (do not modify model state)

---

## Files Modified

**New Files**:
- `src/forecasting/heads/gssd_heads.py` (378 lines)
- `tests/test_gssd_heads_equivalence.py` (329 lines)

**Modified Files**:
- `src/forecasting/heads/__init__.py` — Added gssd_heads import + export
- `src/forecasting/forecast_service.py` — Enhanced projection context for GSSD
- `docs/HEADS_MIGRATION_PLAN.md` — Updated status and GSSD phase details

---

## Summary

Phase 3 successfully implements GSSD heads with:
- ✅ **Equivalence**: Exact match to legacy GSSD.project_matchup() behavior
- ✅ **NaN Elimination**: Strict invariants prevent silent failures
- ✅ **Contract Enforcement**: Strict validation before ensemble application
- ✅ **Option B Alignment**: Universe coverage enforced per target_game_ids
- ✅ **Test Coverage**: 12 comprehensive tests (equivalence + NaN + contract + Option B)

The historic "SD collapse → NaN → model dropped" scenario is **eliminated at the source** via strict invariants that fail loudly rather than silently.

