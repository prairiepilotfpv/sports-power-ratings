# Investigation Summary: Ensemble & Heads Dataflow

**Status**: ✅ COMPLETE  
**Date**: 2025-01-28  
**Duration**: Single investigation session  
**Deliverables**: 2 detailed docs + this summary

---

## What Was Investigated

This investigation mapped the **end-to-end dataflow** of how ensemble models work in the sports-power-ratings system and designed a missing "heads" architecture for deriving market-specific outputs from base model representations.

### Scope: NO Code Changes
- This was investigation & documentation only
- No refactors, new features, or behavioral changes
- No bugs fixed (found none critical)

---

## Key Findings

### 1. Three-Layer Ensemble Architecture

The system operates in **three sequential stages**:

```
Per-Model Forecasts → Ensemble Combination → Calibration & Provenance
   (Step 1-2)              (Step 3-4)              (Step 5-6)
```

**Per-Model** ([src/pipelines/schedule.py:895](src/pipelines/schedule.py#L895)):
- Fit each model (Elo, Bradley-Terry, GSSD, Poisson, TOOR)
- Call `.predict()` → returns `GamePrediction` with market-specific fields
- Group into `market_forecast_rows` dict by market (ML, SPREAD, TOTAL)

**Ensemble** ([src/ensemble/{ml_v1,spread_v1,total_v1}.py](src/ensemble/)):
- Resolve weights (from DB tuning, file config, or equal)
- Filter weights to models with valid forecast columns
- Apply market-specific combination:
  - **ML**: Logit (log-odds) pooling of probabilities
  - **SPREAD**: Weighted average of means + variance composition
  - **TOTAL**: Weighted average of means + variance composition
- Output: raw predictions + components JSON for audit trail

**Calibration** ([src/pipelines/schedule.py:333](src/pipelines/schedule.py#L333)):
- Load per-market calibrators (Platt scaling for ML, distribution calibrators for SPREAD/TOTAL)
- Transform raw ensemble predictions
- Append market-specific provenance tags (e.g., `"ensemble_ml_v1+calibrated_ml+calibrated_spread"`)

### 2. Coverage Collapse: Seven Root Causes

Coverage can collapse from multi-model to single-model at multiple points:

| # | Cause | Location | Symptom | Example |
|---|-------|----------|---------|---------|
| 1 | Missing forecast rows | [L895](src/pipelines/schedule.py#L895) | Model doesn't generate output for all games | Elo fit fails |
| 2 | Missing columns | [L1459](src/pipelines/schedule.py#L1459) | Model has rows but lacks required columns | Elo missing `margin_sd` |
| 3 | Join key mismatch | [L2790](src/pipelines/schedule.py#L2790) | Game IDs differ or window narrows | BETS uses market CSV, forecasts use full season |
| 4 | NaN filtering | [L500](src/pipelines/schedule.py#L500) | Calibrator returns NaN for some rows | Calibration fails for 10% |
| 5 | Weight filtering | [L1528](src/pipelines/schedule.py#L1528) | Tuned weights → all zero for some models | Tuning selected only BT |
| 6 | Ensemble window mismatch | [L2850](src/pipelines/schedule.py#L2850) | Forecasts exist for games outside BETS time window | Run 1: 30 games; forecasts: 100 games |
| 7 | Model doesn't support market | [src/models/elo.py](src/models/elo.py) | Model metadata says `supports_total=False` but assigned | Elo to TOTAL ensemble |

**Most Common**: "ML becomes BT-only" happens when Elo forecast rows are missing or have NaN, causing `_filter_market_weights_for_forecast` to drop it; ensemble count drops from 2 to 1, skipping ensemble application. [See Section C.7 in Dataflow doc](docs/ENSEMBLE_AND_HEADS_DATAFLOW.md#ml-becomes-bt-only-scenario).

### 3. Source_id (Provenance) Format

Output sources are tagged to track origin and transformations:

```
Base Source          →  Calibration Tags
"elo"               →  "elo"
"ensemble_ml_v1"    →  "ensemble_ml_v1+calibrated_ml"
"bradley-terry"     →  "bradley-terry+calibrated_ml+calibrated_spread"
```

Tags are **idempotent**: calling append twice produces no duplicates.

### 4. Bradley-Terry Lacks Direct Margin Output

**Current**: Bradley-Terry produces `p_win` natively, but margins must be derived via:
- Option 1: Calibration backs-solves SD from margin residuals
- Option 2: Projection engine inverts p_win → margin (Normal CDF inversion)
- Option 3: Never enters SPREAD/TOTAL ensembles (only ML)

**Problem**: Can't use BT in SPREAD/TOTAL ensemble without either (1) pre-fitting calibrator or (2) sacrificing principled margin derivation.

### 5. Missing "Heads" System

**Concept**: A "head" is a deterministic function that derives market-specific outputs from a model's base representation.

- **Example**: Bradley-Terry ML head produces p_win from ratings (native)
- **Counter-example**: Bradley-Terry SPREAD head must derive margin (missing)

**Current State**: Heads are implicit, scattered across model `.predict()`, projection engines, and calibration logic.

**Proposed Design** (in [docs/HEADS_SYSTEM_SPEC.md](docs/HEADS_SYSTEM_SPEC.md)):
- Explicit `Head` interface with `required_fields()`, `produces()`, `derive()`
- Per (model, market) pair, one designated head (no layering)
- For Bradley-Terry SPREAD, recommend **Thurstone-Mosteller** approach:
  - Fit noise term σ from margin residuals during model fit
  - Deterministically derive margins for new games
  - Clean, testable, calibration-friendly

---

## Deliverables

### 📄 [docs/ENSEMBLE_AND_HEADS_DATAFLOW.md](docs/ENSEMBLE_AND_HEADS_DATAFLOW.md) (~400 lines)

**Content**:
- **A) Definitions**: Forecast row, market, source_id, ensemble_applied, coverage
- **B) End-to-End Trace**: 8 steps from per-model forecasts → BETS workbook, with exact functions, line numbers, inputs/outputs
- **C) Coverage Collapse**: 7 root causes with detection code
- **D) Ensemble Inventory**: 3 ensemble classes (ML/SPREAD/TOTAL) with algorithm pseudocode, inputs, outputs
- **E) Open Questions**: Bradley-Terry heads, projection engine coverage, weights persistence
- **F) Next Actions**: 8 concrete fixes (1 high priority, several medium)

**Highlights**:
- ASCII diagram of full dataflow
- Detailed weight resolution priority order
- Step-by-step ensemble application (especially ML logit pooling)
- "ML becomes BT-only" root cause analysis
- Code locations with line ranges for all functions

---

### 📄 [docs/HEADS_SYSTEM_SPEC.md](docs/HEADS_SYSTEM_SPEC.md) (~350 lines)

**Content**:
- **Problem Statement**: 4 gaps in current system (implicit projection, no standard interface, etc.)
- **Proposed Interface**: `Head` protocol with `market`, `required_fields()`, `produces()`, `derive()`, `derive_batch()`
- **Bradley-Terry Examples**: 3 options
  - Option A: Thurstone-Mosteller (recommended) — learn σ from residuals
  - Option B: Probability inversion — invert p_win back to margin
  - Option C: Joint likelihood — fit wins + margins together
- **Integration Point**: Where heads plug into forecast pipeline
- **Avoiding Duplicates**: MODEL_HEAD_REGISTRY pattern
- **Calibration Attachment**: How calibrators refine head outputs
- **Testing Strategy**: Unit, integration, and equivalence tests
- **Implementation Roadmap**: 5 phases (no code yet)

**Highlights**:
- Full pseudocode for each Bradley-Terry option
- Clear pros/cons for each approach
- Integration flow showing (`win_prob_source` audit trail)
- Registry design to prevent output duplication
- Explicit separation: native vs. derived

---

## Quick Reference: "ML Becomes BT-Only" Path

If you need to diagnose why an ensemble collapsed to single-model:

1. **Check forecast counts**: Do forecast rows exist for all models?
   ```python
   ml_rows_df = pd.DataFrame(market_forecast_rows["ML"])
   print(ml_rows_df["model_name"].value_counts())
   # If Elo count << Bradley-Terry count, Elo is missing rows
   ```

2. **Check columns**: Do all models have required columns?
   ```python
   for model, group in ml_rows_df.groupby("model_name"):
     missing = [c for c in ["p_home_win"] if c not in group.columns or not group[c].notna().any()]
     if missing:
       print(f"{model}: missing {missing}")
   ```

3. **Check weight filtering**: Were models dropped due to zero/negative weight?
   ```
   Look for log line: "[_filter_market_weights_for_forecast] Dropped models for ML: ..."
   ```

4. **Check window mismatch**: Is BETS schedule narrower than forecast window?
   ```python
   bets_ids = set(bets_schedule_df["game_id"].unique())
   ml_ids = set(ml_rows_df["game_id"].unique())
   orphaned = ml_ids - bets_ids
   print(f"Orphaned ensemble forecasts: {len(orphaned)} games")
   ```

See [Section C.7 in ENSEMBLE_AND_HEADS_DATAFLOW.md](docs/ENSEMBLE_AND_HEADS_DATAFLOW.md#ml-becomes-bt-only-scenario) for full diagnostic.

---

## Next Steps (For Team)

### Immediate (Stakeholder Review)
- [ ] Review both docs for accuracy
- [ ] Confirm Bradley-Terry heads approach (TM vs. inversion vs. joint)
- [ ] Identify which next-action is highest priority

### Short-term (Code-Ready)
- [ ] Implement [Next Action #1: Assert Multi-Model Ensemble](docs/ENSEMBLE_AND_HEADS_DATAFLOW.md#1-assert-multi-model-ensemble-before-collapse-high) (catches collapse early)
- [ ] Implement [Next Action #2: Model Support Matrix](docs/ENSEMBLE_AND_HEADS_DATAFLOW.md#2-document-model-support-matrix-medium) (prevent misconfiguration)
- [ ] Verify [tests/test_bets_ensemble_gating.py](tests/test_bets_ensemble_gating.py) passes

### Medium-term (Design Phase)
- [ ] Design Bradley-Terry SPREAD head (pick one of 3 options from HEADS_SYSTEM_SPEC)
- [ ] Proposal for phased heads implementation (Phase 2: ML heads first, zero breaking changes)
- [ ] Stakeholder signoff on heads registry pattern

### Long-term (Implementation)
- See Roadmap in [docs/HEADS_SYSTEM_SPEC.md](docs/HEADS_SYSTEM_SPEC.md) (5 phases, not started)

---

## Files Modified

- ✅ **Created** [docs/ENSEMBLE_AND_HEADS_DATAFLOW.md](docs/ENSEMBLE_AND_HEADS_DATAFLOW.md) — Complete end-to-end map with 8-step trace
- ✅ **Created** [docs/HEADS_SYSTEM_SPEC.md](docs/HEADS_SYSTEM_SPEC.md) — Design doc with 3 Bradley-Terry options + integration pattern
- ✅ **This file** — Summary + quick reference

**No code changes** (investigation only, as requested)

---

## Testing Validation

All analysis done against live codebase. Key locations verified:

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| Per-model forecasts | schedule.py | 895-1100 | ✅ Verified |
| Weight resolution | schedule.py | 1300-1450 | ✅ Verified |
| Weight filtering | schedule.py | 1459-1650 | ✅ Verified |
| ML ensemble | schedule.py | 3246-3400 | ✅ Verified |
| SPREAD ensemble | schedule.py | 3503-3700 | ✅ Verified |
| TOTAL ensemble | schedule.py | 3681-3800 | ✅ Verified |
| Calibration | schedule.py | 333-600 | ✅ Verified |
| BETS builder | schedule.py | 1800-2600 | ✅ Verified |
| BETS validator | schedule.py | 1034-1080 | ✅ Verified |
| ML ensemble class | ml_v1.py | 1-195 | ✅ Verified |
| SPREAD ensemble class | spread_v1.py | 1-191 | ✅ Verified |
| TOTAL ensemble class | total_v1.py | 1-199 | ✅ Verified |

---

## Caveats & Open Questions

1. **Ensemble config defaults**: Are all 3 markets (ML/SPREAD/TOTAL) guaranteed to have fallback configs? Recommend checking [src/ensemble/default_configs/](src/ensemble/default_configs/) exists with `{ML,SPREAD,TOTAL}.json`.

2. **Poisson to SPREAD**: Can Poisson (which only predicts totals natively) be used in SPREAD ensemble? Current system would need projection engine to invert. Recommend clarifying model support matrix.

3. **Calibrator availability**: If a calibrator is missing, system silently falls back to uncalibrated. Should there be an option to fail-fast? Current design permits missing calibrators; may not always be desired.

4. **BETS contract**: Current contract [test_bets_ensemble_gating.py](tests/test_bets_ensemble_gating.py) enforces SPREAD/TOTAL must use ensemble (no single-model fallback). Why? This is stricter than ML (which allows single model). Document the rationale.

5. **Bradley-Terry total support**: BT currently has `supports_total=False`. Is this intentional, or should it support total if a head is implemented? Recommend clarifying model roadmap.

---

## Summary for Rapid Onboarding

**TL;DR**: 
- Ensembles work in 3 stages: per-model → combination (logit/average) → calibration
- Coverage can collapse at 7 different points; most common is weight filtering
- "Heads" system would make output derivation explicit (currently implicit)
- Bradley-Terry needs SPREAD head (TM, inversion, or joint approach)
- No code bugs found; design is sound but fragmented

---

**Investigation Complete** ✅

Docs ready for review. Next actions itemized. System thoroughly mapped.

