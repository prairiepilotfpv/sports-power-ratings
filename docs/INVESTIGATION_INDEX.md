# Investigation: Ensemble & Heads Architecture — Document Index

This folder contains three comprehensive documentation files investigating how ensembles and market-specific output derivation ("heads") work in the sports-power-ratings system.

## 📚 Reading Guide

Start with the **Investigation Summary**, then dive into the detailed maps based on interest.

### [INVESTIGATION_SUMMARY_ENSEMBLE_HEADS.md](INVESTIGATION_SUMMARY_ENSEMBLE_HEADS.md) (START HERE)
**~100 lines | 10-minute read**

Executive summary of findings, including:
- Three-layer architecture overview
- Seven coverage collapse root causes (quick reference table)
- Quick diagnostic path for "ML becomes BT-only"
- Links to deeper docs

**Best for**: Quick context, identifying which detailed doc to read

---

### [ENSEMBLE_AND_HEADS_DATAFLOW.md](ENSEMBLE_AND_HEADS_DATAFLOW.md)
**~1000 lines | 30-minute detailed read**

Comprehensive end-to-end mapping of ensemble operations:

**Section A) Definitions & Contracts** (50 lines)
- What is a forecast row, market, source_id, ensemble_applied, coverage
- Core types and their requirements

**Section B) End-to-End Pipeline Trace** (400 lines)
- 8-step visual flowchart from per-model forecasts → final BETS workbook
- Exact function names, file paths, line numbers
- Detailed breakdowns of each step:
  1. Per-model forecasts (`_build_market_forecasts_for_ensembles`)
  2. Weight resolution (`_resolve_ensemble_weights`)
  3. Weight filtering (`_filter_market_weights_for_forecast`)
  4. Ensemble application (ML/SPREAD/TOTAL blocks)
  5. Calibration application (`_apply_calibration_to_schedule_df`)
  6. BETS dataframe construction (`_build_bets_dataframe`)
  7. Contract validation (`_validate_bets_ensemble_contract`)
  8. Export (workbook writing)
- Algorithm pseudocode for each ensemble type
- Input/output specs for every step

**Section C) Coverage Collapse Diagnostics** (250 lines)
- 7 root causes with code locations, detection methods, examples
- "ML becomes BT-only" root cause analysis
- Prevention strategies (assertions, assertions)

**Section D) Ensemble Layer Inventory** (150 lines)
- MLWeightedAverageEnsemble (logit pooling algorithm)
- SpreadWeightedAverageEnsemble (weighted average + variance composition)
- TotalWeightedAverageEnsemble (variance handling + fallback to uniform)
- Component JSON schema

**Section E) Open Questions** (50 lines)
- Bradley-Terry heads design gap
- Projection engine coverage
- Ensemble weights persistence

**Section F) Next Actions** (50 lines)
- 8 concrete fixes with priorities
- Each tied to specific code location and testable assertion

**Best for**: Deep understanding of dataflow, diagnosing pipeline failures, detailed algorithm review

---

### [HEADS_SYSTEM_SPEC.md](HEADS_SYSTEM_SPEC.md)
**~350 lines | 20-minute read**

Design specification for a missing "heads" architecture (no code implementation):

**Problem Statement** (40 lines)
- Bradley-Terry lacks margin/total outputs
- Projection engine is implicit and fragmented
- No standard interface for derived outputs
- Hard to audit output origin

**Proposed Heads Architecture** (150 lines)
- `Head` protocol/interface definition (pseudocode)
- Example: Three Bradley-Terry SPREAD head options:
  - **Option A**: Thurstone-Mosteller (recommended) — learn σ from margin residuals
  - **Option B**: Probability inversion — invert p_win back to margin
  - **Option C**: Joint likelihood — fit wins + margins together
- Head integration point in forecast pipeline
- MODEL_HEAD_REGISTRY pattern to avoid duplicates

**How Calibration Attaches** (30 lines)
- Calibrators refine head outputs (not replace them)
- Provenance tags track both derivation and calibration

**Testing Strategy** (40 lines)
- Unit tests per head
- Integration tests
- Equivalence tests (new vs. old)

**Implementation Roadmap** (30 lines)
- 5 phases (design complete, implementation not started)

**Best for**: Understanding output derivation design, Bradley-Terry options, future heads implementation planning

---

## 🔍 Key Takeaways

### The Three-Layer Model
```
Per-Model Forecasts → Ensemble Combination → Calibration & Provenance
   (Fit & predict)    (Logit/Average)      (Refine + tag)
```

### Coverage Collapse: Seven Failure Points
1. Missing forecast rows (model didn't produce output)
2. Missing columns (model lacks required fields)
3. Join key mismatch (game_id, date, team name differ)
4. NaN filtering (calibration failed for some rows)
5. Weight filtering (tuned weights dropped models)
6. Schedule window mismatch (BETS narrower than forecasts)
7. Model doesn't support market (metadata says `supports_total=False`)

### "ML Becomes BT-Only" Path
Ensemble collapses to single-model when:
1. Elo forecast rows exist but have NaN p_home_win
2. `_filter_market_weights_for_forecast` drops Elo
3. Final ensemble has only 1 model
4. Ensemble application skipped (count ≤ 1)
5. Output marked "bradley-terry" not "ensemble_ml_v1"

**Detection**: Check forecast counts per model + watch logs for `_filter_market_weights_for_forecast` lines.

### Bradley-Terry Output Gap
**Current**: BT produces `p_win` natively but margin only via calibration  
**Missing**: Native SPREAD head (no direct margin SD)  
**Proposed**: Thurstone-Mosteller approach (fit σ from residuals during fit)  
**Benefit**: Deterministic, testable, calibration-compatible

---

## 📋 Quick Navigation

**I want to...**

- **Understand the full ensemble pipeline**: Read Section B in [ENSEMBLE_AND_HEADS_DATAFLOW.md](ENSEMBLE_AND_HEADS_DATAFLOW.md)

- **Diagnose why ML became single-model**: See Section C.7 in [ENSEMBLE_AND_HEADS_DATAFLOW.md](ENSEMBLE_AND_HEADS_DATAFLOW.md)

- **Know why forecast rows are missing**: See Section C (all 7 root causes) in [ENSEMBLE_AND_HEADS_DATAFLOW.md](ENSEMBLE_AND_HEADS_DATAFLOW.md)

- **Understand ensemble algorithms**: See Section D in [ENSEMBLE_AND_HEADS_DATAFLOW.md](ENSEMBLE_AND_HEADS_DATAFLOW.md)

- **See next priorities**: Check Section F in [ENSEMBLE_AND_HEADS_DATAFLOW.md](ENSEMBLE_AND_HEADS_DATAFLOW.md)

- **Design Bradley-Terry heads**: Read [HEADS_SYSTEM_SPEC.md](HEADS_SYSTEM_SPEC.md) Section 2 (all 3 options)

- **Quick 10-minute overview**: Start with [INVESTIGATION_SUMMARY_ENSEMBLE_HEADS.md](INVESTIGATION_SUMMARY_ENSEMBLE_HEADS.md)

---

## 📂 Related Existing Docs

These docs reference and build on:
- [ENSEMBLE_ARCHITECTURE.md](ENSEMBLE_ARCHITECTURE.md) — Ensemble class overview
- [ensembles.md](ensembles.md) — Ensemble concept guide
- [model_canonization_playbook.md](model_canonization_playbook.md) — Model output standards
- [CLI.md](CLI.md) — CLI reference for schedule/backtest commands
- [bradley_terry_heads.md](bradley_terry_heads.md) — Existing BT heads notes

---

## 🔗 Code References

### Key Files
- **Ensemble orchestration**: [src/pipelines/schedule.py](../src/pipelines/schedule.py) (lines 895-3800)
- **ML ensemble**: [src/ensemble/ml_v1.py](../src/ensemble/ml_v1.py)
- **SPREAD ensemble**: [src/ensemble/spread_v1.py](../src/ensemble/spread_v1.py)
- **TOTAL ensemble**: [src/ensemble/total_v1.py](../src/ensemble/total_v1.py)
- **Ensemble config**: [src/ensemble/config.py](../src/ensemble/config.py)

### Key Tests
- [tests/test_bets_ensemble_gating.py](../tests/test_bets_ensemble_gating.py) — Ensemble coverage + contract
- [tests/test_calibration_bets_integration.py](../tests/test_calibration_bets_integration.py) — Calibration + provenance tags
- [tests/test_pipeline_canonization.py](../tests/test_pipeline_canonization.py) — System invariants

---

## 📝 Document Status

✅ **COMPLETE** — All investigation done, no code changes made (as requested)

- **Created**: [ENSEMBLE_AND_HEADS_DATAFLOW.md](ENSEMBLE_AND_HEADS_DATAFLOW.md) — Full end-to-end map (1036 lines)
- **Created**: [HEADS_SYSTEM_SPEC.md](HEADS_SYSTEM_SPEC.md) — Design spec (350 lines)
- **Created**: [INVESTIGATION_SUMMARY_ENSEMBLE_HEADS.md](INVESTIGATION_SUMMARY_ENSEMBLE_HEADS.md) — Executive summary (200 lines)
- **Created**: This index file

**No bugs found or code modified** — This was investigation-only as specified.

---

## ❓ Questions or Updates?

Refer to Section E (Open Questions) in [ENSEMBLE_AND_HEADS_DATAFLOW.md](ENSEMBLE_AND_HEADS_DATAFLOW.md) for known unknowns.

If you need clarification on any section, the corresponding code location and line numbers are provided for quick reference.

