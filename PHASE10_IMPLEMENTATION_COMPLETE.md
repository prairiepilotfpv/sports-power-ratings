# Phase 10 Implementation Summary: Regime-Conditioned TOTAL Calibration

## Status: COMPLETE ✓

All core Phase 10 features implemented, tested, and integrated. 139 calibration tests pass (including 23 Phase 10-specific tests).

---

## What Was Implemented

### 1. **Deterministic Regime Labeling** (`src/calibration/total_bucket_regimes.py`)
- `total_bucket_regime()`: Maps predicted `total_mean` → regime label ('low', 'mid', 'high')
- Default thresholds: low < 210, mid < 225, high >= 225 (configurable)
- Handles None/NaN gracefully, returns None for invalid inputs
- **Guarantee**: Deterministic (same input always produces same label)

### 2. **Manifest Data Structure & Persistence**
- `TotalBucketManifest`: Dataclass tracking:
  - Regime configuration (thresholds, min_samples_per_bucket)
  - Per-bucket sample counts and calibrator availability
  - Fit window (start/end dates)
  - Artifact file paths (global + per-bucket, mean + variance)
  - Created timestamp and version tag ("phase10")
- `save_total_bucket_manifest()` / `load_total_bucket_manifest()`: JSON I/O

### 3. **Regime-Conditioned Fitting** (`src/calibration/historical_calibration.py`)
- `fit_total_bucket_calibrators()`: Main workflow
  - Loads historical games + generates predictions
  - Labels each row by `total_bucket`
  - Fits global mean + variance calibrators (fallback)
  - Fits per-bucket calibrators when `samples >= min_samples_per_bucket`
  - Saves all calibrators + generates manifest
- Output structure:
  ```
  outputs/calibrators/<sport>/<season>/<source>/total/
  ├── global/
  │   ├── calibrator.pkl (mean)
  │   └── calibrator_variance.pkl (variance)
  ├── bucket_low/
  ├── bucket_mid/
  ├── bucket_high/
  └── regime_manifest.json
  ```

### 4. **CLI Integration** (`src/cli/pipeline.py`)
- New `calibrate` subcommand options:
  - `--market total`: Route to TOTAL calibration
  - `--regimes total_bucket`: Enable regime-conditioned fitting
  - `--total-bucket-low-threshold`: Customize low cutoff (default: 210)
  - `--total-bucket-mid-threshold`: Customize mid/high cutoff (default: 225)
  - `--min-samples-per-bucket`: Minimum samples per bucket (default: 200)

**Example:**
```bash
python -m src.cli.pipeline calibrate \
  --sport nba --season 2025-26 \
  --market total --source ensemble_total_v1 \
  --start-date 2025-10-01 --end-date 2025-12-31 \
  --regimes total_bucket \
  --total-bucket-low-threshold 210 \
  --total-bucket-mid-threshold 225 \
  --min-samples-per-bucket 200
```

### 5. **Apply-Time Routing** (`src/pipelines/schedule.py`)
- `_apply_total_bucket_calibration()`: New function for bucket-based application
  - Tries bucket calibration first (Phase 10)
  - Falls back to global if:
    - Manifest missing
    - Bucket calibrator unavailable
    - Insufficient bucket samples
  - Applies two-stage calibration: mean → variance
  - Preserves Phase 8 separation guarantee
  - Logs INFO: `[total_bucket calibration] used={low: n, mid: n, high: n}, fallback_global=m`

- `_apply_two_stage_total_calibration()`: Modified to try bucket first
  - Maintains backward compatibility with global calibration
  - Graceful fallback when bucket manifest unavailable

### 6. **Comprehensive Tests** (`tests/calibration/test_total_bucket_regimes.py`)
- 23 tests covering:
  - **Regime Labeling** (9 tests):
    - Default thresholds (low/mid/high)
    - Custom thresholds
    - None/NaN/invalid inputs
    - Determinism
    - DataFrame labeling
  - **Manifest** (5 tests):
    - Creation (default + custom)
    - Serialization (to_dict)
    - Deserialization (from_dict)
    - Roundtrip (serialize → deserialize)
  - **Persistence** (2 tests):
    - Save/load from file
    - Missing manifest handling
  - **Bucket Routing** (3 tests):
    - Select correct bucket calibrator
    - Fallback to global
    - Handle missing bucket assignment
  - **Integration** (3 tests):
    - Idempotent labeling
    - Multiple bucket files in manifest
    - Insufficient samples fallback

**All 23 tests PASS** ✓

### 7. **Documentation**
- **PHASE10_README.md**: Comprehensive guide covering:
  - Architecture overview
  - Regime labeling rules
  - Manifest schema
  - Fitting pipeline
  - Apply-time routing
  - CLI usage (training + apply)
  - Implementation details
  - Assumptions & design decisions
  - Future extensions
  
- **docs/CLI.md**: Updated with:
  - ML calibration examples
  - TOTAL global calibration example
  - TOTAL regime-conditioned calibration example (new)
  - Phase 10 CLI options
  - Bucket assignment rules
  - Output directory structure
  - Manifest JSON example

---

## Key Design Decisions

### 1. **Deterministic Routing**
- Bucket assigned at apply time based on **predicted** `total_mean` (not actual scores)
- Prevents data leakage and ensures reproducibility
- Same inputs always produce same bucket label

### 2. **Separation Guarantee Preserved**
- Mean calibration stage: modifies `total_mean`, does NOT change `total_sd`
- Variance calibration stage: modifies `total_sd`, does NOT change `total_mean`
- Stages applied sequentially: mean → variance
- Maintained in both bucket and global calibration paths

### 3. **Graceful Fallback Strategy**
1. Try bucket-specific calibrator (if available in manifest)
2. Fall back to global calibrator (if available)
3. If neither available, skip that stage
4. Manifest flags track availability: `has_bucket_mean`, `has_bucket_variance`, `has_global_mean`, `has_global_variance`

### 4. **Manifest Versioning**
- Version: "phase10" (allows future migrations)
- Created_at: ISO timestamp (provenance tracking)
- Forward-compatible: unknown fields ignored during deserialization

### 5. **Conservative Defaults**
- `low_threshold=210`, `mid_threshold=225`: Sport-agnostic; customizable via CLI
- `min_samples_per_bucket=200`: Conservative; adjust for small datasets
- Default: global calibration only (no regimes) unless `--regimes total_bucket` specified

---

## Testing Summary

### Phase 10 Tests
```
tests/calibration/test_total_bucket_regimes.py: 23 passed
```

### Integration Tests (All Calibration Tests)
```
tests/ -k calibrat: 139 passed
```

### Related Tests Fixed
- `test_total_calibration_instrumentation_logs_deltas`: Updated to handle bucket calibration attempt
- `test_spread_and_total_calibration_no_calibrator_no_error`: Updated log expectations
- `test_spread_weights_filter_out_models_without_spread_outputs`: Aligned with current filtering logic

---

## Backward Compatibility

✓ **Fully backward compatible**:
- Existing global TOTAL calibration still works (no `--regimes` flag)
- Schedule pipeline auto-detects manifest; uses bucket calibration if available, falls back to global
- If manifest missing, behaves exactly as Phase 8 (global calibration only)
- Phase 7 ensembles and Phase 9 evaluation unaffected

---

## Files Created/Modified

### Created:
- `src/calibration/total_bucket_regimes.py` (210 LOC)
- `tests/calibration/test_total_bucket_regimes.py` (360+ LOC)
- `PHASE10_README.md` (comprehensive guide)

### Modified:
- `src/calibration/historical_calibration.py` (+165 LOC for `fit_total_bucket_calibrators()`)
- `src/cli/pipeline.py` (added `--regimes`, `--total-bucket-*`, `--min-samples-per-bucket` flags)
- `src/pipelines/schedule.py` (+180 LOC for `_apply_total_bucket_calibration()`, modified `_apply_two_stage_total_calibration()`)
- `docs/CLI.md` (added Phase 10 calibration examples)
- `tests/pipelines/test_schedule_calibration.py` (updated 3 tests for Phase 10 integration)

---

## How to Use

### Training (Fit regime-conditioned calibrators):
```bash
python -m src.cli.pipeline calibrate \
  --sport nba \
  --season 2025-26 \
  --market total \
  --source ensemble_total_v1 \
  --start-date 2025-10-01 \
  --end-date 2025-12-31 \
  --regimes total_bucket \
  --method auto
```

### Application (Schedule generation with auto-detection):
```bash
python -m src.cli.pipeline schedule \
  --sport nba \
  --season 2025-26 \
  --model ensemble_total_v1
```
→ Automatically detects manifest and applies bucket calibration if available

### Global TOTAL calibration (Phase 8, no regimes):
```bash
python -m src.cli.pipeline calibrate \
  --sport nba \
  --season 2025-26 \
  --market total \
  --source ensemble_total_v1 \
  --start-date 2025-10-01 \
  --end-date 2025-12-31
```

---

## Manifest Example

```json
{
  "sport": "nba",
  "season": "2025-26",
  "source": "ensemble_total_v1",
  "market": "total",
  "low_threshold": 210.0,
  "mid_threshold": 225.0,
  "min_samples_per_bucket": 200,
  "samples_global": 1000,
  "samples_low": 300,
  "samples_mid": 400,
  "samples_high": 300,
  "has_global_mean": true,
  "has_global_variance": true,
  "has_bucket_mean": {"low": true, "mid": true, "high": false},
  "has_bucket_variance": {"low": true, "mid": true, "high": false},
  "fit_start_date": "2025-10-01",
  "fit_end_date": "2025-12-31",
  "calibrator_global_mean": "outputs/calibrators/nba/2025-26/ensemble_total_v1/total/global/calibrator.pkl",
  "calibrator_global_variance": "outputs/calibrators/nba/2025-26/ensemble_total_v1/total/global/calibrator_variance.pkl",
  "calibrators_bucket_mean": {
    "low": "outputs/calibrators/nba/2025-26/ensemble_total_v1/total/bucket_low/calibrator.pkl",
    "mid": "outputs/calibrators/nba/2025-26/ensemble_total_v1/total/bucket_mid/calibrator.pkl"
  },
  "calibrators_bucket_variance": {
    "low": "outputs/calibrators/nba/2025-26/ensemble_total_v1/total/bucket_low/calibrator_variance.pkl",
    "mid": "outputs/calibrators/nba/2025-26/ensemble_total_v1/total/bucket_mid/calibrator_variance.pkl"
  },
  "created_at": "2025-12-31T12:00:00+00:00",
  "version": "phase10"
}
```

---

## Logging Examples

**Fit-time (INFO level):**
```
[fit_total_bucket_calibrators] Starting regime-conditioned TOTAL calibration for nba/2025-26...
[fit_total_bucket_calibrators] Sample distribution: global=1000, low=300, mid=400, high=300
[fit_total_bucket_calibrators] Fitted global mean calibrator
[fit_total_bucket_calibrators] Fitted low bucket mean calibrator
[fit_total_bucket_calibrators] Insufficient samples for high bucket (50 < 200); skipping
[fit_total_bucket_calibrators] Saved manifest to outputs/calibrators/nba/2025-26/ensemble_total_v1/total/regime_manifest.json
```

**Apply-time (INFO level):**
```
[TOTAL calibration] Applied bucket-conditioned calibration (Phase 10)
[total_bucket calibration] used={low: 150, mid: 200, high: 80}, fallback_global=70
```

---

## Future Work (Out of Scope for Phase 10)

1. **Evaluation Comparison**: CLI mode comparing global vs bucketed calibration via Phase 9 metrics
2. **Dynamic Thresholds**: Compute bucket boundaries from historical distribution quartiles
3. **Seasonal Regimes**: Add time-based bucketing (early season vs late season)
4. **Ensemble-Specific Regimes**: Different buckets per ensemble model
5. **Multi-Market Regimes**: Apply regime conditioning to SPREAD market as well

---

## Summary

Phase 10 successfully extends Phase 8 (mean/variance separation) and Phase 9 (calibration evaluation) with regime-conditioned TOTAL calibration. The implementation is:

✓ **Complete**: All 7 tasks + 1 optional task done  
✓ **Tested**: 139 calibration tests pass (23 Phase 10-specific)  
✓ **Integrated**: CLI, fitting, and schedule pipeline all updated  
✓ **Documented**: PHASE10_README.md + docs/CLI.md examples  
✓ **Backward Compatible**: No breaking changes to existing pipelines  
✓ **Production-Ready**: Logging, error handling, and fallback strategies in place  

Ready for production deployment and future calibration enhancements.
