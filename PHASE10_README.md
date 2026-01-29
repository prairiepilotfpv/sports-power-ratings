# Phase 10: Regime-Conditioned TOTAL Calibration

## Overview

Phase 10 implements **regime-conditioned TOTAL calibration** using `total_bucket` regimes. This extends Phase 8 (mean/variance separation) and Phase 9 (calibration evaluation) by fitting separate calibrators per total prediction bucket (low, mid, high), with fallback to global calibrators.

**Key Feature**: Separate TOTAL calibrators (mean + variance) per `total_bucket`, enabling calibration strategies tailored to low-scoring, mid-range, and high-scoring games.

## Architecture

### 1. Regime Labeling (`total_bucket_regime()`)

Deterministically assigns rows to buckets based on predicted `total_mean`:
- **low**: `total_mean < 210`
- **mid**: `210 <= total_mean < 225`
- **high**: `total_mean >= 225`

Thresholds are configurable via CLI flags or manifest.

### 2. Manifest Schema (`TotalBucketManifest`)

JSON structure tracking:
- Regime configuration (thresholds, min_samples per bucket)
- Per-bucket sample counts and calibrator availability flags
- Artifact file paths (global + per-bucket, mean + variance)
- Fit window (start/end dates) for provenance
- Created timestamp and version tag

Example manifest location:
```
outputs/calibrators/nba/2025-26/historical/regime_manifest.json
```

### 3. Fitting Pipeline (`fit_total_bucket_calibrators()`)

Workflow:
1. Load historical games + model predictions for TOTAL market
2. Generate per-bucket calibration datasets
3. Fit global mean + variance calibrators (fallback)
4. Fit per-bucket mean + variance calibrators (if samples >= min_samples_per_bucket)
5. Save all calibrators to disk + generate manifest

Output structure:
```
outputs/calibrators/nba/2025-26/historical/
├── global/
│   ├── calibrator.pkl        # global mean calibrator
│   ├── calibrator_variance.pkl  # global variance calibrator
│   └── metadata.json
├── bucket_low/
│   ├── calibrator.pkl        # low bucket mean calibrator
│   ├── calibrator_variance.pkl  # low bucket variance calibrator
│   └── metadata.json
├── bucket_mid/
│   └── ... (similar)
├── bucket_high/
│   └── ... (similar)
└── regime_manifest.json       # Manifest with sample counts & flags
```

### 4. Apply-Time Routing (`_apply_total_bucket_calibration()`)

In schedule pipeline:
1. Load manifest + calibrators (global + available buckets)
2. For each row, compute `total_bucket` from predicted `total_mean`
3. Route to bucket calibrator if available; else global
4. Apply two-stage calibration (mean → variance)
5. Log usage summary: `used={low: n, mid: n, high: n}, fallback_global=m`

## CLI Usage

### Training: Fit regime-conditioned TOTAL calibrators

```bash
python -m src.cli.pipeline calibrate \
  --sport nba \
  --season 2025-26 \
  --market total \
  --source ensemble_total_v1 \
  --start-date 2025-10-01 \
  --end-date 2025-12-31 \
  --regimes total_bucket \
  --total-bucket-low-threshold 210.0 \
  --total-bucket-mid-threshold 225.0 \
  --min-samples-per-bucket 200 \
  --method auto
```

**Options:**
- `--regimes total_bucket`: Enable Phase 10 bucketing (required for regime-conditioned fit)
- `--total-bucket-low-threshold`: Cutoff for low bucket (default: 210)
- `--total-bucket-mid-threshold`: Cutoff for mid/high boundary (default: 225)
- `--min-samples-per-bucket`: Minimum samples to fit a bucket calibrator (default: 200)

### Training: Global TOTAL calibration (no regimes)

```bash
python -m src.cli.pipeline calibrate \
  --sport nba \
  --season 2025-26 \
  --market total \
  --source ensemble_total_v1 \
  --start-date 2025-10-01 \
  --end-date 2025-12-31
```

Without `--regimes`, uses standard global calibration (Phase 8).

### Apply: Schedule generation

```bash
python -m src.cli.pipeline schedule \
  --sport nba \
  --season 2025-26 \
  --model ensemble_total_v1
```

Automatically detects and applies bucket-conditioned calibration if manifest exists; falls back to global if not.

## Implementation Details

### Determinism Guarantee

- Regime labels are deterministic: same `total_mean` always produces same bucket
- Manifest created_at timestamp is fixed at save time
- Bucket calibrators are loaded from consistent artifact paths
- Idempotent: multiple apply passes produce same results

### Phase 8 Guarantees Preserved

- **Mean stage**: Does not modify `total_sd`
- **Variance stage**: Does not modify `total_mean` (applied after mean)
- Separation enforced at apply time and in fit

### Fallback Strategy

1. Try bucket-specific calibrator (mean or variance stage)
2. If unavailable, try global calibrator
3. If no global, skip that stage
4. If insufficient bucket samples, manifest flags as unavailable → global

### Logging

Apply-time logging (INFO level):
```
[TOTAL calibration] Applied bucket-conditioned calibration (Phase 10)
[total_bucket calibration] used={low: n, mid: n, high: n}, fallback_global=m
```

Fit-time logging (INFO level):
```
[fit_total_bucket_calibrators] Starting regime-conditioned TOTAL calibration for nba/2025-26...
[fit_total_bucket_calibrators] Sample distribution: global=1000, low=300, mid=400, high=300
[fit_total_bucket_calibrators] Insufficient samples for high bucket (50 < 200); skipping
[fit_total_bucket_calibrators] Saved manifest to outputs/calibrators/nba/2025-26/historical/regime_manifest.json
```

## Testing

### Unit Tests

Run all Phase 10 tests:
```bash
pytest tests/calibration/test_total_bucket_regimes.py -v
```

Coverage:
- Deterministic regime labeling (low/mid/high, custom thresholds, edge cases)
- Manifest creation, serialization, deserialization (roundtrip)
- Manifest file I/O (save/load)
- Bucket selection and fallback routing
- Idempotency and separation guarantees
- Insufficient sample handling

### Integration Tests

1. **Full pipeline test** (optional): Create historical games, fit bucket calibrators, apply in schedule
2. **Fallback test**: Verify global calibration used when manifest missing
3. **Determinism test**: Multiple apply passes produce identical results

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
  "has_bucket_mean": {
    "low": true,
    "mid": true,
    "high": false
  },
  "has_bucket_variance": {
    "low": true,
    "mid": true,
    "high": false
  },
  "fit_start_date": "2025-10-01",
  "fit_end_date": "2025-12-31",
  "calibrator_global_mean": "outputs/calibrators/nba/2025-26/historical/global/calibrator.pkl",
  "calibrator_global_variance": "outputs/calibrators/nba/2025-26/historical/global/calibrator_variance.pkl",
  "calibrators_bucket_mean": {
    "low": "outputs/calibrators/nba/2025-26/historical/bucket_low/calibrator.pkl",
    "mid": "outputs/calibrators/nba/2025-26/historical/bucket_mid/calibrator.pkl"
  },
  "calibrators_bucket_variance": {
    "low": "outputs/calibrators/nba/2025-26/historical/bucket_low/calibrator_variance.pkl",
    "mid": "outputs/calibrators/nba/2025-26/historical/bucket_mid/calibrator_variance.pkl"
  },
  "created_at": "2025-12-31T12:00:00+00:00",
  "version": "phase10"
}
```

## Assumptions & Design Decisions

1. **Bucket thresholds are sport-agnostic by default**: 210/225 work well for NBA/NFL. Customize via CLI flags if needed.
2. **min_samples_per_bucket=200**: Conservative default; adjust if dataset is small.
3. **Deterministic routing**: Bucket assigned at apply time based on predicted `total_mean` (not actual scores), preventing data leakage.
4. **Two-stage within buckets**: Mean calibration followed by variance ensures separation guarantee is preserved per bucket.
5. **Manifest versioning**: `version: "phase10"` allows forward compatibility.

## Future Extensions

- **Seasonal regimes**: Add time-based bucketing (e.g., early season vs late season)
- **Ensemble-specific regimes**: Different buckets per ensemble model
- **Dynamic thresholds**: Compute thresholds from historical distribution quartiles
- **Evaluation comparison**: CLI mode to compare global vs bucketed MAE/RMSE/coverage

## Files Modified/Created

- **New**:
  - `src/calibration/total_bucket_regimes.py` (regime labeling, manifest, routing)
  - `tests/calibration/test_total_bucket_regimes.py` (comprehensive tests)
  
- **Extended**:
  - `src/calibration/historical_calibration.py` (fit_total_bucket_calibrators, CLI integration)
  - `src/cli/pipeline.py` (calibrate subcommand with --regimes)
  - `src/pipelines/schedule.py` (_apply_total_bucket_calibration, integration in _apply_two_stage_total_calibration)

## Summary

Phase 10 brings regime-conditioned TOTAL calibration to the platform. By fitting separate calibrators per total_bucket, we enable:
- Better calibration for specific game contexts (low-scoring vs high-scoring)
- Maintained separation guarantees (mean ≠ variance)
- Deterministic, reproducible routing
- Graceful fallback to global calibration

All implementations are additive, preserving Phase 8 global calibration as fallback and Phase 7/9 ensembles/evaluation unchanged.
