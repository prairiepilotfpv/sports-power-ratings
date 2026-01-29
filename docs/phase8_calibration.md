# Phase 8 Calibration: Mean-Variance Separation

This directory contains new calibration tools for Phase 8, focused on explicit separation of mean (bias) and variance (uncertainty) corrections.

## New Modules

### `mean_calibrator.py`
**Purpose**: Shift predicted mean to correct bias without modifying uncertainty.

```python
from calibration.mean_calibrator import MeanCalibrator

# Fit on historical data
calibrator = MeanCalibrator(delta_min=-30, delta_max=30)
calibrator.fit(df)  # expects: pred_mean, actual_value, (optional pred_sd)

# Apply to predictions
result = calibrator.transform(predictions)
# Returns: calibrated_mean, calibrated_sd (unchanged)
```

**Key Properties**:
- Fits additive bias: `delta = mean(residuals) / (1 + regularization)`
- SD remains unchanged
- Validates: No NaNs, SD > 0, minimum 2 samples
- Respects guardrails: delta ∈ [delta_min, delta_max]
- Metadata: stores delta, rmse_before, rmse_after, mean_before, mean_after

### `variance_audit.py`
**Purpose**: Compute calibration quality metrics and log diagnostics.

```python
from calibration.variance_audit import compute_variance_audit, log_variance_audit

audit = compute_variance_audit(predictions_df)
# Returns: VarianceAuditResult with:
#   - sd_before, sd_after (median SD)
#   - empirical_mae (mean abs error vs SD)
#   - coverage_1sd, coverage_2sd (fraction within bounds)
#   - outside_95ci (% beyond 95% CI)

log_variance_audit(audit, "TOTAL")
# Logs: [VARIANCE AUDIT TOTAL] sd_before=..., sd_after=..., coverage_1sd=...%, ...
```

**Metrics**:
- **empirical_mae**: How well predicted SD matches residual magnitudes
- **coverage_1sd**: Fraction of outcomes within ±1 standard deviation
- **coverage_2sd**: Fraction of outcomes within ±2 standard deviations  
- **outside_95ci**: Fraction beyond ±1.96 SD (ideal: ~5%)

### `validation.py`
**Purpose**: Strict validation of calibration input data.

```python
from calibration.validation import (
    validate_calibration_data_source,
    validate_no_nans_in_columns,
    check_outcome_range,
)

# Full data source validation
validate_calibration_data_source(
    df,
    market="TOTAL",
    sport="nba",
    season="2025-26",
    fail_on_errors=True,
)

# Strict NaN checks
validate_no_nans_in_columns(df, ["pred_mean", "actual_value"], fail=True)

# Bounds checks
check_outcome_range(df, "total", min_val=50, max_val=250, fail=True)
```

**Validation**:
- Required columns by market
- NaN detection (>10% triggers error)
- SD > 0 where present
- Outcome value ranges
- Minimum sample count (2+ for fitting)

## Usage: Two-Stage TOTAL Calibration

The schedule pipeline now applies calibration in two explicit stages:

```python
from pipelines.schedule import _apply_two_stage_total_calibration

# Apply both mean and variance calibration
df, applied = _apply_two_stage_total_calibration(
    df,
    sport="nba",
    season="2025-26",
    model="elo",
)

# Logs:
# [TOTAL calibration] MEAN stage: delta=+4.5, improvement=+12.3%, samples=234
# [TOTAL calibration] VARIANCE stage: sd_before=7.45, sd_after=8.12, change=+9.0%, ...
# [TOTAL calibration] Completed stages: mean+variance
```

### Stage Separation Guarantees

**Stage 1 (Mean Calibration)**:
- Input: (pred_mean, pred_sd) → (actual_value)
- Output: (pred_mean + delta, pred_sd)
- Effect: Shifts location only
- SD preserved exactly

**Stage 2 (Variance Calibration)**:
- Input: (pred_mean, pred_sd) → (actual_value)
- Output: (pred_mean, sqrt((c*sd)^2 + tau^2))
- Effect: Scales uncertainty
- Mean preserved exactly

## Calibration Window Clarity

Historical calibration now logs which data was used:

```
[calibration window] market=TOTAL mode=expanding start=2025-01-01 
end=2025-01-28 actual_start=2025-01-02 actual_end=2025-01-27 games=42
```

This provides:
- **mode**: "expanding" (all data) or "rolling" (recent window)
- **start/end**: Explicit date filters applied
- **actual_start/end**: Actual date range in data
- **games**: Sample count used for fitting

## Test Coverage

See `tests/test_phase8_calibration.py` (19 tests):
- Mean calibration: shifts mean, preserves SD, respects bounds
- Variance calibration: preserves mean, adjusts SD
- Separation guarantee: stages work independently
- Audit metrics: computed correctly
- Validation: rejects NaNs, invalid ranges
- API: transform contract enforcement

Run tests:
```bash
pytest tests/test_phase8_calibration.py -v
```

## Integration Points

### In Schedule Pipeline
- `_apply_two_stage_total_calibration()`: Two-stage application
- Logs to `[TOTAL calibration]` prefix
- Appends "calibrated_total" to win_prob_source

### In Historical Calibration
- `_log_calibration_window()`: Window clarity logging
- Called before each market's fit_calibrator_for_market()
- Window info: mode, start, end, game count

### Downstream Consumers
- Mean and variance corrections now traceable separately
- Metadata includes stage-specific diagnostics
- Audit metrics available for monitoring

## Design Principles

1. **Separation of Concerns**: Mean ≠ Variance
   - One never touches what the other modifies
   - Tested explicitly in test suite

2. **Data Validation First**: Fail loudly on bad inputs
   - NaNs rejected with context
   - SD ≤ 0 caught early
   - Sample count verified

3. **Transparency**: All numbers logged
   - Pre/post statistics visible
   - Guardrail clipping reported
   - Window/mode always explicit

4. **No Silent Defaults**: Every decision logged
   - Missing calibrators logged at DEBUG
   - Stage completions at INFO
   - Validation failures with reasons

## Performance Notes

- Mean calibration: O(n) (single pass for mean residual)
- Variance calibration: O(n) per optimization iteration
- Audit metrics: O(n) (one pass for all stats)
- Validation: O(n) (one pass for NaN/range checks)

All operations scale linearly with sample size.

## Future Extensions

1. **Per-Market Calibration Windows**: rolling vs expanding per sport
2. **Adaptive Parameters**: Delta bounds, regularization per season
3. **Cross-Validation**: Out-of-sample validation for each stage
4. **Ensemble Calibration**: Separate by ensemble member before combining
5. **Performance Tracking**: Maintain audit metrics over time
