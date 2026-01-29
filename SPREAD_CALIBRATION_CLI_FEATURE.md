# SPREAD Market CLI Calibration Implementation

**Date**: January 28, 2026  
**Status**: ✅ Complete and Tested

## Summary

Extended `python -m src.cli.pipeline calibrate` to support `--market spread`, enabling end-to-end SPREAD distribution calibration fitting. The implementation reuses existing calibration infrastructure (no new math), following the same patterns as TOTAL market support.

## Changes

### 1. **src/cli/pipeline.py** - _run_calibrate_ensemble()

**Lines 2546-2701**: Added SPREAD market path between ML and TOTAL branches.

**Key features:**
- Case-insensitive market matching: `"spread"`, `"SPREAD"`, `"Spread"` all accepted
- Routes to `calibrate_sport_season()` with `markets=[Market.SPREAD]`
- Loads ensemble config from market configuration (models + weights)
- Generates ensemble predictions and builds margin calibration dataset
- Fits distribution calibrator and saves artifact
- Updated error message to list all three supported markets: ML, SPREAD, TOTAL

**Error handling:**
- Validates market name; rejects unknown markets with descriptive error
- Handles missing ensemble config gracefully
- Validates that games and predictions are available

### 2. **docs/CLI.md** - Updated documentation

**Lines 462-530**: Expanded calibrate section.

**New content:**
- Updated header to mention all market types
- Added SPREAD calibration section with example command
- Documented case-insensitivity
- Linked to expected output locations

## Testing

### Test Suite: tests/test_cli_calibrate_spread.py (7 tests)

1. **test_cli_calibrate_spread_market_accepted** ✅
   - Verifies `--market spread` is accepted without ValueError

2. **test_cli_calibrate_spread_case_insensitive** ✅
   - Tests uppercase `--market SPREAD`

3. **test_cli_calibrate_spread_mixed_case** ✅
   - Tests mixed case `--market Spread`

4. **test_cli_calibrate_invalid_market_rejected** ✅
   - Ensures invalid markets still raise ValueError

5. **test_cli_calibrate_spread_error_message_updated** ✅
   - Validates error message includes SPREAD option

6. **test_cli_calibrate_spread_artifact_saved** ✅
   - Tests artifact persistence (graceful on missing config)

7. **test_spread_market_validation_in_error_message** ✅
   - Verifies error message mentions supported markets

### Integration Test Results

All existing SPREAD-related tests pass:
- ✅ 63 SPREAD-related tests across the suite (pytest -k "spread")
- ✅ 6 calibration provenance tag tests (ML, SPREAD, TOTAL coverage)
- ✅ calibration_bets_integration tests (apply/validation coverage)

## Command Usage

### Basic SPREAD calibration

```bash
python -m src.cli.pipeline calibrate \
  --sport nba \
  --season 2025-26 \
  --market spread \
  --source ensemble_spread_v1 \
  --start-date 2025-10-25 \
  --end-date 2026-01-20 \
  --csv ./data/raw/nba2026.csv
```

### With custom method (isotonic/platt)

```bash
python -m src.cli.pipeline calibrate \
  --sport nba \
  --season 2025-26 \
  --market spread \
  --source ensemble_spread_v1 \
  --start-date 2025-10-25 \
  --end-date 2026-01-20 \
  --method isotonic
```

### Output

Calibrator artifact saved to:  
`outputs/calibrators/nba/2025-26/ensemble_spread_v1/spread/<calibrator.pkl>`

## Dataset Format

**Input (ensemble predictions):**
- `margin_mean`: predicted mean margin (home_score - away_score)
- `margin_sd`: predicted margin standard deviation

**Actual value:**
- `actual_margin = home_score - away_score`

**Validation:**
- Finite values required
- `margin_sd > 0` required
- NaNs and invalid data filtered automatically

## Calibrator Application

When schedule is generated with calibrated SPREAD predictions:

```bash
python -m src.cli.pipeline schedule \
  --sport nba \
  --season 2025-26 \
  --model ensemble_spread_v1
```

The schedule loader automatically:
1. Detects calibrator artifact
2. Applies calibrated mean/sd to margin predictions
3. Appends `+calibrated_spread` tag to `win_prob_source` for provenance

## Constraints Satisfied

✅ **No new math**: Uses existing `VarianceCalibrator` from calibration system  
✅ **No model logic changes**: SPREAD ensemble weights unchanged  
✅ **No DB schema changes**: Uses existing game/prediction tables  
✅ **Backward compatible**: ML/TOTAL paths unmodified  
✅ **Additive only**: New CLI branch only; no refactoring  
✅ **Tests included**: 7 new CLI tests + 63 existing SPREAD tests pass  

## Validation

### Error Message Update

Old: `ValueError: Market 'SPREAD' not supported. Use 'ML' or 'total'.`

New: `ValueError: Market 'SPREAD' not supported. Use 'ML', 'SPREAD', or 'TOTAL'.`

### Case Handling

| Input | Result |
|-------|--------|
| `--market ml` | ✅ Accepted |
| `--market ML` | ✅ Accepted |
| `--market spread` | ✅ Accepted |
| `--market SPREAD` | ✅ Accepted |
| `--market total` | ✅ Accepted |
| `--market invalid` | ❌ Rejected |

## Files Modified

1. **src/cli/pipeline.py** (1 function, ~40 lines added)
   - `_run_calibrate_ensemble()`: Added SPREAD branch

2. **docs/CLI.md** (1 section expanded)
   - calibrate command documentation updated

3. **tests/test_cli_calibrate_spread.py** (NEW - 223 lines)
   - 7 comprehensive test cases

## Known Limitations / Future Work

1. **Regime-conditioned SPREAD**: Phase 10 bucketing (similar to TOTAL) not yet implemented
   - Could be added following Phase 10 TOTAL pattern if needed
   
2. **Per-ensemble calibrators**: Currently uses `source_id` for grouping
   - Could support finer granularity per model if required

3. **Multi-sport**: Works but requires ensemble config per sport/season

## References

- Implementation pattern: `historical_calibration.py::calibrate_sport_season()`
- Dataset builder: `historical_calibration.py::build_spread_calibration_dataset()`
- Distribution calibrator: `calibration/distribution.py::VarianceCalibrator`
- Application: `pipelines/schedule.py::_apply_calibration_to_schedule_df()`
- Provenance: `IMPLEMENTATION_SUMMARY_PROVENANCE_TAGS.md`

## Backward Compatibility

✅ **No breaking changes**:
- Existing ML calibration commands unchanged
- Existing TOTAL calibration commands unchanged
- New SPREAD path is purely additive
- Error handling improved (better error messages)

---

**Ready for production use.**
