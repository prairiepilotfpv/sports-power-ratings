# Logging & Testing Updates Summary

## Overview

Comprehensive logging and testing infrastructure has been added for the market-specific calibration provenance tags feature in `src/pipelines/schedule.py`.

## Logging Enhancements

### 1. Code Location: [src/pipelines/schedule.py](src/pipelines/schedule.py#L520)

Added INFO-level logging statement when provenance tags are appended:

```python
_LOG.info(
    f"[_apply_calibration_to_schedule_df] Appended calibration provenance tags to "
    f"win_prob_source: {tags_list}"
)
```

**When logged**: After all market calibrations, only if:
- `win_prob_source` column exists
- At least one market was successfully calibrated

**Example output**:
```
INFO src.pipelines.schedule:[_apply_calibration_to_schedule_df] 
     Appended calibration provenance tags to win_prob_source: 
     calibrated_ml, calibrated_spread, calibrated_total
```

### 2. Existing Calibrator Logging

Existing logging statements are preserved and continue to log:
- When each calibrator is successfully applied
- How many rows were calibrated per market
- Calibration failures (warning level)

## Testing Updates

### 1. Test Module Documentation

Enhanced module docstring in [tests/test_calibration_bets_integration.py](tests/test_calibration_bets_integration.py#L1-L27):
- Clearly describes what the module tests
- Breaks down testing categories (provenance tags, calibration, BETS integration)
- References architecture documentation

### 2. Test Function Documentation

Added comprehensive docstrings to each provenance tag test explaining:
- What aspect is being tested
- Expected behavior
- References to architecture documentation

Example:
```python
def test_calibration_provenance_tags_multiple_markets():
    """Test that multiple market calibrations append all relevant tags.
    
    Validates:
    - All three markets can be calibrated simultaneously
    - Tags are appended in sorted order for deterministic output
    - Multiple calibrators coordinate correctly
    """
```

### 3. General Testing Documentation

Updated [TESTING.md](TESTING.md#L24-L45):

Added new section "Calibration Provenance Tags Tests":
```bash
# Run all calibration provenance tests
python -m pytest tests/test_calibration_bets_integration.py -k "provenance_tags" -v

# Run specific test
python -m pytest tests/test_calibration_bets_integration.py::test_calibration_provenance_tags_multiple_markets -v
```

Documented what the tests verify:
- ML calibration appends correct tag
- SPREAD calibration appends correct tag
- TOTAL calibration appends correct tag
- Multiple market calibrations coordinate
- Tags are idempotent (no duplicates)
- Missing win_prob_source handled gracefully

## New Documentation Files

### 1. [TESTING_CALIBRATION_PROVENANCE.md](TESTING_CALIBRATION_PROVENANCE.md)

Comprehensive guide covering:
- **Logging section**: Complete logging reference with examples
- **Testing section**: Detailed description of all 6 provenance tag tests
- **Running tests**: Multiple ways to run tests with examples
- **Test dependencies**: Lists required packages
- **Coverage**: What aspects are tested and verified
- **Integration points**: How this integrates with related features
- **Debugging**: How to troubleshoot and inspect output

### 2. [IMPLEMENTATION_SUMMARY_PROVENANCE_TAGS.md](IMPLEMENTATION_SUMMARY_PROVENANCE_TAGS.md)

Architecture and implementation details (created previously):
- Code changes and modifications
- Idempotency logic
- Tag appending rules
- Test results summary
- Design rationale

### 3. [CALIBRATION_PROVENANCE_DEMO.md](CALIBRATION_PROVENANCE_DEMO.md)

Feature usage examples (created previously):
- Overview of tags
- Before/after examples
- Idempotency examples
- Implementation details
- Testing section

## Test Coverage

### Test Count
- **Provenance tag tests**: 6 tests
- All tests passing ✅

### Test Categories

1. **Single market calibration** (3 tests)
   - ML market test
   - SPREAD market test
   - TOTAL market test

2. **Multi-market coordination** (1 test)
   - Multiple markets calibrated simultaneously

3. **Edge cases** (2 tests)
   - Idempotency (no duplicate tags)
   - Missing win_prob_source column

### Test Execution

**All provenance tag tests**:
```bash
python -m pytest tests/test_calibration_bets_integration.py -k "provenance_tags" -v
# Result: 6 passed in 0.53s ✅
```

**With logging enabled**:
```bash
python -m pytest tests/test_calibration_bets_integration.py -k "provenance_tags" -v --log-cli-level=INFO
# Shows: Applied SPREAD/TOTAL distribution calibrators
#        Appended calibration provenance tags to win_prob_source: ...
```

## Logging Examples

### Schedule Generation with Calibration

When running schedule generation with calibrators:

```bash
python -m src.cli.pipeline schedule --sport nba --season 2025-26
```

Console output includes:
```
INFO src.pipelines.schedule:[_apply_calibration_to_schedule_df] Applied SPREAD distribution calibrator to 85 rows
INFO src.pipelines.schedule:[_apply_calibration_to_schedule_df] Applied TOTAL distribution calibrator to 85 rows
INFO src.pipelines.schedule:[_apply_calibration_to_schedule_df] Appended calibration provenance tags to win_prob_source: calibrated_spread, calibrated_total
```

### Test Execution with Logging

```bash
pytest tests/test_calibration_bets_integration.py::test_calibration_provenance_tags_multiple_markets -v --log-cli-level=INFO
```

Output shows:
```
INFO src.pipelines.schedule:[_apply_calibration_to_schedule_df] Applied SPREAD distribution calibrator to 3 rows
INFO src.pipelines.schedule:[_apply_calibration_to_schedule_df] Applied TOTAL distribution calibrator to 3 rows
INFO src.pipelines.schedule:[_apply_calibration_to_schedule_df] Appended calibration provenance tags to win_prob_source: calibrated_spread, calibrated_total
```

## Backward Compatibility

✅ All logging and testing changes are:
- **Non-breaking**: Existing tests still pass
- **Additive**: New tests don't affect existing functionality
- **Transparent**: Logging is optional (configurable via log level)
- **Well-documented**: Clear references to features and architecture

## Summary of Changes

| File | Change | Status |
|------|--------|--------|
| `src/pipelines/schedule.py` | Added logging when tags appended | ✅ |
| `tests/test_calibration_bets_integration.py` | Enhanced module/function docstrings | ✅ |
| `TESTING.md` | Added provenance tags test section | ✅ |
| `TESTING_CALIBRATION_PROVENANCE.md` | New comprehensive testing guide | ✅ |
| `IMPLEMENTATION_SUMMARY_PROVENANCE_TAGS.md` | Architecture details (existing) | ✅ |
| `CALIBRATION_PROVENANCE_DEMO.md` | Feature examples (existing) | ✅ |

## Recent Calibration/Test Refinements

### Test Coverage Updates
- `tests/pipelines/test_schedule_bets_workbook.py` now derives ensemble probabilities from whichever key the component JSON exposes (`prob` or the normalized `value`), keeping the ML combination assertion aligned with what the pipeline writes.
- `tests/test_total_ensemble_bets_workflow.py` now exercises the new default-weight behavior so `_build_bets_dataframe` sees a concrete total mean even when no tuned weights are supplied.

### Distribution Calibration Behavior
- `src/calibration/distribution.py` now resolves `pred_mean`/`pred_sd` from `margin_*` or `total_*` aliases before computing calibrated outputs, ensuring the standalone calibration system test can run without schema changes.

## Quick Reference

**View logging for a run**:
```bash
python -m pytest tests/test_calibration_bets_integration.py -k "provenance_tags" --log-cli-level=INFO -v
```

**Run schedule with calibration**:
```bash
python -m src.cli.pipeline schedule --sport nba --season 2025-26 2>&1 | grep calibrat
```

**Check test documentation**:
- Full testing guide: `TESTING_CALIBRATION_PROVENANCE.md`
- General testing: `TESTING.md`
- Implementation: `IMPLEMENTATION_SUMMARY_PROVENANCE_TAGS.md`
- Usage examples: `CALIBRATION_PROVENANCE_DEMO.md`
