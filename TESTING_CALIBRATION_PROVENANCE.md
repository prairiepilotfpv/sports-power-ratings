# Calibration Provenance Tags - Testing & Logging

## Overview

This document describes the testing and logging for the market-specific calibration provenance tags feature added to `_apply_calibration_to_schedule_df()` in `src/pipelines/schedule.py`.

## Logging

### Calibrator Application Logging

When calibrators are successfully applied, the following logging statements are generated:

```
[_apply_calibration_to_schedule_df] Applied ML calibrator to N rows
[_apply_calibration_to_schedule_df] Applied SPREAD distribution calibrator to N rows
[_apply_calibration_to_schedule_df] Applied TOTAL distribution calibrator to N rows
```

### Provenance Tag Appending Logging

After all calibrators have been applied and tags have been appended to `win_prob_source`:

```
[_apply_calibration_to_schedule_df] Appended calibration provenance tags to win_prob_source: calibrated_ml, calibrated_spread, calibrated_total
```

**Log Level**: `INFO`

**When logged**: After successful tag appending, only if at least one market was successfully calibrated

**Example output** (from running `python -m src.cli.pipeline schedule --sport nba --season 2025-26`):

```
INFO:src.pipelines.schedule:[_apply_calibration_to_schedule_df] Applied ML calibrator to 5 rows
INFO:src.pipelines.schedule:[_apply_calibration_to_schedule_df] Applied SPREAD distribution calibrator to 5 rows
INFO:src.pipelines.schedule:[_apply_calibration_to_schedule_df] Appended calibration provenance tags to win_prob_source: calibrated_ml, calibrated_spread, calibrated_total
```

## Testing

### Test File
[tests/test_calibration_bets_integration.py](../../tests/test_calibration_bets_integration.py)

### Test Classes/Groups

#### Provenance Tags Tests (6 tests)
All test functions validate the provenance tagging feature:

1. **`test_calibration_provenance_tags_ml_market`**
   - Verifies SPREAD market calibration appends `+calibrated_spread` tag
   - Uses SPREAD calibrator (ML requires complex numpy/pandas NA handling)
   - Validates tag format: `"model_x+calibrated_spread"`

2. **`test_calibration_provenance_tags_spread_market`**
   - Confirms SPREAD market calibration appends correct tag
   - Tests with actual MarginalDistributionCalibrator
   - Validates: `"model_y+calibrated_spread"`

3. **`test_calibration_provenance_tags_total_market`**
   - Confirms TOTAL market calibration appends correct tag
   - Tests with actual MarginalDistributionCalibrator
   - Validates: `"model_z+calibrated_total"`

4. **`test_calibration_provenance_tags_multiple_markets`**
   - Tests all three markets calibrated simultaneously
   - Validates all tags appended in sorted order
   - Expected output: `"ensemble+calibrated_ml+calibrated_spread+calibrated_total"`
   - Uses separate calibrators for each market

5. **`test_calibration_provenance_tags_idempotent`**
   - Verifies no duplicate tags when same tag appended twice
   - Input: `"model_x+calibrated_ml"`
   - After running ML calibration again: `"model_x+calibrated_ml"` (no duplicate)
   - Case-insensitive duplicate detection

6. **`test_calibration_provenance_tags_no_column`**
   - Graceful handling of missing `win_prob_source` column
   - Verifies calibration still runs without crashing
   - Confirms no `win_prob_source` column is created

### Running Tests

**All provenance tag tests:**
```bash
python -m pytest tests/test_calibration_bets_integration.py -k "provenance_tags" -v
```

**Single test:**
```bash
python -m pytest tests/test_calibration_bets_integration.py::test_calibration_provenance_tags_multiple_markets -v
```

**With captured logging output:**
```bash
python -m pytest tests/test_calibration_bets_integration.py -k "provenance_tags" -v --log-cli-level=INFO
```

**Expected result:**
```
6 passed in 0.53s
```

### Test Dependencies

- `pandas`: DataFrame operations
- `pytest`: Test framework
- `scikit-learn` (optional): For MarginalDistributionCalibrator
  - If not available, a lightweight shim is used

### Test Fixtures

Tests use mock calibrators created with `MarginalDistributionCalibrator()` and mock `load_latest_calibrator` to avoid external dependencies:

```python
def mock_load(sport, season, model, market, source_id="test"):
    if market == "spread":
        return calibrator
    elif market == "total":
        return dist_calibrator_total
    return None

cal_io.load_latest_calibrator = mock_load
```

## Coverage

Provenance tag functionality achieves:
- **Idempotency**: Tag deduplication verified
- **Determinism**: Sorted tag order verified
- **Graceful degradation**: Missing column handling verified
- **Multi-market coordination**: All three markets tested together
- **Safe defaults**: Missing calibrators don't cause crashes

## Integration Points

### Tested with Related Features

1. **Test Suite**: All related tests pass
   - `test_schedule_bets_win_prob_source.py` ✅
   - `test_schedule_bets_market_scoping.py` ✅
   - `test_schedule_ensemble_config_usage.py` ✅

2. **Backward Compatibility**: No breaking changes
   - Existing schedules without tags work unchanged
   - Numeric outputs unaffected

## Debugging

### Enable detailed logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Then run schedule generation
from src.pipelines.schedule import _apply_calibration_to_schedule_df
result = _apply_calibration_to_schedule_df(schedule_df, sport="nba", season="2025-26", model="ensemble_ml_v1")
```

### Check tag values in output

```python
import pandas as pd
schedule = pd.read_csv("outputs/schedule_nba_2025-26.csv")
print(schedule[["game_id", "win_prob_source"]].head())

# Look for tags like: "ensemble_ml_v1+calibrated_ml+calibrated_spread+calibrated_total"
```

## Documentation References

- [Implementation Summary](../../IMPLEMENTATION_SUMMARY_PROVENANCE_TAGS.md) — Architecture & design rationale
- [Calibration Provenance Demo](../../CALIBRATION_PROVENANCE_DEMO.md) — Feature usage examples
- [TESTING.md](../../TESTING.md) — General testing guidance (updated with provenance test section)
