# Market-Specific Calibration Provenance Tags

## Overview
When calibration for a market successfully runs in `_apply_calibration_to_schedule_df()`, the function appends a market-specific tag to `win_prob_source` to indicate which calibrators were applied.

## Tags

- **ML**: `+calibrated_ml` — appended when Platt/sigmoid calibration is applied to home/away win probabilities
- **SPREAD**: `+calibrated_spread` — appended when distribution calibration is applied to margin predictions
- **TOTAL**: `+calibrated_total` — appended when distribution calibration is applied to total predictions

## Example

### Input
```python
{
    "game_id": "game_1",
    "home_win_prob": 0.60,
    "away_win_prob": 0.40,
    "margin_mean": 2.5,
    "margin_sd": 1.0,
    "total_mean": 210.0,
    "total_sd": 5.0,
    "win_prob_source": "ensemble_ml_v1"
}
```

### After calibration (all three markets)
```python
{
    "game_id": "game_1",
    "home_win_prob": 0.62,  # calibrated
    "away_win_prob": 0.38,  # calibrated
    "margin_mean": 2.45,    # calibrated
    "margin_sd": 0.98,      # calibrated
    "total_mean": 211.2,    # calibrated
    "total_sd": 4.85,       # calibrated
    "win_prob_source": "ensemble_ml_v1+calibrated_ml+calibrated_spread+calibrated_total"
}
```

## Idempotency

If a tag is already present (case-insensitive), it will not be appended again:

```python
# Input
{
    "win_prob_source": "ensemble_ml_v1+calibrated_ml"
}

# After ML calibration runs again
{
    "win_prob_source": "ensemble_ml_v1+calibrated_ml"  # no duplicate
}
```

## Implementation Details

### Code Location
[src/pipelines/schedule.py](src/pipelines/schedule.py#L333-L490) — `_apply_calibration_to_schedule_df()` function

### Key Logic
1. Track which markets have calibrators successfully applied in a `calibrated_markets` set
2. Only add tags if `win_prob_source` column exists AND calibration ran successfully
3. Tags are appended in sorted order for deterministic output
4. Case-insensitive duplicate checks ensure idempotency

### When Tags Are Applied
- **NOT** when calibrator is missing (returns `None` from `load_latest_calibrator`)
- **NOT** when there are no valid prediction values to calibrate
- **NOT** when calibration transformation raises an exception
- Tags are applied only after successful transformation and numeric update

## Testing

Six test cases in [tests/test_calibration_bets_integration.py](tests/test_calibration_bets_integration.py):

1. `test_calibration_provenance_tags_ml_market` — SPREAD tag appended
2. `test_calibration_provenance_tags_spread_market` — SPREAD tag appended
3. `test_calibration_provenance_tags_total_market` — TOTAL tag appended
4. `test_calibration_provenance_tags_multiple_markets` — All three tags appended together
5. `test_calibration_provenance_tags_idempotent` — No duplicate tags
6. `test_calibration_provenance_tags_no_column` — Handles missing column gracefully

Run tests:
```bash
python -m pytest tests/test_calibration_bets_integration.py::test_calibration_provenance_tags_* -v
```

## Backward Compatibility

- No changes to numeric outputs or model behavior
- Only metadata (win_prob_source string) is modified
- Existing schedules without tags continue to work
- Empty win_prob_source is handled gracefully
