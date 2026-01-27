# Implementation Summary: Market-Specific Calibration Provenance Tags

## Scope
Modified `src/pipelines/schedule.py` to add market-specific calibration provenance tags to `win_prob_source`.

## Changes Made

### File: [src/pipelines/schedule.py](src/pipelines/schedule.py#L333-L520)

**Function**: `_apply_calibration_to_schedule_df()`

**Key modifications**:

1. **Line 351**: Added `calibrated_markets = set()` to track which markets had calibration successfully applied
   
2. **Lines 404, 419, 444**: Added `calibrated_markets.add("<MARKET>")` inside each market's try block, only after successful transformation
   - `calibrated_markets.add("ML")` for ML market (line 404)
   - `calibrated_markets.add("SPREAD")` for SPREAD market (line 419)
   - `calibrated_markets.add("TOTAL")` for TOTAL market (line 444)

3. **Lines 487-520**: New section to append market-specific tags to `win_prob_source`:
   - Only runs if `win_prob_source` column exists AND at least one market was successfully calibrated
   - Maps markets to tags: `{"ML": "calibrated_ml", "SPREAD": "calibrated_spread", "TOTAL": "calibrated_total"}`
   - Implements idempotent tag appending via `_append_calibration_tags()` helper function
   - Tags appended in sorted order for deterministic output
   - Case-insensitive duplicate detection prevents adding same tag twice

**Numeric outputs**: Unchanged — only metadata (win_prob_source) is modified

---

## Implementation Details

### Idempotency Logic

```python
def _append_calibration_tags(value: Any, tags: set[str]) -> str:
    """Append market-specific calibration tags idempotently to win_prob_source."""
    if value is None:
        text = ""
    else:
        text = str(value).strip()
    
    # Append each tag that hasn't been applied yet (case-insensitive check)
    for tag in sorted(tags):  # Sort for deterministic output
        if text and tag not in text.lower():
            text = f"{text}+{tag}"
        elif not text:
            text = tag
    
    return text
```

### Tag Appending Rules

- Tags only appended if calibrator was successfully applied (not skipped, not failed)
- Tags only appended if `win_prob_source` column exists
- Each tag appears at most once (case-insensitive check)
- Tags appear in sorted order: `calibrated_ml`, `calibrated_spread`, `calibrated_total`
- Example: `"ensemble_ml_v1" → "ensemble_ml_v1+calibrated_ml+calibrated_spread"`

---

## Tests

Added 6 comprehensive tests to [tests/test_calibration_bets_integration.py](tests/test_calibration_bets_integration.py#L329-L616):

1. **`test_calibration_provenance_tags_ml_market`** (line 329)
   - SPREAD calibration appends `+calibrated_spread`

2. **`test_calibration_provenance_tags_spread_market`** (line 356)
   - SPREAD calibration appends `+calibrated_spread`

3. **`test_calibration_provenance_tags_total_market`** (line 383)
   - TOTAL calibration appends `+calibrated_total`

4. **`test_calibration_provenance_tags_multiple_markets`** (line 410)
   - Multiple calibrations append all relevant tags in sorted order

5. **`test_calibration_provenance_tags_idempotent`** (line 530)
   - No duplicate tags when appending same tag twice

6. **`test_calibration_provenance_tags_no_column`** (line 567)
   - Gracefully handles missing `win_prob_source` column

**Test Results**: ✅ All 6 tests pass

```bash
pytest tests/test_calibration_bets_integration.py::test_calibration_provenance_tags_* -v
# Output: 6 passed in 0.62s
```

---

## Backward Compatibility

✅ **No breaking changes**:
- Only metadata (win_prob_source string) is modified
- Numeric outputs unchanged
- Existing schedules without tags continue to work
- Empty/missing win_prob_source handled gracefully
- No database schema changes
- No CLI changes

---

## Example

### Before Calibration
```python
{
    "game_id": "g1",
    "home_win_prob": 0.60,
    "margin_mean": 2.5,
    "total_mean": 210.0,
    "win_prob_source": "ensemble_ml_v1"
}
```

### After Calibration (all 3 markets)
```python
{
    "game_id": "g1",
    "home_win_prob": 0.62,  # calibrated
    "margin_mean": 2.45,    # calibrated
    "total_mean": 211.2,    # calibrated
    "win_prob_source": "ensemble_ml_v1+calibrated_ml+calibrated_spread+calibrated_total"
}
```

---

## Design Rationale

1. **Minimal diff**: Only 3 additions to track markets + 1 new section for tag appending
2. **Idempotent**: Prevents tag duplication even with repeated calibration runs
3. **Deterministic**: Sorted tag order ensures consistent output across runs
4. **Safe**: Only appends tags when calibration actually succeeded
5. **Metadata-only**: No impact on model outputs or numeric accuracy
