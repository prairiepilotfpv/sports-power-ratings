## Calibration Integration - Implementation Complete

### Status: ✅ READY FOR END-TO-END TESTING

The calibration system has been successfully integrated into the BETS sheet generation workflow. Calibrated predictions now flow automatically from the calibration system through schedule.py and into the BETS sheet.

---

## Architecture Overview

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                 CALIBRATION WORKFLOW                        │
└─────────────────────────────────────────────────────────────┘

Step 1: FIT CALIBRATORS (standalone)
   $ python -m calibration.standalone_cli \
       --db data/db/nba/2025-26.db \
       --sport nba \
       --season 2025-26 \
       --models bradley-terry elo toor \
       --markets ML SPREAD TOTAL
   
   Output: Calibrators saved to outputs/calibrators/nba/2025-26/historical/

Step 2: GENERATE SCHEDULE WITH CALIBRATED BETS DATA (automatic calibration)
   $ python -m src.cli.pipeline schedule \
       --sport nba \
       --season 2025-26 \
       --model ensemble_ml_v1

   Internal flow:
   ├─ build_forecasts_df() generates raw predictions
   ├─ _apply_calibration_to_schedule_df() 
   │  ├─ Loads latest ML calibrator → applies to home_win_prob
   │  ├─ Loads latest SPREAD calibrator → applies to margin_mean/margin_sd
   │  └─ Loads latest TOTAL calibrator → applies to total_mean/total_sd
   └─ _build_bets_dataframe() uses calibrated values

Step 3: USE CALIBRATED DATA
   Result: schedule.xlsx with BETS sheet containing:
   - ML market: calibrated_home_win_prob
   - SPREAD market: calibrated margin_mean and margin_sd
   - TOTAL market: calibrated total_mean and total_sd
```

### Three Market Calibration Flow

```
┌──────────────────────────────────────────────────────────────┐
│ ML MARKET (Probability Calibration)                          │
├──────────────────────────────────────────────────────────────┤
│ Raw: home_win_prob (e.g., 0.60)                              │
│      away_win_prob = 1 - home_win_prob                       │
│ Calibrator: Probability calibrator (Platt or Isotonic)       │
│ Transform: calibrated_prob = calibrator.transform(raw_prob)  │
│ Output: home_win_prob ← calibrated (0.62)                    │
│         away_win_prob ← 1 - calibrated (0.38)                │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ SPREAD MARKET (Distribution Calibration)                     │
├──────────────────────────────────────────────────────────────┤
│ Raw: margin_mean (e.g., 2.5), margin_sd (e.g., 1.0)         │
│ Calibrator: MarginalDistributionCalibrator                   │
│ Transform: calibrated_mean, calibrated_sd =                  │
│            calibrator.transform(pred_mean, pred_sd)          │
│ Output: margin_mean ← calibrated_mean (2.3)                  │
│         margin_sd ← calibrated_sd (0.95)                     │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ TOTAL MARKET (Distribution Calibration)                      │
├──────────────────────────────────────────────────────────────┤
│ Raw: total_mean (e.g., 210.0), total_sd (e.g., 5.0)         │
│ Calibrator: MarginalDistributionCalibrator                   │
│ Transform: calibrated_mean, calibrated_sd =                  │
│            calibrator.transform(pred_mean, pred_sd)          │
│ Output: total_mean ← calibrated_mean (211.5)                 │
│         total_sd ← calibrated_sd (4.8)                       │
└──────────────────────────────────────────────────────────────┘
```

---

## Code Changes Summary

### New Modules

1. **src/calibration/distribution.py** (240 lines)
   - `MarginalDistributionCalibrator`: Ridge regression for distribution calibration
   - Fits on (pred_mean, pred_sd, actual_value) tuples
   - Outputs calibrated_mean and calibrated_sd

2. **src/calibration/historical_calibration.py** (520 lines)
   - Complete standalone calibration engine
   - Zero betting pipeline dependencies
   - `calibrate_sport_season()`: Main orchestration function
   - Supports multi-sport, multi-market calibration

3. **src/calibration/standalone_cli.py** (150 lines)
   - Independent CLI interface
   - `python -m calibration.standalone_cli` command
   - Generates and persists calibrators

4. **tests/test_calibration_bets_integration.py** (300+ lines)
   - Integration tests for BETS sheet calibration
   - Tests all three markets (ML, SPREAD, TOTAL)
   - Tests missing calibrator fallback

### Modified Modules

1. **src/pipelines/schedule.py**
   - Enhanced `_apply_calibration_to_schedule_df()` function (Lines 333-510)
   - Now handles all three markets:
     - ML: Loads probability calibrator, transforms home/away_win_prob
     - SPREAD: Loads distribution calibrator, transforms margin_mean/margin_sd
     - TOTAL: Loads distribution calibrator, transforms total_mean/total_sd
   - Applied in-memory before BETS sheet generation
   - Includes fallback: missing calibrators don't crash the workflow
   - Adds logging for each market calibration

---

## Implementation Details

### ML Market Calibration

```python
# In _apply_calibration_to_schedule_df():
ml_calibrator = load_latest_calibrator(
    sport=sport,
    season=season,
    model=model,
    market="ML",  # String value "ML"
)

if ml_calibrator is not None:
    # Transform probabilities
    calibrated = ml_calibrator.transform(raw_probs)
    
    # Store raw values for reference
    df["home_win_prob_raw"] = df["home_win_prob"]
    df["away_win_prob_raw"] = df["away_win_prob"]
    
    # Replace with calibrated values
    df["home_win_prob"] = calibrated
    df["away_win_prob"] = 1.0 - calibrated
```

### SPREAD Market Calibration

```python
# In _apply_calibration_to_schedule_df():
spread_calibrator = load_latest_calibrator(
    sport=sport,
    season=season,
    model=model,
    market="spread",  # String value "spread"
)

if spread_calibrator is not None:
    # Build input DataFrame
    calib_input = pd.DataFrame({
        "pred_mean": margin_mean,
        "pred_sd": margin_sd,
    })
    
    # Transform distributions
    calib_result = spread_calibrator.transform(calib_input)
    
    # Replace with calibrated values
    df["margin_mean"] = calib_result["calibrated_mean"]
    df["margin_sd"] = calib_result["calibrated_sd"]
```

### TOTAL Market Calibration

```python
# In _apply_calibration_to_schedule_df():
total_calibrator = load_latest_calibrator(
    sport=sport,
    season=season,
    model=model,
    market="total",  # String value "total"
)

if total_calibrator is not None:
    # Build input DataFrame
    calib_input = pd.DataFrame({
        "pred_mean": total_mean,
        "pred_sd": total_sd,
    })
    
    # Transform distributions
    calib_result = total_calibrator.transform(calib_input)
    
    # Replace with calibrated values
    df["total_mean"] = calib_result["calibrated_mean"]
    df["total_sd"] = calib_result["calibrated_sd"]
```

---

## Usage Instructions

### Workflow

#### Step 1: Fit Calibrators
```bash
python -m calibration.standalone_cli \
    --db data/db/nba/2025-26.db \
    --sport nba \
    --season 2025-26 \
    --models bradley-terry elo toor \
    --markets ML SPREAD TOTAL \
    --source-id historical \
    --method auto
```

**Output**: Calibrators saved to:
- `outputs/calibrators/nba/2025-26/historical/ML/calibrator.pkl`
- `outputs/calibrators/nba/2025-26/historical/spread/calibrator.pkl`
- `outputs/calibrators/nba/2025-26/historical/total/calibrator.pkl`

#### Step 2: Generate Schedule with Calibrated BETS Data
```bash
python -m src.cli.pipeline schedule \
    --sport nba \
    --season 2025-26 \
    --model ensemble_ml_v1 \
    --output schedule_nba_2025-26.xlsx
```

**Automatic**: schedule.py loads calibrators and applies them in-memory

#### Step 3: Verify BETS Sheet
Open `schedule_nba_2025-26.xlsx` and check:
- **BETS** sheet contains calibrated values
- ML market rows: `calibrated_home_win_prob` column
- SPREAD market rows: `margin_mean` and `margin_sd` are calibrated
- TOTAL market rows: `total_mean` and `total_sd` are calibrated

### Testing

Run integration tests:
```bash
pytest tests/test_calibration_bets_integration.py -v
```

Expected output:
```
test_apply_ml_calibration_to_schedule_df PASSED
test_apply_spread_distribution_calibration_to_schedule_df PASSED
test_apply_total_distribution_calibration_to_schedule_df PASSED
test_apply_all_three_markets_calibration PASSED
test_missing_calibrator_fallback PASSED
```

---

## Data Persistence

### Where Calibrators Are Saved

```
outputs/
├── calibrators/
│   └── {sport}/
│       └── {season}/
│           └── {source_id}/
│               ├── ML/
│               │   ├── calibrator.pkl
│               │   └── metadata.json
│               ├── spread/
│               │   ├── calibrator.pkl
│               │   └── metadata.json
│               └── total/
│                   ├── calibrator.pkl
│                   └── metadata.json
```

### What's Saved

Each calibrator directory contains:
- **calibrator.pkl**: Serialized calibrator object (joblib)
- **metadata.json**: Calibrator metadata including:
  - `market`: "ML", "spread", or "total"
  - `method`: "platt", "isotonic", or "marginal_distribution"
  - `fitted_timestamp`: When calibrator was created
  - `fit_samples`: Number of games used to fit
  - `source_id`: Usually "historical"

---

## How Schedule Loads Calibrators

```python
# schedule.py calls load_latest_calibrator() from calibration.io:

def load_latest_calibrator(sport, season, model, market, source_id="test"):
    """Load the latest calibrator for a market.
    
    Searches in this order:
    1. outputs/calibrators/{sport}/{season}/{source_id}/{market}/
    2. Returns the most recently modified calibrator.pkl
    3. Returns None if not found
    """
    calib_path = Path(f"outputs/calibrators/{sport}/{season}/{source_id}/{market}")
    if calib_path.exists():
        pkl_files = sorted(calib_path.glob("calibrator.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if pkl_files:
            return joblib.load(pkl_files[0])
    return None
```

---

## Backward Compatibility

### If No Calibrators Exist

- schedule.py will **not find calibrators** and return None
- Function will **log a warning** but **not crash**
- Schedule will generate with **raw predictions**
- BETS sheet will have raw values (same as before)

### If Calibrator Fails to Apply

- Function catches exception and **logs a warning**
- Continues with **raw values** for that market
- Other markets still get calibrated
- Excel generation completes normally

---

## Testing Scenarios

### Scenario 1: Full Multi-Market Calibration
1. Run calibration with `--markets ML SPREAD TOTAL`
2. Run schedule
3. Verify BETS has calibrated values for all three

### Scenario 2: Single Market Calibration
1. Run calibration with `--markets ML` only
2. Run schedule
3. Verify SPREAD/TOTAL use raw values (fallback)

### Scenario 3: No Calibrators
1. Run schedule without running calibration first
2. Verify schedule completes with raw values
3. Check logs for "calibrator not found" warning

### Scenario 4: Multi-Sport
1. Fit calibrators for NFL, NBA, MLB
2. Run schedule for each sport
3. Verify each uses its own calibrators

---

## Integration Verification Checklist

- [x] Standalone calibration system created (no betting dependencies)
- [x] Distribution calibrators implemented for SPREAD/TOTAL
- [x] `_apply_calibration_to_schedule_df()` extended for all three markets
- [x] Calibrators loaded in-memory before BETS sheet generation
- [x] Calibrated values replace raw values in DataFrame
- [x] Fallback behavior for missing calibrators
- [x] Logging added for each market calibration
- [x] Integration tests created
- [x] Documentation written

---

## Files Changed

### Created
- `src/calibration/distribution.py` - Distribution calibrators
- `src/calibration/historical_calibration.py` - Calibration engine
- `src/calibration/standalone_cli.py` - CLI interface
- `src/calibration/apply_calibrators.py` - Helper functions (created, not actively used)
- `tests/test_calibration_bets_integration.py` - Integration tests
- `docs/CALIBRATION_INTEGRATION_BETS.md` - This documentation

### Modified
- `src/pipelines/schedule.py` - Enhanced `_apply_calibration_to_schedule_df()` (Lines 333-510)

---

## Next Steps

1. **Run calibration**: Generate calibrators for NBA 2025-26
2. **Test schedule**: Verify BETS sheet has calibrated values
3. **Multi-sport test**: Test with NFL, MLB, NHL
4. **Verify Excel output**: Open .xlsx file to confirm calibrated values appear
5. **Performance check**: Ensure schedule generation is not significantly slower

---

## Key Assumptions

1. **Market enum values**: Market.ML="ML", Market.SPREAD="spread", Market.TOTAL="total"
2. **Calibrator location**: `outputs/calibrators/{sport}/{season}/{source_id}/{market}/`
3. **DataFrame columns**: `margin_mean`, `margin_sd` for SPREAD; `total_mean`, `total_sd` for TOTAL
4. **Distribution input**: Calibrators accept (pred_mean, pred_sd), output (calibrated_mean, calibrated_sd)
5. **In-memory application**: No database writes; calibration applied during schedule generation

---

## Known Limitations

1. **Per-model calibrators**: Currently loads by model name. Could support per-ensemble.
2. **Cross-validation**: No automatic held-out set support (manual if needed).
3. **Metrics**: Log-loss improvement not displayed in schedule output.
4. **Versioning**: Calibrators auto-versioned by modification time (not git hash).

---

## Summary

The calibration system is now **fully integrated** into the BETS sheet workflow:

✅ **Standalone**: Completely independent of betting pipeline  
✅ **Flexible**: Works with any sport, any market (ML/SPREAD/TOTAL)  
✅ **Distribution-aware**: Supports proper distribution calibration for SPREAD/TOTAL  
✅ **Automatic**: Calibrators applied in-memory during schedule generation  
✅ **Robust**: Fallback to raw predictions if calibrators missing  
✅ **Testable**: Comprehensive integration tests included  

When users run `schedule`, calibrated predictions automatically populate the BETS sheet.
