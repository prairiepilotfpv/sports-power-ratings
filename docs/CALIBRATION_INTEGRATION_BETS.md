## Calibration Integration with BETS Sheet - Implementation Summary

### Architecture

The calibration system is now fully integrated into the BETS sheet generation workflow:

```
1. User runs calibration standalone:
   $ python -m calibration.standalone_cli --db data/db/nba/2025-26.db ...
   ↓
   Saves calibrators to: outputs/calibrators/nba/2025-26/historical/{ML,spread,total}/

2. User runs schedule to generate BETS sheet:
   $ python -m src.cli.pipeline schedule --sport nba --season 2025-26 ...
   ↓
   schedule.py calls build_schedule_dataframe()
   ↓
   build_schedule_dataframe() calls _apply_calibration_to_schedule_df()
   ↓
   _apply_calibration_to_schedule_df() loads and applies calibrators:
      - ML: Calibrates home_win_prob and away_win_prob
      - SPREAD: Calibrates margin_mean and margin_sd  
      - TOTAL: Calibrates total_mean and total_sd
   ↓
   Calibrated values flow directly into BETS sheet generation
   ↓
   BETS sheet has calibrated data in all probability/distribution columns
```

### Key Changes Made

#### 1. Enhanced `_apply_calibration_to_schedule_df()` in `schedule.py`

**Before**: Only applied ML calibrators to home_win_prob

**After**: Applies calibrators for all three markets:

```python
# ML MARKET
- Loads ML calibrator
- Calibrates: home_win_prob → home_win_prob_calibrated
- Replaces home_win_prob with calibrated value

# SPREAD MARKET  
- Loads SPREAD calibrator (MarginalDistributionCalibrator)
- Calibrates: margin_mean, margin_sd
- Replaces in-place with calibrated distribution

# TOTAL MARKET
- Loads TOTAL calibrator (MarginalDistributionCalibrator)
- Calibrates: total_mean, total_sd
- Replaces in-place with calibrated distribution
```

### Data Flow Diagram

```
Historical Games (completed)
    ↓
[Calibration System]
    ├─ Load completed games
    ├─ Generate model predictions
    ├─ Fit calibrators for ML/SPREAD/TOTAL
    └─ Save to outputs/calibrators/
    ↓
[Schedule Generation]
    ├─ Build forecasts from models
    ├─ Load latest calibrators
    ├─ Apply calibrations:
    │   ├─ ML: transform(p_home_win) → calibrated_p_home_win
    │   ├─ SPREAD: transform(margin_mean, margin_sd) → calibrated_margin
    │   └─ TOTAL: transform(total_mean, total_sd) → calibrated_total
    └─ Update DataFrame in-place
    ↓
[BETS Sheet Generation]
    ├─ Read calibrated_home_win_prob for ML market rows
    ├─ Read calibrated_margin_mean/margin_sd for SPREAD market rows
    ├─ Read calibrated_total_mean/total_sd for TOTAL market rows
    └─ Populate BETS sheet with calibrated values
```

### What BETS Sheet Gets

When you run `schedule`, the BETS sheet is automatically populated with calibrated data:

| Market | Column | Source | Before Calibration | After Calibration |
|--------|--------|--------|-------------------|-------------------|
| ML | home_win_prob | ML calibrator | Raw model output | Calibrated probability |
| ML | away_win_prob | ML calibrator | 1 - raw | 1 - calibrated |
| SPREAD | margin_mean | SPREAD calibrator | Raw model prediction | Calibrated mean |
| SPREAD | margin_sd | SPREAD calibrator | Raw model prediction | Calibrated SD |
| TOTAL | total | TOTAL calibrator | Raw model prediction | Calibrated mean |
| TOTAL | total_sd | TOTAL calibrator | Raw model prediction | Calibrated SD |

### Usage Workflow

#### Step 1: Generate and fit calibrators
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

Output: Calibrators saved to `outputs/calibrators/nba/2025-26/historical/{ML,spread,total}/`

#### Step 2: Generate schedule with calibrated BETS data
```bash
python -m src.cli.pipeline schedule \
    --sport nba \
    --season 2025-26 \
    --model ensemble_ml_v1 \
    --output schedule_nba_2025-26.xlsx
```

This automatically:
- Loads calibrators from step 1
- Applies them to all predictions
- Bets sheet has calibrated values in all columns

### Implementation Details

#### Calibrator Loading
- Uses `load_latest_calibrator()` from `calibration.io`
- Searches `outputs/calibrators/{sport}/{season}/{source_id}/{market}/`
- Returns most recently modified calibrator file

#### ML Calibration
- Input: home_win_prob, away_win_prob
- Calibrator type: Probability calibrator (Platt or Isotonic)
- Transform: `calibrated_prob = calibrator.transform(raw_prob)`
- Output: Replaces home_win_prob and away_win_prob with calibrated values

#### SPREAD Calibration
- Input: margin_mean, margin_sd
- Calibrator type: MarginalDistributionCalibrator
- Transform: Fits `calibrated_mean = α * raw_mean + β`
- Output: Replaces margin_mean and margin_sd with calibrated values

#### TOTAL Calibration
- Input: total_mean, total_sd  
- Calibrator type: MarginalDistributionCalibrator
- Transform: Fits `calibrated_mean = α * raw_mean + β`
- Output: Replaces total_mean and total_sd with calibrated values

### Backward Compatibility

- If no calibrator exists → Uses raw predictions (no error)
- If calibration fails → Falls back to raw predictions (logged warning)
- Existing schedule without calibrators still works normally
- Old `calibrate-history` command still available (deprecated)

### Notes

1. **Distribution Calibrators**: The system detects if a calibrator is distribution-based by checking the metadata. Only distribution calibrators are applied to SPREAD/TOTAL.

2. **In-Memory Operations**: Calibration happens in-memory during schedule generation - no database writes needed (was considered but unnecessary since we apply during schedule generation).

3. **Multiple Calibration Runs**: Different `--source-id` values allow storing multiple calibration runs. Schedule loads the latest (by timestamp).

4. **Sport/Market Flexibility**: Works with any sport and any model. Calibrators are market-specific, allowing independent tuning per market type.

5. **Error Handling**: If a calibrator can't be loaded or applied, the function logs a warning and continues with raw predictions. This prevents schedule generation from failing.

### Troubleshooting

**Q: BETS sheet still has raw probabilities**
- A: Check that calibrators exist in `outputs/calibrators/{sport}/{season}/historical/`
- A: Verify filenames match the source_id used in calibration command

**Q: Calibration warnings in logs but no failures**
- A: This is normal - it means some markets don't have calibrators yet
- A: Only markets with fitted calibrators will be calibrated

**Q: Want to use different calibration run**
- A: Change the --source-id in calibration command to create separate run
- A: Schedule will automatically use the latest one

**Q: Want to turn off calibration for testing**
- A: Just rename/delete the `outputs/calibrators/` directory temporarily
- A: Schedule will generate with raw predictions

### Future Enhancements

1. **Per-Ensemble Calibration**: Currently loads calibrators by model name. Could support per-ensemble ID.
2. **Cross-Validation**: Add held-out set support for validation.
3. **Metrics Display**: Show log-loss improvement in schedule output.
4. **Calibration Versioning**: Auto-version calibrators with git hash or timestamp.
