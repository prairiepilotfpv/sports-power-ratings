## Calibration Integration - Validation Checklist

Use this checklist to verify the end-to-end calibration integration is working correctly.

---

## Pre-Validation Setup

### Verify Installation
```bash
# Confirm calibration modules exist
ls -la src/calibration/
# Should show: distribution.py, historical_calibration.py, standalone_cli.py, apply_calibrators.py, io.py

# Run tests
make test
# Should pass: test_calibration_bets_integration.py tests
```

---

## Step 1: Generate Calibrators for NBA 2025-26

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

### Validation Checklist for Step 1:
- [ ] Command runs without errors
- [ ] Output shows "Generating calibration data for basketball" (or similar)
- [ ] Output shows "Fitting calibrators..."
- [ ] Three calibrators are saved:
  ```bash
  ls -la outputs/calibrators/nba/2025-26/historical/
  # Should show: ML/, spread/, total/
  ```
- [ ] Each market has calibrator files:
  ```bash
  ls -la outputs/calibrators/nba/2025-26/historical/ML/
  # Should show: calibrator.pkl, metadata.json
  ```
- [ ] Metadata shows correct market:
  ```bash
  cat outputs/calibrators/nba/2025-26/historical/ML/metadata.json | grep -i market
  # Should show: "market": "ML"
  ```

---

## Step 2: Generate Schedule with Calibrated BETS Data

```bash
python -m src.cli.pipeline schedule \
    --sport nba \
    --season 2025-26 \
    --model ensemble_ml_v1 \
    --output schedule_nba_2025-26.xlsx
```

### Validation Checklist for Step 2:
- [ ] Command runs without errors
- [ ] Output shows "Building schedule..." and "Writing schedule..."
- [ ] Output shows "Applied ML calibrator to X rows"
- [ ] Output shows "Applied SPREAD distribution calibrator to X rows"
- [ ] Output shows "Applied TOTAL distribution calibrator to X rows"
- [ ] Excel file created: `schedule_nba_2025-26.xlsx`
- [ ] File size is reasonable (not empty, >1MB)

---

## Step 3: Verify BETS Sheet Data

### Using Python to Inspect BETS Sheet

```python
import pandas as pd

# Load the schedule
df = pd.read_excel("schedule_nba_2025-26.xlsx", sheet_name="BETS")

# Check ML market rows
ml_rows = df[df["market"] == "ML"]
print(f"ML rows: {len(ml_rows)}")
print(ml_rows[["game_id", "home_team", "away_team", "home_win_prob"]].head())

# Check SPREAD market rows
spread_rows = df[df["market"] == "spread"]
print(f"SPREAD rows: {len(spread_rows)}")
print(spread_rows[["game_id", "home_team", "away_team", "margin_mean", "margin_sd"]].head())

# Check TOTAL market rows
total_rows = df[df["market"] == "total"]
print(f"TOTAL rows: {len(total_rows)}")
print(total_rows[["game_id", "home_team", "away_team", "total_mean", "total_sd"]].head())
```

### Validation Checklist for Step 3:
- [ ] BETS sheet loads without errors
- [ ] All three markets present (ML, spread, total)
- [ ] ML market rows have valid home_win_prob values (0.0 to 1.0)
- [ ] SPREAD market rows have margin_mean and margin_sd values
- [ ] TOTAL market rows have total_mean and total_sd values
- [ ] No NaN values in probability/distribution columns (unless expected)
- [ ] Calibrated values are different from raw (if raw available for comparison)

---

## Step 4: Verify Calibration Actually Applied

```python
import pandas as pd

df = pd.read_excel("schedule_nba_2025-26.xlsx", sheet_name="BETS")

# Check if calibration columns exist (they should be merged into base columns)
ml_rows = df[df["market"] == "ML"]

# These should exist if calibration was applied:
if "home_win_prob_raw" in ml_rows.columns:
    print("✓ Raw home_win_prob stored")
    print(ml_rows[["home_win_prob_raw", "home_win_prob"]].head())
else:
    print("? No home_win_prob_raw column (may still have calibration applied)")

if "win_prob_source" in ml_rows.columns:
    print("✓ win_prob_source column exists")
    print(ml_rows["win_prob_source"].unique())
```

### Validation Checklist for Step 4:
- [ ] home_win_prob values look reasonable (0.3 to 0.7 typical range)
- [ ] home_win_prob + away_win_prob values look reasonable (check a few rows)
- [ ] margin_mean values reasonable (typically -5 to +10)
- [ ] margin_sd values reasonable (typically 0.5 to 3.0)
- [ ] total_mean values reasonable (for NBA: 200-230)
- [ ] total_sd values reasonable (typically 3.0 to 8.0)

---

## Step 5: Check Logs for Calibration Application

```bash
# Re-run schedule with verbose logging to see calibration steps
python -m src.cli.pipeline schedule \
    --sport nba \
    --season 2025-26 \
    --model ensemble_ml_v1 \
    --output schedule_nba_2025-26_v2.xlsx \
    2>&1 | grep -i calibrat
```

### Expected Log Output
```
Applied ML calibrator to 245 rows
Applied SPREAD distribution calibrator to 245 rows
Applied TOTAL distribution calibrator to 245 rows
```

### Validation Checklist for Step 5:
- [ ] Logs show "Applied ML calibrator to X rows" (X > 0)
- [ ] Logs show "Applied SPREAD distribution calibrator to X rows" (X > 0)
- [ ] Logs show "Applied TOTAL distribution calibrator to X rows" (X > 0)
- [ ] No "calibration failed" warnings
- [ ] No "Calibrator not found" warnings (unless expected)

---

## Step 6: Multi-Sport Validation

If you have other sports data:

### For NFL
```bash
python -m calibration.standalone_cli \
    --db data/db/nfl/2025-26.db \
    --sport nfl \
    --season 2025-26 \
    --models bradley-terry elo \
    --markets ML SPREAD TOTAL

python -m src.cli.pipeline schedule \
    --sport nfl \
    --season 2025-26 \
    --model ensemble_ml_v1 \
    --output schedule_nfl_2025-26.xlsx
```

### Validation Checklist for Step 6:
- [ ] Calibration runs for NFL
- [ ] Calibrators saved to: `outputs/calibrators/nfl/2025-26/...`
- [ ] Schedule generation for NFL works
- [ ] BETS sheet for NFL has calibrated values
- [ ] Calibrated values are different from NBA values (sport-specific)

---

## Step 7: Fallback Testing (No Calibrators)

```bash
# Temporarily move calibrators
mv outputs/calibrators outputs/calibrators.backup

# Run schedule - should work with raw predictions
python -m src.cli.pipeline schedule \
    --sport nba \
    --season 2025-26 \
    --model ensemble_ml_v1 \
    --output schedule_nba_no_calibrators.xlsx

# Check logs for "Calibrator not found" warnings
# Should see warnings but schedule completes normally

# Restore calibrators
mv outputs/calibrators.backup outputs/calibrators
```

### Validation Checklist for Step 7:
- [ ] Schedule runs without calibrators (doesn't crash)
- [ ] Logs show warnings about missing calibrators
- [ ] BETS sheet is generated with raw predictions
- [ ] File size is similar to calibrated version
- [ ] No errors in output

---

## Step 8: Performance Check

```bash
import time

# Time the schedule generation with calibrators
start = time.time()
# Run schedule command
end = time.time()

print(f"Schedule generation took {end - start:.2f} seconds")
```

### Validation Checklist for Step 8:
- [ ] Schedule generation takes < 5 minutes (typical: 30-120 seconds)
- [ ] No significant slowdown compared to non-calibrated version
- [ ] Memory usage is reasonable (< 2GB for typical datasets)

---

## Step 9: Manual Spot-Check

Open the Excel file and manually verify:

### BETS Sheet - ML Market
1. Find a row where market = "ML"
2. Check: home_win_prob value is between 0 and 1
3. Check: home_win_prob + away_win_prob ≈ 1.0 (allow ±0.01 floating point error)

### BETS Sheet - SPREAD Market
1. Find a row where market = "spread"
2. Check: margin_mean is reasonable (e.g., -5 to +15 for NBA)
3. Check: margin_sd is positive and reasonable (e.g., 0.5 to 3.0)

### BETS Sheet - TOTAL Market
1. Find a row where market = "total"
2. Check: total_mean is reasonable (e.g., 200-230 for NBA)
3. Check: total_sd is positive and reasonable (e.g., 3.0 to 8.0)

### Validation Checklist for Step 9:
- [ ] ML market probabilities look valid
- [ ] SPREAD market distributions look valid
- [ ] TOTAL market distributions look valid
- [ ] No obviously wrong values (e.g., probability > 1, SD = 0)

---

## Troubleshooting

### Issue: "Calibrator not found" warnings

**Cause**: Calibrators not generated yet

**Solution**:
```bash
# Run calibration first
python -m calibration.standalone_cli \
    --db data/db/nba/2025-26.db \
    --sport nba \
    --season 2025-26 \
    --models bradley-terry elo toor \
    --markets ML SPREAD TOTAL
```

### Issue: "ML calibration failed" error

**Cause**: Probability calibrator has wrong interface

**Solution**: Check calibrator metadata:
```bash
cat outputs/calibrators/nba/2025-26/historical/ML/metadata.json
# Should show: "method": "platt" or "isotonic"
```

### Issue: "SPREAD calibration failed" error

**Cause**: Distribution calibrator has wrong shape

**Solution**: Check calibrator type:
```bash
cat outputs/calibrators/nba/2025-26/historical/spread/metadata.json
# Should show: "method": "marginal_distribution"
```

### Issue: BETS sheet has raw predictions, not calibrated

**Cause**: Calibrators not being loaded

**Solution**: Verify calibrators exist:
```bash
ls -la outputs/calibrators/nba/2025-26/historical/*/
# Should show calibrator.pkl in each market directory
```

---

## Success Criteria

✅ **All of the following must be true**:

1. Calibration command runs successfully
2. Three calibrator files saved (ML, spread, total)
3. Schedule command runs successfully
4. Schedule logs show calibrators applied
5. BETS sheet has all three markets
6. All probability/distribution values are valid
7. No crashes or unhandled exceptions
8. Performance is acceptable

---

## Rollback Plan

If issues found:

```bash
# Option 1: Restore backup
mv outputs/calibrators.backup outputs/calibrators

# Option 2: Regenerate calibrators
rm -rf outputs/calibrators/nba/2025-26/
python -m calibration.standalone_cli \
    --db data/db/nba/2025-26.db \
    --sport nba \
    --season 2025-26 \
    --models bradley-terry elo toor \
    --markets ML SPREAD TOTAL \
    --method auto

# Option 3: Run schedule without calibrators (remove calibrators directory)
mv outputs/calibrators outputs/calibrators.disabled
python -m src.cli.pipeline schedule --sport nba --season 2025-26 --model ensemble_ml_v1
```

---

## Notes

- First time running calibration may take 2-5 minutes (database scan + model prediction generation)
- Subsequent calibrations are faster (incremental updates if available)
- Calibrators are sport-specific and season-specific
- Different models can share calibrators if using same historical data
