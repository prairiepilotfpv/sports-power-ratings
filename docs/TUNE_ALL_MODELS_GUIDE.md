# Tune All Models & Markets - Complete Guide

## Overview

The `tune_all_models_markets.py` script orchestrates hyperparameter tuning for **all models** across **all markets** in a single command.

**What it does:**
- Automatically loops through all models (bradley-terry, elo, gssd, poisson, toor)
- For each model, tunes across all 3 markets: ML, SPREAD, TOTAL
- Uses the correct metric per market (log_loss for ML, mae_margin for SPREAD, mae_total for TOTAL)
- Saves (activates) tuned parameters to the database immediately
- Provides real-time progress and summary reports

---

## Quick Start

### Basic Command
```bash
python scripts/tune_all_models_markets.py \
  --sport nba \
  --season 2025-26 \
  --csv data/raw/nba.csv \
  --start 2024-11-01 \
  --end 2024-12-31
```

**What happens:**
1. ✅ Validates CSV exists, dates are valid
2. ✅ Tunes `bradley-terry` on all 3 markets
3. ✅ Tunes `elo` on all 3 markets
4. ✅ Tunes `gssd` on all 3 markets
5. ✅ Tunes `poisson` on all 3 markets
6. ✅ Tunes `toor` on all 3 markets
7. ✅ Saves all tuned params to database
8. ✅ Displays summary at the end

---

## Command Options

### Required Arguments

```bash
--sport TEXT              Sport identifier (e.g., nba, nhl, nfl)
--season TEXT             Season identifier (e.g., 2025-26)
--csv PATH                CSV file with historical games for tuning
--start YYYY-MM-DD        Backtest start date
--end YYYY-MM-DD          Backtest end date
```

### Optional Arguments

```bash
--models TEXT             Comma-separated models to tune
                         Default: bradley-terry,elo,gssd,poisson,toor
                         Example: --models elo,bradley-terry

--market TEXT             Market(s) to tune (ML, SPREAD, TOTAL)
                         Default: all three markets
                         Example: --market ML,SPREAD

--window TYPE             Expanding or rolling window
                         Choices: expanding, rolling
                         Default: expanding

--rolling-days INT        Rolling window size in days
                         Required if --window=rolling
                         Example: --rolling-days 30

--rolling-games INT       Rolling window size in games
                         Alternative to --rolling-days
                         Example: --rolling-games 82

--grid-file PATH          Custom parameter grid (JSON)
                         Example: --grid-file custom_grid.json

--output-dir PATH         Override output directory
                         Default: outputs/tuning/<sport>/<season>/

--db PATH                 Override SQLite DB path
                         Default: data/db/<sport>/<season>.db

--jobs INT                Parallel jobs per model
                         Default: 1 (sequential)
                         Example: --jobs 4

--no-activate             Skip saving params to DB
                         Default: False (params are saved)

--allow-worse             Accept worse params than baseline
                         Default: False (only accept improvements)

--metric-overrides JSON   Market-to-metric override map
                         Example: '{"ML":"brier_score"}'
```

---

## Examples

### Example 1: Basic Tuning (All Models, All Markets)

```bash
python scripts/tune_all_models_markets.py \
  --sport nba \
  --season 2025-26 \
  --csv data/raw/nba.csv \
  --start 2024-11-01 \
  --end 2024-12-31
```

**Expected output:**
```
================================================================================
TUNE ALL MODELS ACROSS ALL MARKETS
================================================================================
Sport: nba
Season: 2025-26
Date range: 2024-11-01 to 2024-12-31
Window: expanding
Models to tune: bradley-terry, elo, gssd, poisson, toor
Activate params: yes
================================================================================

=== [1/5] Tuning model: bradley-terry ===
ML: metric=log_loss best_score=0.545 params_source=tuned activated=yes
SPREAD: metric=mae_margin best_score=2.34 params_source=tuned activated=yes
TOTAL: metric=mae_total best_score=4.12 params_source=tuned activated=yes
✓ Model bradley-terry tuning completed successfully.

=== [2/5] Tuning model: elo ===
ML: metric=log_loss best_score=0.542 params_source=tuned activated=yes
SPREAD: metric=mae_margin best_score=2.31 params_source=tuned activated=yes
TOTAL: metric=mae_total best_score=4.10 params_source=tuned activated=yes
✓ Model elo tuning completed successfully.

... (gssd, poisson, toor follow)

================================================================================
SUMMARY
================================================================================
✓ bradley-terry: success
✓ elo: success
✓ gssd: success
✓ poisson: success
✓ toor: success

Results: 5 succeeded, 0 failed (out of 5 total)
All models tuned successfully!
```

**Then you can immediately use:**
```bash
python -m src.cli.pipeline schedule \
  --sport nba \
  --season 2025-26 \
  --db data/db/nba/2025-26.db \
  --market-csv data/processed/nba/2025-26/lines.csv \
  --strict
```

---

### Example 2: Tune Specific Models Only

If you only want to tune `elo` and `bradley-terry`:

```bash
python scripts/tune_all_models_markets.py \
  --sport nba \
  --season 2025-26 \
  --csv data/raw/nba.csv \
  --start 2024-11-01 \
  --end 2024-12-31 \
  --models elo,bradley-terry
```

This will:
- Tune `elo` on all 3 markets
- Tune `bradley-terry` on all 3 markets
- Skip `gssd`, `poisson`, `toor`
- Save results to database

---

### Example 3: With Parallelization (Faster)

```bash
python scripts/tune_all_models_markets.py \
  --sport nba \
  --season 2025-26 \
  --csv data/raw/nba.csv \
  --start 2024-11-01 \
  --end 2024-12-31 \
  --jobs 4
```

This runs 4 parallel jobs **per model**, speeding up the grid search for each model.

---

### Example 4: Rolling Window Instead of Expanding

For a 30-day rolling window:

```bash
python scripts/tune_all_models_markets.py \
  --sport nba \
  --season 2025-26 \
  --csv data/raw/nba.csv \
  --start 2024-11-01 \
  --end 2024-12-31 \
  --window rolling \
  --rolling-days 30
```

---

### Example 5: Custom Parameter Grid

Create a file `custom_grid.json`:
```json
{
  "elo": {
    "k_factor": [15.0, 20.0, 25.0],
    "home_advantage": [40.0, 60.0, 80.0],
    "initial_rating": [1500.0]
  },
  "bradley-terry": {
    "temp": [2.5, 3.0, 3.5],
    "l2_lambda": [1e-3, 1e-2]
  }
}
```

Then run:
```bash
python scripts/tune_all_models_markets.py \
  --sport nba \
  --season 2025-26 \
  --csv data/raw/nba.csv \
  --start 2024-11-01 \
  --end 2024-12-31 \
  --grid-file custom_grid.json
```

---

### Example 6: Skip Activation (Dry Run)

To test without saving to database:

```bash
python scripts/tune_all_models_markets.py \
  --sport nba \
  --season 2025-26 \
  --csv data/raw/nba.csv \
  --start 2024-11-01 \
  --end 2024-12-31 \
  --no-activate
```

Results are still generated in `outputs/tuning/`, but **not** saved to the database. You can review them and activate later if desired.

---

### Example 7: Different Metrics Per Market

```bash
python scripts/tune_all_models_markets.py \
  --sport nba \
  --season 2025-26 \
  --csv data/raw/nba.csv \
  --start 2024-11-01 \
  --end 2024-12-31 \
  --metric-overrides '{"ML":"brier_score","SPREAD":"mae_margin"}'
```

This overrides the defaults:
- ML: Use `brier_score` instead of `log_loss`
- SPREAD: Use `mae_margin` (default, explicit)
- TOTAL: Use `mae_total` (default, unchanged)

---

### Example 8: Real-World NHL Example

```bash
python scripts/tune_all_models_markets.py \
  --sport nhl \
  --season 2025-26 \
  --csv data/raw/nhl_2025-26.csv \
  --start 2024-10-01 \
  --end 2025-01-25 \
  --jobs 4 \
  --output-dir outputs/nhl_jan2025
```

This:
- Tunes all models for NHL 2025-26 season
- Uses games from Oct 1, 2024 to Jan 25, 2025
- Runs 4 parallel jobs per model (faster)
- Saves artifacts to `outputs/nhl_jan2025/`
- Activates tuned params to database

---

## Step-by-Step Workflow

### Step 1: Prepare Your Data

Ensure you have a CSV file with historical game results:

```bash
ls -la data/raw/nba.csv
```

Required columns: `date`, `home_team`, `away_team`, `home_score`, `away_score`

---

### Step 2: Choose Your Date Range

Decide on a backtest period. Example: Last 2 months of games:

```bash
# Nov 1, 2024 to Dec 31, 2024
--start 2024-11-01 --end 2024-12-31
```

---

### Step 3: Run the Tuning Script

```bash
python scripts/tune_all_models_markets.py \
  --sport nba \
  --season 2025-26 \
  --csv data/raw/nba.csv \
  --start 2024-11-01 \
  --end 2024-12-31
```

This will take **5-30 minutes** depending on:
- CSV size
- Date range
- Number of games
- Your system specs

---

### Step 4: Monitor Progress

Watch the logs in real-time:
```
=== [1/5] Tuning model: bradley-terry ===
...
=== [2/5] Tuning model: elo ===
...
```

---

### Step 5: Verify Results

Check the summary:
```
================================================================================
SUMMARY
================================================================================
✓ bradley-terry: success
✓ elo: success
✓ gssd: success
✓ poisson: success
✓ toor: success

Results: 5 succeeded, 0 failed (out of 5 total)
All models tuned successfully!
```

---

### Step 6: View Tuned Parameters

Check what was activated:

```bash
python -m src.cli.pipeline show-active-params \
  --sport nba \
  --season 2025-26
```

Output example:
```
Model: elo
  ML (log_loss): k_factor=20.0, home_advantage=50.0, ...
  SPREAD (mae_margin): k_factor=25.0, home_advantage=55.0, ...
  TOTAL (mae_total): total_shrinkage=0.75, total_team_prior_games=15, ...

Model: bradley-terry
  ML (log_loss): temp=3.0, l2_lambda=1e-3, ...
  SPREAD (mae_margin): temp=3.2, l2_lambda=5e-4, ...
  TOTAL (mae_total): temp=3.1, l2_lambda=8e-4, ...
```

---

### Step 7: Use Tuned Params in Production

Now run schedule with your tuned parameters:

```bash
python -m src.cli.pipeline schedule \
  --sport nba \
  --season 2025-26 \
  --db data/db/nba/2025-26.db \
  --market-csv data/processed/nba/2025-26/lines.csv \
  --strict
```

The system automatically loads the tuned parameters you just created. ✅

---

## Troubleshooting

### Problem: "CSV file not found"
**Solution:** Verify the CSV path exists
```bash
ls -la data/raw/nba.csv
```

### Problem: "Invalid date format"
**Solution:** Use YYYY-MM-DD format
```bash
# ✓ Correct
--start 2024-11-01 --end 2024-12-31

# ✗ Wrong
--start 11/01/2024 --end 12/31/2024
```

### Problem: "No valid models specified"
**Solution:** Check model names are lowercase and comma-separated
```bash
# ✓ Correct
--models elo,bradley-terry,gssd

# ✗ Wrong
--models Elo Bradley-Terry GSSD
```

### Problem: Rolling window fails
**Solution:** Provide `--rolling-days` or `--rolling-games` when using `--window rolling`
```bash
# ✓ Correct
--window rolling --rolling-days 30

# ✗ Wrong
--window rolling
```

### Problem: Tuning takes too long
**Solution:** Add `--jobs` flag to parallelize
```bash
# Use 4 parallel jobs per model (much faster)
--jobs 4
```

### Problem: Want to run again but skip activation
**Solution:** Use `--no-activate` to test without persisting
```bash
python scripts/tune_all_models_markets.py \
  --sport nba \
  --season 2025-26 \
  --csv data/raw/nba.csv \
  --start 2024-11-01 \
  --end 2024-12-31 \
  --no-activate
```

Results are saved to `outputs/tuning/` but not to the database.

---

## Output Structure

After running, artifacts are saved here:

```
outputs/tuning/
  bradley-terry/
    log_loss/
      2024-11-01_2024-12-31_expanding/
        tuning_results_*.csv
        best_params_*.json
    mae_margin/
      2024-11-01_2024-12-31_expanding/
        ...
    mae_total/
      ...
  elo/
    ...
  gssd/
    ...
  poisson/
    ...
  toor/
    ...
```

Each model/metric combination has its own directory with detailed results.

---

## Viewing Tuning Results

### Option 1: Check Best Params JSON

```bash
cat outputs/tuning/elo/log_loss/2024-11-01_2024-12-31_expanding/best_params_*.json
```

Output:
```json
{
  "apply": false,
  "applied": true,
  "baseline_score": 0.5501,
  "best_params": {
    "home_advantage": 50.0,
    "k_factor": 20.0,
    "initial_rating": 1500.0,
    "min_rating": 1.0
  },
  "best_score": 0.5432,
  "improved": true,
  "metric": "log_loss",
  "model": "elo",
  "run_id": "20250125_120000"
}
```

### Option 2: Check Results CSV

```bash
head outputs/tuning/elo/log_loss/2024-11-01_2024-12-31_expanding/tuning_results_*.csv
```

Shows all candidates tested with their scores.

---

## FAQ

**Q: Can I tune only one market instead of all three?**
A: Yes! Use `--market ML` (or SPREAD, TOTAL)

```bash
python scripts/tune_all_models_markets.py \
  --sport nba \
  --season 2025-26 \
  --csv data/raw/nba.csv \
  --start 2024-11-01 \
  --end 2024-12-31 \
  --market ML
```

**Q: How long does it take?**
A: Typically 5-30 minutes depending on:
- CSV size (number of games)
- Number of models (5 by default)
- Number of markets (3 by default)
- `--jobs` setting (more jobs = faster)

**Q: Can I interrupt and resume?**
A: Each model runs independently. If interrupted, run again with specific `--models` to retry.

**Q: What if a model fails?**
A: The script continues with other models and reports failures in the summary. Check logs for details.

**Q: Can I use different grid sizes per model?**
A: Yes! Use `--grid-file` with per-model grids:
```json
{
  "elo": { "k_factor": [10, 20, 30], ... },
  "bradley-terry": { "temp": [2.5, 3.0], ... }
}
```

**Q: Do tuned params override CLI flags?**
A: Yes. Tuned params are auto-loaded by the system when available. To use CLI overrides, pass `--model-params`.

---

## Next Steps

After tuning:

1. **Verify params loaded:** `python -m src.cli.pipeline show-active-params --sport nba --season 2025-26`
2. **Generate schedule:** `python -m src.cli.pipeline schedule --sport nba --season 2025-26 --strict`
3. **Review projections:** Open the output schedule in Excel

---

## Script Location

The script is located at:
```
scripts/tune_all_models_markets.py
```

Run from the repo root:
```bash
python scripts/tune_all_models_markets.py --sport nba --season 2025-26 ...
```
