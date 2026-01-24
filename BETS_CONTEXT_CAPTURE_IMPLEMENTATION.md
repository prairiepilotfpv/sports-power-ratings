# Bet Context Capture Implementation

## Overview

You now have the ability to **capture the full decision context** when logging bets, including:
- Model probabilities (home win %, away win %)
- Model-specific probability for the market
- Your calculated edge
- Expected value ($)
- Spread/total forecast details (mean, standard deviation)
- Ensemble source and component weights (as JSON)

This enables **complete validation** of your ensemble weights later against actual bet outcomes.

## What Changed

### 1. Enhanced Database Schema

The `bets` table now includes 11 new optional columns to store prediction context:

```
home_win_prob REAL              -- Home team win probability from your model
away_win_prob REAL              -- Away team win probability
model_prob REAL                 -- Model probability specific to this market
edge REAL                       -- Your calculated edge (%)
ev REAL                         -- Expected value ($)
margin_mean REAL                -- For spreads: predicted margin mean
margin_sd REAL                  -- For spreads: predicted margin std dev
total REAL                      -- For totals: predicted total points
total_sd REAL                   -- For totals: predicted total std dev
market_forecast_source TEXT     -- Which ensemble/model (e.g., "elo_ml_ensemble")
ensemble_components_json TEXT   -- JSON: {"elo": 0.6, "bradley_terry": 0.4, ...}
```

### 2. Enhanced Bet Logging (`log_bets`)

When you run the BETS sheet workflow:

```powershell
python -m src.cli.pipeline betting log-bets \
  --workbook schedule_with_bets.xlsx \
  --db data/db/nba/2025-26.db \
  --writeback
```

The system now **automatically extracts** all prediction columns from the BETS sheet and stores them alongside the bet details. Your BETS sheet should already have these columns from the schedule generation.

### 3. Historical Bet Import (`import-csv`)

You can now import bets from your external betting app. **Important: Each sport has its own database, so CSV files must contain bets for a single sport only.**

✅ **Correct usage:**
```powershell
# Import all NBA bets to data/db/nba/2025-26.db
python -m src.cli.pipeline betting import-csv \
  --csv nba_history.csv \
  --sport nba \
  --season 2025-26 \
  --dry-run  # first verify the import

# Then actually import
python -m src.cli.pipeline betting import-csv \
  --csv nba_history.csv \
  --sport nba \
  --season 2025-26
```

❌ **Incorrect usage (will fail):**
```powershell
# This will FAIL because the CSV contains both NBA and NHL bets
# Error: "CSV contains bets for mismatched sports: ['nhl']. Target database is for sport 'nba'."
python -m src.cli.pipeline betting import-csv \
  --csv all_sports_history.csv \
  --sport nba \
  --season 2025-26
```

**If you have a mixed-sport CSV, split it by sport first:**
```powershell
# Import NBA bets separately
python -m src.cli.pipeline betting import-csv --csv nba_only.csv --sport nba --season 2025-26

# Import NHL bets separately  
python -m src.cli.pipeline betting import-csv --csv nhl_only.csv --sport nhl --season 2025-26
```

This ensures:
- ✅ NHL bets go **only** to `data/db/nhl/2025-26.db`
- ✅ NBA bets go **only** to `data/db/nba/2025-26.db`
- ✅ No mixing of sports in the same database
- ✅ No new databases are created (uses existing database for sport/season)

The import handles:
- **CSV format**: league, start_time, game, type, odds, odds_spread_total, result, units_wagered
- **Sport validation**: Rejects CSVs with mixed sports (e.g., NBA + NHL in same file)
- **Game resolution**: Either fuzzy-match team names or provide explicit game_id column
- **Market type mapping**: ml_away/ml_home → ML, spread_away/spread_home → spread, over/under → total
- **Outcome tracking**: Automatically maps win/loss/push results and computes profit
- **Idempotency**: Re-importing the same CSV is safe; bets are updated, not duplicated

### 4. CLI Command

New command: `betting import-csv`

```bash
python -m src.cli.pipeline betting import-csv \
  --csv <path> \
  --sport <sport> \
  --season <season> \
  [--db <path>] \
  [--review-run-id <id>] \
  [--dry-run]
```

**Flags:**
- `--csv` (required): Path to CSV file
- `--sport` (required): Sport code (nba, nhl, etc.). **CSV must contain only bets for this sport; mixed-sport CSVs are rejected.**
- `--season` (required): Season code (e.g., 2025-26)
- `--db`: Optional override DB path; defaults to `data/db/<sport>/<season>.db`
- `--review-run-id`: Optional; auto-generated as `history-import-<sport>-<YYYYMMDD>` if omitted
- `--dry-run`: Validate without writing to DB

## Workflow: Phase 2 (Daily Bet Tracking)

### Step 1: Generate Schedule with Projections

```powershell
python -m src.cli.pipeline schedule \
  --sport nba \
  --season 2025-26 \
  --output schedule_with_bets.xlsx \
  [--model elo]  # optional
```

This produces an Excel workbook with:
- `PROJECTIONS` sheet: Matchups, probabilities, market forecasts
- `BETS` sheet: 6 rows per game (2×ML, 2×spread, 2×total) with all prediction columns

### Step 2: Fill in Stakes

Open `schedule_with_bets.xlsx` → `BETS` sheet
- Leave blank rows or `stake=0` to PASS
- Fill in `stake` for bets you want to place
- Optionally override `odds`, `line`, `book`

### Step 3: Log Bets to DB

```powershell
python -m src.cli.pipeline betting log-bets \
  --workbook schedule_with_bets.xlsx \
  --db data/db/nba/2025-26.db \
  --dry-run  # first

python -m src.cli.pipeline betting log-bets \
  --workbook schedule_with_bets.xlsx \
  --db data/db/nba/2025-26.db \
  --writeback  # writes bet_id and logged_at back to sheet
```

The system **captures**:
- ✅ game_id, market_type, selection, line, odds, stake, book
- ✅ home_win_prob, away_win_prob, model_prob, edge, ev
- ✅ margin_mean, margin_sd (for spreads)
- ✅ total, total_sd (for totals)
- ✅ market_forecast_source, ensemble_components_json
- ✅ clv_close_odds, clv_close_line (closing line if available)

### Step 4: Settle Bets (Later)

Once games finish and scores are recorded:

```powershell
python -m src.cli.pipeline betting settle-bets \
  --sport nba \
  --season 2025-26 \
  --db data/db/nba/2025-26.db
```

This auto-computes `outcome` (win/loss/push) and `profit` for all pending bets.

### Step 5: Validate & Report

Query the database for validation:

```sql
-- Get all bets with their context
SELECT game_id, market_type, selection, line, odds, stake, 
       home_win_prob, model_prob, edge, ev,
       outcome, profit
FROM bets
WHERE sport = 'nba' AND season = '2025-26'
ORDER BY logged_at DESC;

-- Calculate P&L by edge bucket
SELECT 
  CASE 
    WHEN edge >= 0.05 THEN '>5%'
    WHEN edge >= 0.02 THEN '2-5%'
    WHEN edge >= 0 THEN '0-2%'
    ELSE '<0%'
  END as edge_bucket,
  COUNT(*) as bet_count,
  SUM(stake) as total_stake,
  SUM(profit) as total_profit,
  SUM(profit) / SUM(stake) as roi
FROM bets
WHERE sport = 'nba' AND season = '2025-26' AND outcome = 'win'
GROUP BY edge_bucket
ORDER BY edge_bucket DESC;
```

## CSV Import Format (from your betting app)

**Important: Each CSV file should contain bets for a SINGLE sport only.** If you have mixed-sport history, split it by sport before importing.

**Single-sport example: `nba_history.csv`**
```csv
league,start_time,game,game_id,pick_desc,type,odds,odds_spread_total,result,units_wagered
nba,2026-01-08T00:00:00.000Z,WAS @ PHI,nba:2025-26:2026-01-08:...,WAS +12.5 -110,spread_away,-110,12.5,loss,1.1
nba,2026-01-08T00:00:00.000Z,TOR @ CHA,nba:2025-26:2026-01-08:...,TOR -135,ml_away,-135,-135,win,1.35
```

**❌ Avoid: Mixed-sport file (will be rejected)**
```csv
league,start_time,game,game_id,pick_desc,type,odds,odds_spread_total,result,units_wagered
nba,2026-01-08T00:00:00.000Z,WAS @ PHI,nba:2025-26:...,WAS +12.5 -110,spread_away,-110,12.5,loss,1.1
nhl,2026-01-08T02:30:00.000Z,OTT @ UTA,nhl:2025-26:...,UTA -125,ml_home,-125,-125,win,1.25
```

**Required columns:**
- `league` or `sport` — Sport code (nba, nhl, etc.). **All rows in this CSV must use the SAME sport code; mixed sports will be rejected.** This column is useful for sorting history by sport in your original export.
- `start_time` — **Game time in ISO format** (e.g., `2026-01-08T00:00:00.000Z`). This is the time the game was scheduled, NOT the bet placement time. The import function extracts the game date from this timestamp to resolve the game ID.
- `game` (e.g., `WAS @ PHI`)
- `game_id` (explicit ID; if missing, fuzzy matching is attempted)
- `type` (ml_away, ml_home, spread_away, spread_home, under, over)
- `odds` (American odds, e.g., -110, 120)
- `odds_spread_total` or `line` (numeric line)
- `result` (win, loss, push)
- `units_wagered` (stake)

## Database Backward Compatibility

✅ **No breaking changes.** All new columns are optional (NULL). Existing bets remain queryable.

## Tests

Run the full test suite to verify:

```bash
python -m pytest tests/test_bets_context_capture.py -v
```

Tests cover:
- `test_log_bets_captures_prediction_context`: Verifies all fields are extracted and stored
- `test_import_bets_csv_with_history_export`: Verifies CSV import and outcome calculation
- `test_import_bets_csv_idempotent`: Verifies re-importing is safe
- `test_import_bets_csv_dry_run`: Verifies dry-run doesn't write

## Key Assumptions

1. **Review Run ID**: For historical imports, a new review_run_id is auto-generated as `history-import-<sport>-<date>`. Override with `--review-run-id` if needed.

2. **Prediction Context**: The BETS sheet columns (`home_win_prob`, `model_prob`, etc.) are populated by the schedule generation process. If they're missing or blank, they default to NULL in the database.

3. **Game ID Resolution**: The import function checks for explicit `game_id` column first. If not provided, it attempts fuzzy matching on team names. For reliability, always include `game_id` in your export.

4. **Outcome Calculation**: Profit is computed based on American odds:
   - Win: profit = stake × (abs(odds) / 100) if odds > 0, else stake / (abs(odds) / 100)
   - Loss: profit = -stake
   - Push: profit = 0

## Files Modified

- **Schema**: `src/data/betting_repository.py` (added 11 columns to `bets` table)
- **Logging**: `src/pipelines/bets.py` (enhanced `log_bets()` to extract context)
- **Import**: `src/data/betting_repository.py` (new `import_bets_csv()` function)
- **CLI**: `src/cli/betting.py`, `src/cli/pipeline.py` (wired up `betting import-csv`)
- **Tests**: `tests/test_bets_context_capture.py` (4 new tests)

## Next Steps (Optional)

1. **Calibration**: Once you have enough settled bets, you can re-run ensemble calibration with historical outcomes to validate your edge estimates.

2. **Bet Reporting**: Use the reporting functions to generate weekly/monthly summaries with P&L by model, market, edge bucket, etc.

3. **Backtest Integration**: Your logged bets (with full context) can feed into backtests to compare paper vs. real performance.

---

**You're all set!** You now have:
✅ Full bet context capture (every prediction detail)
✅ Historical import from your betting app
✅ Daily bet logging workflow
✅ Idempotent, auditable database
✅ Complete validation capability

Any questions, feel free to ask!
