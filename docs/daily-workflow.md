# Daily Operations Runbook

Practical, step-by-step guide to manage your betting workflow: import schedules/results, generate daily workbooks, log bets from Excel, and track performance. All historical and new bets are treated identically—they're stored in the same database with the same prediction context for validation and strategy analysis.

## Key concepts

**Historical vs. new bets:** Once imported, there is no distinction. Both store the same fields (prediction context, wager details, game info) and are analyzed together.

**Prediction context at bet time:** Every bet captures: `model_prob`, `edge`, `ev`, `home_win_prob`, `away_win_prob`, and `ensemble_components_json`. This lets you validate strategy later (e.g., "did +3 edge bets hit at the expected rate?").

**Book is optional:** The `book` column (DraftKings, FanDuel, etc.) is for your reference and strategy analysis. Leaving it blank is fine, especially for historical bets.

**Idempotent updates:** Re-logging the same bet (same game, market, selection) updates rather than duplicates. This lets you correct line/odds without creating duplicates.

## Conventions used below

- Example variables: `SPORT=nba`, `SEASON=2025-26`, `DB=data/db/nba/2025-26.db`.
- All commands run from the repo root: `python -m src.cli.pipeline ...`.
- The schedule workbook contains a `BETS` sheet where you enter your bets; Excel formulas calculate edge/EV automatically.
- All bets (historical and new) store: prediction context (model probs, edge, EV), your wager (stake, odds, line, book), and game info (game_id, teams, scores).
- Optional `book` column tracks where you placed each bet; leaving it blank is fine for backtesting.

## One-time setup (per machine)

```bash
python -m venv .pyenv
./.pyenv/Scripts/Activate.ps1
pip install -r requirements.txt
```

## Seasonal prep (do once when a new season starts)

1) Import the schedule/results into the season DB (skip if already present):

```bash
python -m src.cli.pipeline import --sport nba --season 2025-26 --input data/raw/nba_schedule.csv
```

2) Tune individual models on history (persist best params for the season):

```bash
python -m src.cli.pipeline tune --model elo --csv data/raw/nba_history.csv \
  --start 2020-01-01 --end 2024-12-31 --metric log_loss \
  --apply-best --sport nba --season 2025-26 --db data/db/nba/2025-26.db
```

3) (Optional) Tune per-market and ensemble weights if you use ensembles:

```bash
# Per-model, per-market tuning (runs ML/SPREAD/TOTAL by default)
python -m src.cli.pipeline tune-model --sport nba --season 2025-26 --model elo \
  --csv data/raw/nba_history.csv --start 2020-01-01 --end 2024-12-31 \
  --output-dir outputs/tuning/nba/2025-26/elo

# Ensemble weight tuning
python -m src.cli.pipeline tune-ensemble --sport nba --season 2025-26 --market ML \
  --start-date 2020-01-01 --end-date 2024-12-31 --ensemble ensemble_ml_v1 \
  --csv data/raw/nba_history.csv
```

4) Activate tuned params/weights so `rank` and `schedule` pick them up:

```bash
# Promote best runs to actives (all models/markets)
python -m src.cli.pipeline bootstrap-market-actives --sport nba --season 2025-26 --model all

# Verify what is active
python -m src.cli.pipeline tuning-status --sport nba --season 2025-26
```

## One-time: Import historical bets

If you have betting history from an external app (e.g., DraftKings export), import it once to backfill the database:

### Option A: From Excel workbook (easiest)

If you have a workbook with your historical bets already in `BETS` sheet format:

```bash
python -m src.cli.pipeline betting log-bets --workbook path/to/historical_bets.xlsx --writeback
```

This reads the `BETS` sheet and logs all non-blank rows to the database.

### Option B: From CSV export from betting app

If you exported your bets as a CSV from your betting app (with columns like League, Game, Odds, Result, etc.):

1. **Parse the CSV** to match bets to games and assign game_ids:
```bash
python -m src.cli.pipeline betting parse-export --csv data/raw/betshistory.csv
```

This produces `betshistory_<sport>_with_ids.csv` with game_ids filled in for matched bets.

2. **Verify matches** in the output file:
   - Open `betshistory_nba_with_ids.csv`
   - Check that `game_id` column is populated (e.g., `nba:2025-26:2025-12-27:1d407990241f`)
   - Rows without game_ids (empty game_id cell) are unmatched; you'll have to manually add them or skip them

3. **Import the matched bets:**
```bash
python -m src.cli.pipeline betting import-csv --csv betshistory_nba_with_ids.csv
```

This imports all rows with a game_id into the database. Sport and season are auto-detected from the game_id.

**Note:** Historical bets without a `book` column are fine; just add the column header with blank values if needed.

## Daily flow (repeat every day)

### Step 1: Import today's schedule and results

Update the database with the latest games (lines and scores):

```bash
python -m src.cli.pipeline import --sport nba --season 2025-26 --input data/raw/nba_schedule.csv
```

**Verification:** Check that all games loaded correctly:
- Open the DB: `sqlite3 data/db/nba/2025-26.db`
- Query: `SELECT COUNT(*) FROM games WHERE date = '2025-01-24';` (should show today's games)
- Check for NULLs in critical columns: `SELECT * FROM games WHERE game_id IS NULL LIMIT 5;`

### Step 2: Rebuild power ratings

Recompute model rankings with the latest results (uses active tuned params automatically):

```bash
python -m src.cli.pipeline rank --sport nba --season 2025-26
```

**Verification:** Spot-check a few team ratings:
- `python -m src.cli.pipeline run-model --sport nba --season 2025-26 --model elo` (shows top/bottom teams)

### Step 3: Generate today's workbook

Export a schedule workbook with your model projections and a `BETS` sheet ready for input:

```bash
python -m src.cli.pipeline schedule --sport nba --season 2025-26 \
  --as-of-date 2025-01-24 \
  --output outputs/nba_schedule_2025-01-24.xlsx
```

**Verification:** Open the Excel file:
- `PROJECTIONS` sheet: verify team ratings and win probabilities look reasonable
- `BETS` sheet: all upcoming games appear with game_id pre-filled, blank rows for you to fill in
- Formula columns (implied_prob, edge, ev): should calculate automatically as you enter odds

### Step 4: Enter your bets in Excel

Open `outputs/nba_schedule_2025-01-24.xlsx` and fill the `BETS` sheet:

**Required columns (fill these):**
- `stake`: Amount you're wagering (e.g., 1.0, 5.0)
- `odds`: What you actually got (e.g., -110, +120) — use this OR `price` column, not both
- `book`: Where you placed it (e.g., "DraftKings", "FanDuel") — optional, can leave blank
- `line`: If the spread/total differs from the original, update it here

**Auto-calculated columns (read-only formulas):**
- `implied_prob`: Your odds converted to probability
- `edge`: Model prob minus implied prob (positive = model favors you)
- `ev`: Expected value per unit (edge × decimal odds)
- `exp_profit_stake`: Your expected profit if you won the expected number of these bets

**Leave blank to skip:** If you don't want to bet a game, leave `stake` empty and it will be skipped.

Save the file when done.

### Step 5: Dry-run validation (before committing)

Always validate before writing to the database:

```bash
python -m src.cli.pipeline betting log-bets --workbook outputs/nba_schedule_2025-01-24.xlsx --dry-run
```

**Expected output:**
```
log_bets dry-run: parsing workbook (no writes)
log_bets dry-run parsed 5 bet rows
```

If you see errors or unexpected skips, review the BETS sheet and re-save before proceeding to step 6.

### Step 6: Log bets to the database with writeback

Commit your bets to the database and update the workbook with system columns (bet_id, logged_at):

```bash
python -m src.cli.pipeline betting log-bets --workbook outputs/nba_schedule_2025-01-24.xlsx --writeback
```

**Expected output:**
```
log_bets processed 5 rows
```

The workbook is now updated:
- New `bet_id` column: unique ID for each bet (for tracking/reference)
- New `logged_at` column: timestamp when the bet was logged
- New `log_status` column: should show "logged" for all processed rows

**What happens in the database:**
Each bet is stored with:
- Your wager: `stake`, `odds`, `line`, `book`
- Prediction context at bet time: `model_prob`, `edge`, `ev`, `home_win_prob`, `away_win_prob`
- Game info: `game_id`, `date`, teams (via game_id lookup)
- Ensemble details: `market_forecast_source`, `ensemble_components_json`

### Step 7: Settle bets (once games are final)

After games conclude, settle all bets against final scores:

```bash
python -m src.cli.pipeline betting settle-bets --sport nba --season 2025-26
```

**What happens:**
- `outcome` column is populated: "W" (win), "L" (loss), "P" (push/tie)
- `profit` column is calculated: `(payout - stake)` based on your odds
- ROI and other metrics are computed

### Step 8 (Optional): Generate daily report

Produce a daily report workbook for performance analysis:

```bash
python -m src.cli.pipeline betting report --sport nba --season 2025-26 --type daily \
  --output outputs/reports/nba-daily-2025-01-24.xlsx
```

This workbook includes:
- Daily PnL summary
- Edge buckets (bets by edge range and their actual outcomes)
- ROI by market type, book, model
- Calibration analysis (did +2 EV bets win at the expected rate?)

## Typical daily sequence (copy-paste template)

Replace `2025-01-24` with today's date and `nba_schedule.csv` with your latest file:

```bash
# Step 1: Update schedule and results
python -m src.cli.pipeline import --sport nba --season 2025-26 --input data/raw/nba_schedule.csv

# Step 2: Rebuild ratings
python -m src.cli.pipeline rank --sport nba --season 2025-26

# Step 3: Generate workbook
python -m src.cli.pipeline schedule --sport nba --season 2025-26 \
  --as-of-date 2025-01-24 \
  --output outputs/nba_schedule_2025-01-24.xlsx

# (Pause: open Excel, fill in BETS sheet, save)

# Step 4: Dry-run validation
python -m src.cli.pipeline betting log-bets --workbook outputs/nba_schedule_2025-01-24.xlsx --dry-run

# Step 5: Log bets (with writeback to update workbook)
python -m src.cli.pipeline betting log-bets --workbook outputs/nba_schedule_2025-01-24.xlsx --writeback

# Step 6: Settle (once games are final)
python -m src.cli.pipeline betting settle-bets --sport nba --season 2025-26

# Step 7 (optional): Generate report
python -m src.cli.pipeline betting report --sport nba --season 2025-26 --type daily \
  --output outputs/reports/nba-daily-2025-01-24.xlsx
```

## Tips & troubleshooting

- **Excel formula errors:** If `implied_prob`, `edge`, or `ev` show `#DIV/0!` or `#NAME?`, check that `odds` is a number (not text) and `model_prob` is present.
- **No games found:** If step 1 fails with "no games found", verify your CSV has the right columns: `date`, `home_team`, `away_team`, `home_score`, `away_score`. Team names must match exactly what's in the database (check with `sqlite3 data/db/nba/2025-26.db "SELECT DISTINCT home_team FROM games LIMIT 5;"`).
- **Duplicate bets:** `log-bets` enforces UNIQUE(review_run_id, game_id, market_type, selection). Re-running the same workbook will update existing rows, not duplicate them.
- **Dry-run first:** Always run `--dry-run` before `--writeback` to catch errors without modifying the workbook.
- **Use `--strict` on schedule** to fail fast if tuned params or ensemble weights are missing (optional, for safety).
