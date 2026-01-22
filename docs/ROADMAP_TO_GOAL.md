# Sports Power Ratings: Gap Analysis & Roadmap

**Goal:** A functional data pipeline that ingests CSV data, normalizes/standardizes it, stores it in a sport/season SQLite database, runs forecasting models, and outputs a daily Excel workbook showing today's matchups with expected value calculations using model probabilities.

**Assessment Date:** January 22, 2026  
**Current Status:** ~85-90% Complete — Core infrastructure is solid; primary gaps are in daily workflow polish and EV surfacing.

---

## Executive Summary

| Area | Status | Gap |
|------|--------|-----|
| **Data Ingestion** | ✅ Complete | None |
| **Database Storage** | ✅ Complete | None |
| **Forecasting Models** | ✅ Complete | None |
| **Probability Outputs** | ✅ Complete | None |
| **Tuning/Calibration** | ✅ Complete | None |
| **Excel Workbook Output** | ⚠️ Mostly Complete | Minor polish needed |
| **Daily EV Workflow** | ⚠️ Partial | Needs streamlining |
| **Market Data Integration** | ⚠️ Tables exist, workflow unclear | Needs documentation |

---

## 1. What's Working Today

### 1.1 Data Ingestion Pipeline ✅
- **Status:** Fully functional
- **Commands:**
  ```bash
  python -m src.cli.pipeline import --sport nba --season 2025-26 --input data/raw/nba_schedule.csv
  ```
- **What it does:**
  - Parses CSV/HTML from Sports-Reference and other sources
  - Normalizes team names, dates, scores
  - Generates deterministic `game_id` values
  - Stores in `data/db/<sport>/<season>.db`
- **Current data:** NBA 2025-26 has 1,230 games loaded

### 1.2 Database Layer ✅
- **Status:** Fully functional with migrations
- **Tables available:**
  - `games` — Core schedule/results
  - `teams` / `team_aliases` — Canonical names + mappings
  - `model_metrics` — Calibration data per model
  - `model_market_tuning_runs` — Tuning history
  - `model_market_active_params` — Active tuned parameters
  - `market_lines` — Market odds (ready, currently empty)
  - `bets` — Bet tracking (ready, currently empty)
  - `clv_snapshots` — Closing line value tracking

### 1.3 Forecasting Models ✅
- **Status:** 5 models fully implemented and producing output
- **Available models:**
  | Model | Abbrev | Description |
  |-------|--------|-------------|
  | `bradley-terry` | bt | Bradley-Terry log-linear model |
  | `elo` | elo | Classic Elo with tunable K-factor |
  | `gssd` | gssd | Generalized Spread-Score Distribution |
  | `poisson` | pois | Poisson goal/point model |
  | `toor` | toor | Team Opponent-Adjusted Rating |

- **Command:**
  ```bash
  python -m src.cli.pipeline rank --sport nba --season 2025-26
  ```
- **Outputs:**
  - CSV rankings per model: `data/processed/nba/2025-26/{model}_rankings.csv`
  - Stores calibration in `model_metrics` table

### 1.4 Probability & Projection Outputs ✅
- **Status:** Working and comprehensive
- **What you get per game:**
  - `home_win_prob` / `away_win_prob` — ML probabilities
  - `margin_mean` / `margin_sd` — Spread forecast (normal distribution)
  - `total` / `total_sd` — Total points forecast
  - `win_prob_source` — Tracks where the probability came from
- **Calibrated probabilities available** via ML calibrator when trained

### 1.5 Tuning & Calibration ✅
- **Status:** Full tuning infrastructure in place
- **Commands:**
  ```bash
  # Tune a model's hyperparameters
  python -m src.cli.pipeline tune --model elo --csv data/raw/nba_history.csv \
    --start 2020-01-01 --end 2024-12-31 --metric log_loss \
    --apply-best --sport nba --season 2025-26

  # Per-market tuning (ML/SPREAD/TOTAL)
  python -m src.cli.pipeline tune-model --sport nba --season 2025-26 --model elo

  # Ensemble weight tuning
  python -m src.cli.pipeline tune-ensemble --sport nba --season 2025-26 --market ML

  # ML probability calibration
  python -m src.cli.pipeline calibrate --sport nba --season 2025-26 --source elo
  ```
- **Current state:** All 5 models have tuning runs for NBA 2025-26

### 1.6 Excel Workbook Export ✅
- **Status:** Working, produces multi-sheet workbook
- **Command:**
  ```bash
  python -m src.cli.pipeline schedule --sport nba --season 2025-26
  ```
- **Output:** `data/processed/nba/2025-26/schedule_with_projections.xlsx`
- **Sheets included:**
  - Per-model projection sheets (bt, elo, gssd, pois, toor)
  - `dashboard` — Summary view
  - `BETS` — For logging bets (formula-enabled)

---

## 2. What's Missing or Needs Work

### 2.1 Daily Matchup Focus ⚠️ (Gap Level: Low)
**Current state:** The `schedule` command exports all games (played + upcoming). Filtering to "today only" requires either:
- Using `--upcoming-only` flag (but this shows all future games, not just today)
- Opening the workbook and filtering manually

**What's needed:**
- Add `--as-of-date <YYYY-MM-DD>` flag to `schedule` command to filter to a specific day
- Or create a dedicated `daily-workbook` command that produces a focused today-only view

**Workaround today:**
```bash
python -m src.cli.pipeline schedule --sport nba --season 2025-26 --upcoming-only
# Then filter in Excel to today's date
```

### 2.2 Expected Value (EV) Calculations ⚠️ (Gap Level: Medium)
**Current state:** 
- The infrastructure for EV exists in `opportunities.py` and `excel_formulas.py`
- The `BETS` sheet in review workbooks has EV formula support
- **BUT:** The standard `schedule` workbook doesn't automatically show EV

**What's needed to see EV in the daily workbook:**
1. Market lines must be imported first (odds/lines from a sportsbook)
2. Run the `review-generate` workflow which matches projections to market lines
3. The EV formula is: `EV = (model_prob × payout) - (1 - model_prob) × stake`

**Commands to enable EV:**
```bash
# 1. Import market lines (from a CSV with odds)
python -m src.cli.pipeline betting market-csv --sport nba --season 2025-26 \
  --csv data/raw/nba_markets.csv --default-book dn

# 2. Generate a review workbook with EV formulas
python -m src.cli.pipeline betting review-generate --sport nba --season 2025-26 \
  --model elo --snapshot-date 2026-01-22 --formula-workbook
```

**Gap:** No single-command workflow to produce a "daily EV workbook" that:
1. Takes today's projections
2. Merges in current market lines
3. Calculates edge and EV for each selection

### 2.3 Market Data Integration ⚠️ (Gap Level: Medium)
**Current state:**
- `market_lines` table exists and schema is ready
- `betting market-csv` command exists to import lines
- **BUT:** The table is currently empty (0 rows)

**What's needed:**
- Regular workflow to capture current lines (either via CSV paste or OCR)
- Document the expected CSV format for market lines

**Expected CSV format for market lines:**
```csv
game_date,team_home_raw,team_away_raw,market_type,selection,line,odds,book
2026-01-22,Lakers,Celtics,ML,Lakers,,−150,DraftKings
2026-01-22,Lakers,Celtics,ML,Celtics,,+130,DraftKings
2026-01-22,Lakers,Celtics,spread,Lakers,-3.5,-110,DraftKings
2026-01-22,Lakers,Celtics,spread,Celtics,+3.5,-110,DraftKings
2026-01-22,Lakers,Celtics,total,over,225.5,-110,DraftKings
2026-01-22,Lakers,Celtics,total,under,225.5,-110,DraftKings
```

### 2.4 Single-Command Daily Workflow ⚠️ (Gap Level: Low)
**Current state:** Daily workflow requires multiple commands (see `docs/daily-workflow.md`)

**What would be ideal:** A single command like:
```bash
python -m src.cli.pipeline daily --sport nba --season 2025-26 --date 2026-01-22 \
  --market-lines data/raw/todays_lines.csv
```

This would:
1. Ensure rankings are current
2. Import market lines for today
3. Generate a workbook with:
   - Today's matchups with projections
   - EV calculations against imported lines
   - Ready-to-fill BETS sheet

---

## 3. Complete Daily Workflow (Using Current Commands)

Here's the full workflow to achieve your goal with the current system:

### Step 1: Ensure data is current
```bash
# Import latest results (run after yesterday's games)
python -m src.cli.pipeline import --sport nba --season 2025-26 --input data/raw/nba_schedule.csv

# Rebuild rankings
python -m src.cli.pipeline rank --sport nba --season 2025-26
```

### Step 2: Get today's market lines
Prepare a CSV file (`data/raw/nba_lines_today.csv`) with format:
```csv
game_date,team_home_raw,team_away_raw,market_type,selection,line,odds,book
2026-01-22,Lakers,Celtics,ML,Lakers,,-150,dk
2026-01-22,Lakers,Celtics,spread,Lakers,-3.5,-110,dk
...
```

Import it:
```bash
python -m src.cli.pipeline betting market-csv --sport nba --season 2025-26 \
  --csv data/raw/nba_lines_today.csv --default-book dk
```

### Step 3: Generate the workbook with EV
```bash
# Option A: Standard schedule (no EV, but has all projections)
python -m src.cli.pipeline schedule --sport nba --season 2025-26

# Option B: Review workbook with EV (requires market lines)
python -m src.cli.pipeline betting review-generate --sport nba --season 2025-26 \
  --model elo --snapshot-date 2026-01-22 --formula-workbook \
  --output-dir data/processed/nba/2025-26/review
```

### Step 4: Use the workbook
1. Open `schedule_with_projections.xlsx` or the review workbook
2. Look at the `BETS` or `EV` sheet
3. Columns available:
   - `home_win_prob` / `away_win_prob` — Your model's probability
   - `implied_prob` — Market-implied probability (from odds)
   - `edge` — Your edge: `model_prob - implied_prob`
   - `ev` — Expected value per unit staked
4. Fill in `stake` for bets you want to place
5. Log bets to DB:
   ```bash
   python -m src.cli.pipeline betting log-bets \
     --workbook data/processed/nba/2025-26/review/nba-2025-26-review.xlsx \
     --writeback
   ```

---

## 4. Recommended Improvements (Priority Order)

### Priority 1: Add `--as-of-date` to `schedule` command
**Effort:** ~2 hours  
**Value:** High — Immediately filters to today's games

### Priority 2: Document market lines CSV format
**Effort:** ~1 hour  
**Value:** High — Removes confusion on how to get lines into the system

### Priority 3: Add EV columns to standard `schedule` output
**Effort:** ~4-6 hours  
**Value:** High — Shows EV alongside projections without needing review workflow

### Priority 4: Create `daily` command that combines steps
**Effort:** ~1 day  
**Value:** Medium — Convenience, not essential

### Priority 5: Add Action/FanDuel paste parser for market lines
**Effort:** ~4-6 hours  
**Value:** Medium — Faster line capture vs. manual CSV

---

## 5. Quick Reference: Key Files

| Purpose | File |
|---------|------|
| CLI entrypoint | [src/cli/pipeline.py](src/cli/pipeline.py) |
| Data ingestion | [src/pipelines/ingest.py](src/pipelines/ingest.py) |
| Ranking pipeline | [src/pipelines/run_rankings.py](src/pipelines/run_rankings.py) |
| Schedule export | [src/pipelines/schedule.py](src/pipelines/schedule.py) |
| Model registry | [src/models/registry.py](src/models/registry.py) |
| EV calculations | [src/pipelines/opportunities.py](src/pipelines/opportunities.py) |
| Excel formulas | [src/pipelines/excel_formulas.py](src/pipelines/excel_formulas.py) |
| DB schema | [src/data/repository.py](src/data/repository.py) |
| Daily workflow docs | [docs/daily-workflow.md](docs/daily-workflow.md) |
| CLI docs | [docs/CLI.md](docs/CLI.md) |

---

## 6. Summary

**You're very close.** The core pipeline is complete:
- ✅ CSV → Database ingestion works
- ✅ 5 forecasting models produce probabilities
- ✅ Tuning and calibration are available
- ✅ Excel workbook export exists with BETS sheet

**The gaps are workflow convenience, not missing functionality:**
1. Market lines need to be imported before EV can be calculated
2. The `schedule` workbook focuses on projections; EV requires the `review-generate` path
3. No single "daily" command exists yet

**Immediate action items:**
1. Try importing sample market lines with `betting market-csv`
2. Generate a review workbook with `--formula-workbook`
3. See the EV/edge columns in action

Would you like me to implement any of the recommended improvements?
