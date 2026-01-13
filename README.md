# Sports Power Ratings

Local-first tooling to turn Sports-Reference schedules/results into SQLite databases, power ratings, projections, and reports. Everything runs through a single CLI and works entirely offline once inputs are downloaded.

## Table of Contents
- Overview
- Repository Layout
- Model Overview
- Requirements
- Installation
- Data Locations
- Quickstart
- Workflows
- CLI Reference
- Model Parameter Overrides
- Input Format Notes
- Outputs At A Glance
- Configuration
- Testing
- Troubleshooting
- License

Use the Quickstart below for common tasks; the detailed CLI reference lives in [docs/CLI.md](docs/CLI.md).

## Overview

What it does:

1) Ingest Sports-Reference HTML/CSV (or pasted CSV text) and normalize it.
2) Persist games into per-sport/per-season SQLite DBs.
3) Fit power ratings (Bradley-Terry, Elo, GSSD, TOOR, Poisson) and store calibration metrics.
4) Generate matchup projections and daily schedules with spreads/totals/win probabilities.
5) Export rankings and schedule/report workbooks.
6) Backtest and tune models on historical CSVs.

## Repository layout

```
src/
  cli/                 CLI entry points (pipeline is the preferred one)
  data/                SQLite helpers, paths, validation
  ingest/              Sports-Reference parsing + normalization
  models/              Power rating implementations + registries
  pipelines/           Orchestration for ingest, rankings, projections, reports
  backtest/            Backtest runner + exports

data/
  raw/                 Drop HTML/CSV inputs here (optional convenience)
  processed/           Rankings, schedules, reports
  db/<sport>/<season>.db  SQLite DB per sport/season
outputs/               Backtest and tuning artifacts
```

## Model overview

- **Bradley-Terry**: logistic win-probability ratings for head-to-head outcomes.
- **Elo**: incremental ratings with home-advantage adjustments.
- **GSSD**: ratings derived from per-team scoring splits.
- **TOORPowerRating**: OLS on game margins to derive power ratings.
- **TOORModel**: maps Bradley-Terry strengths to margins via OLS for backtests.
- **Poisson**: attack/defense scoring model with simulation-driven totals.

## Requirements

- Python 3.11+ (tested locally)
- Tesseract OCR if you plan to ingest screenshots/images (optional)
- `OPENAI_API_KEY` / `OPENAI_MODEL` only if you want OCR assistance; not required otherwise

## Installation

1) Create a virtual environment

```bash
python -m venv .pyenv
```

2) Activate it

- Windows (PowerShell): `./.pyenv/Scripts/Activate.ps1`
- macOS/Linux: `source .pyenv/bin/activate`

3) Install runtime dependencies

```bash
pip install -r requirements.txt
```

4) (Optional) Install test/tooling extras

```bash
pip install -r requirements-dev.txt
```

## Data Locations

- Default DB: `data/db/<sport>/<season>.db`
- Default processed outputs: `data/processed/<sport>/<season>/`
- Raw inputs: any path you provide; bare filenames are resolved under `data/raw/`.
- Input CSV/HTML should be straight from Sports-Reference; future games (no scores) are allowed.
- Backtest CSVs must contain `date`, `home_team`, `away_team`, `home_score`, `away_score` (common aliases are auto-detected). Optional: `neutral`, `overtime`, `game_id`.

## Quickstart

All commands run via `python -m src.cli.pipeline <command> ...`. The pipeline is the supported path; `src.cli.ingest` and `src.cli.run_rankings` remain for legacy use only.

Example for NBA 2025-26 with a CSV at `data/raw/nba_2025_26.csv`:

```bash
# 1) Ingest into SQLite (creates data/db/nba/2025-26.db by default)
python -m src.cli.pipeline import --sport nba --season 2025-26 --input data/raw/nba_2025_26.csv

# 2) Build rankings (runs all available models when --model is omitted)
python -m src.cli.pipeline rank --sport nba --season 2025-26

# 3) Export schedule projections (default: Excel workbook with one sheet per model + dashboard)
python -m src.cli.pipeline schedule --sport nba --season 2025-26

# 4) Predict a single matchup
python -m src.cli.pipeline matchup --sport nba --season 2025-26 --matchup "Lakers vs Celtics"

# 5) Generate a rankings-only Excel report
python -m src.cli.pipeline report --sport nba --season 2025-26

# 6) Backtest a model on historical games
python -m src.cli.pipeline backtest --csv nba_results.csv --model bradley_terry_hfa --start 2024-11-01 --end 2024-12-01

# 7) Tune model hyperparameters via repeated backtests
python -m src.cli.pipeline tune --model elo --csv nba_results.csv --start 2024-11-01 --end 2024-12-01 --metric log_loss
```

## Best Probabilities: Quickstart (model + market tuning)

Use this sequence to produce the best calibrated forecasting probabilities for each market (ML, SPREAD, TOTAL) for a given sport/season. These steps assume you have a backtest CSV of historical games (required for tuning) and have ingested games into the DB for the target season.

1) Ingest games into the per-sport/season DB (if not already done):

```bash
python -m src.cli.pipeline import --sport nhl --season 2025-26 --input data/raw/nhl_2025_26.csv
```

2) Build rankings (runs all models by default):

```bash
python -m src.cli.pipeline rank --sport nhl --season 2025-26
```

3) Tune models (per-model grid-search backtests). Use a historical CSV for tuning. Example (tune Elo optimizing log_loss and persist best run):

```bash
python -m src.cli.pipeline tune --model elo --csv data/raw/nhl_history.csv --start 2020-01-01 --end 2024-12-31 --metric log_loss --apply-best --sport nhl --season 2025-26 --db data/db/nhl/2025-26.db
```

4) (Optional) Tune model performance per-market using `tune-model` to persist per-market runs (ML/SPREAD/TOTAL):

```bash
python -m src.cli.pipeline tune-model --sport nhl --season 2025-26 --model elo --market ML --csv data/raw/nhl_history.csv --start 2020-01-01 --end 2024-12-31 --output-dir outputs/tuning/nhl/2025-26/elo
```

5) (Optional) Tune ensemble weights per market (if using ensembles):

```bash
python -m src.cli.pipeline tune-ensemble --sport nhl --season 2025-26 --start-date 2020-01-01 --end-date 2024-12-31 --market ML --ensemble ensemble_ml_v1 --csv data/raw/nhl_history.csv
```

6) Activate/promote a specific model-market tuning run to be used by `rank`/`schedule` (if you need explicit activation). You can either provide `--run-id` (preferred) or `--metric` to pick the best run for that metric:

```bash
# Using a run id
python -m src.cli.pipeline activate-tuning --sport nhl --season 2025-26 --model elo --market ML --run-id tune-20250101-elo-ml

# Or activate the best run for metric=log_loss
python -m src.cli.pipeline activate-tuning --sport nhl --season 2025-26 --model elo --market ML --metric log_loss
```

7) Verify tuning is being used and then export the schedule (final step):

```bash
# Check status (new read-only command)
python -m src.cli.pipeline tuning-status --sport nhl --season 2025-26

# Export schedule (uses active or auto-selected tuned params)
python -m src.cli.pipeline schedule --sport nhl --season 2025-26
```

Notes:
- `rank`/`schedule` prefer explicitly activated per-model-market params (via `activate-tuning`). If none exist, they auto-select the best tuning run for the market's optimized metric (and print that they auto-selected it). If neither is available, defaults are used.
- Use `--strict` on `schedule` to fail when tuned params or ensemble weights are missing.
- If you want explicit, durable actives for every model+market (including defaults), run `bootstrap-market-actives` once for the sport/season.

### NHL 2025-26 exact command sequence (concise)

```bash
python -m src.cli.pipeline import --sport nhl --season 2025-26 --input data/raw/nhl_2025_26.csv
python -m src.cli.pipeline rank --sport nhl --season 2025-26
python -m src.cli.pipeline tune --model elo --csv data/raw/nhl_history.csv --start 2018-01-01 --end 2024-12-31 --metric log_loss --apply-best --sport nhl --season 2025-26 --db data/db/nhl/2025-26.db
python -m src.cli.pipeline tuning-status --sport nhl --season 2025-26
python -m src.cli.pipeline schedule --sport nhl --season 2025-26 --strict
```

## Sanity Check (what to see in logs)

When everything is configured and active, `rank`/`schedule` logs should show one of the following per model+market:

- `Using active market params ...` — explicit activation via `activate-tuning`.
- `Auto-selected tuned params from best run (metric=...)` — no explicit activation but a tuned run was chosen.
- It should NOT print `Missing active params ... using defaults` when a tuned run exists and is active/auto-selected.

If you see `Missing active params ... using defaults` while tuned runs exist, run `tuning-status` to verify activation and use `activate-tuning` to promote the desired run. If there are no actives yet, bootstrap them:

```bash
python -m src.cli.pipeline bootstrap-market-actives --sport nhl --season 2025-26 --model all
```


## Bet Tracking Suite

The bet-tracking flow layers on top of the core pipelines and adds OCR ingestion, staging review, bet logging, closing-line (CLV) snapshots, settlement, and formatted reporting. Use these steps to turn screenshot dumps into structured logs and weekly/monthly summaries:
See the daily checklist in [docs/daily-workflow.md](docs/daily-workflow.md).

1. **Capture sportsbook boards via OCR.**
   ```bash
   python -m src.cli.pipeline market-ocr \
     --sport nba --season 2025-26 \
     --input screenshots/2024-12-01 \
     --book dn
   ```
   - If `--json-output path.json` is provided, no DB writes occur; instead a
     structured JSON file is emitted that contains every parsed moneyline/spread/total row.
   - If `--captured-at` is omitted, the pipeline records a per-image ISO
     timestamp when each image is read and uses that timestamp for JSON output
     and DB staging rows.
   
   ```
   # Example (JSON-only dry run)
   python -m src.cli.pipeline market-ocr --sport nba --season 2025-26 \
     --images screenshots/2024-12-01 --book dn --json-output tmp/lines.json
   ```
   - Accepts individual files or directories. Without `--json-output`, rows are
     persisted into `market_snapshot_staging` for reconciliation against the
     schedule database.

2. **Match OCR rows to games.**
   - Staging rows automatically call `resolve_staging_to_game()` to attempt a fuzzy match against the ingested schedule.
   - Use `python -m src.cli.pipeline market-review` to accept/reject anything left in `needs_review` status (see docs/market-review.md). Matched rows gain a `game_id`.

3. **Pivot reviewed rows into bets.**
   ```bash
   python -m src.cli.pipeline market-bets --sport nba --season 2025-26 --stake-preset unit --unit-stake 1.0
   ```
   - Auto-detects duplicate markets within the same screenshot (`hold_reason=duplicate_in_image`).
   - Uses stake presets (`half`, `unit`, `double`); see [docs/market-bets.md](docs/market-bets.md).

4. **Ingest closing lines (CLV) and backfill bets.**
   ```bash
   python -m src.cli.pipeline betting clv-csv --sport nba --season 2025-26 --csv data/raw/nba_closing_lines.csv --default-market-type ML
   ```
   - Resolves rows by `game_id` or by `team_home`/`team_away` + `game_date` and stores snapshots in `clv_snapshots`.
   - Updates matching bets with `clv_close_odds` / `clv_close_line` so reports and PnL scenarios carry closing numbers (details in [docs/market-clv.md](docs/market-clv.md)).

5. **Settle and report.**
   - Settle recorded bets against completed games: `python -m src.cli.pipeline betting settle-bets --sport nba --season 2025-26`.
   - Generate daily/weekly/monthly reports: `python -m src.cli.pipeline betting report --sport nba --season 2025-26 --type weekly --output outputs/reports/nba-weekly.xlsx`.
   - Workbooks include the main period sheet, `edge_buckets`, `clv`, `PnL_Scenarios`, and a lightweight dashboard.

The workflow is intentionally modular: you can pause after OCR (JSON only) when book lines need manual cleanup, then append finalized wagers into the DB, ingest closing lines later, and finally run settlement + reports.

To generate a review workbook from committed market snapshots, run `python -m src.cli.pipeline betting review-generate --sport <sport> --season <season> --model <model> --snapshot-run-id <run_id>` (or use `--snapshot-date` to filter by capture date). The command creates a review run, evaluates opportunities, and writes the workbook.

## CLI reference (detailed)

### Common flags

- `--model-params` accepts a JSON object string; `--model-params-file` points to a JSON file. Provide only one. When multiple models run, the file may contain per-model keys.
- If multiple models are run at once, outputs are prefixed with their abbreviations (bt, elo, gssd, toor).

### `import` — ingest into SQLite

```bash
python -m src.cli.pipeline import --sport nba --season 2025-26 --input data/raw/nba_schedule.html
```

- Required: `--sport`, `--season`, and one of `--input` (CSV/HTML path) or `--input-text` (pasted CSV text).
- Input resolution: if the provided path does not exist, the CLI also checks `data/raw/<value>`.
- Output: `data/db/<sport>/<season>.db` unless `--db` overrides it.

### `rank` — build power ratings and store calibration

```bash
python -m src.cli.pipeline rank --sport nba --season 2025-26
python -m src.cli.pipeline rank --sport nba --season 2025-26 --model elo --output data/processed/nba/2025-26/custom_rankings.csv
```

- Default runs **all** available models; use `--model` to limit to one.
- Default output: `data/processed/<sport>/<season>/rankings.csv`. When multiple models run, each file is prefixed (e.g., `bt_rankings.csv`).
- Stores calibration metrics (`home_advantage`, `win_prob_k`, `base_total`, residual spreads/totals) in the SQLite DB for downstream projections.

### `schedule` — export played + upcoming games with projections

```bash
# Excel workbook (default)
python -m src.cli.pipeline schedule --sport nba --season 2025-26

# CSV (one file per model)
python -m src.cli.pipeline schedule --sport nba --season 2025-26 --output data/processed/nba/2025-26/schedule_with_projections.csv
```

- Default output: Excel workbook at `data/processed/<sport>/<season>/schedule_with_projections.xlsx` with one sheet per model plus a `dashboard` sheet for today’s games.
- The schedule workbook also includes a `BETS` sheet for manual bet entry plus a hidden `META` sheet so you can log bets without OCR:
  - `import -> rank -> schedule -> fill BETS -> betting log-bets -> settle-bets`
  - Optional filters: `--as-of-date YYYY-MM-DD` and `--bets-model <model>` for multi-model runs.
- If `--output` ends with `.csv`, a CSV is written for each model (prefixed when multiple models run).
- `--upcoming-only` limits the export to games without scores.
- Each row includes schedule fields (`date`, `home_team`, `away_team`, `neutral`, `overtime`, `game_id`), projections (ratings, spreads, totals, win probabilities), calibration info (home advantage, uncertainty), and results when scores exist.

### `matchup` — predict one matchup

```bash
python -m src.cli.pipeline matchup --sport nba --season 2025-26 --matchup "Lakers vs Celtics"
python -m src.cli.pipeline matchup --sport nba --season 2025-26 --home Lakers --away Celtics --model toor
```

- Uses stored games/rankings to produce winner, spread, total, win probability, and uncertainty bands.
- Requires the teams to exist in the ingested dataset (run `rank` after new imports before predicting).

### `report` — rankings-only Excel workbook

```bash
python -m src.cli.pipeline report --sport nba --season 2025-26 --models bradley-terry,elo
```

- Default output: `data/processed/<sport>/<season>/report.xlsx` (one sheet per model). Prefixed filenames when multiple models are written.

### `market-ocr` — capture sportsbook boards

```bash
python -m src.cli.pipeline market-ocr \
  --sport nba --season 2025-26 \
  --input screenshots/2024-12-01 \
  --book dn \
  --captured-at 2024-12-01T14:30:00Z \
  --json-output tmp/lines-2024-12-01.json
```

- Accepts `--input` (file/dir), multiple `--input` values, or `--input-dir`.
- Set `--json-output` to emit parsed records without touching SQLite. The JSON contains the raw text, inferred market type (ML/spread/total), odds, line, and detected home/away teams.
- When JSON output is omitted, rows are inserted into `market_snapshot_staging` with fuzzy team matching results (`match_status`, `match_confidence`).
- Requires Tesseract to be installed; the OCR wrapper auto-detects common Windows install paths when it is not on `PATH`.

### `bet-report` — aggregate logged bets

```bash
python -m src.cli.pipeline bet-report \
  --sport nba --season 2025-26 \
  --type weekly \
  --start 2024-12-01 --end 2024-12-31 \
  --format xlsx \
  --output outputs/reports/bets-nba-dec.xlsx
```

- `--type` controls the aggregation window (`daily`, `weekly`, or `monthly`).
- CSV output mirrors the aggregation table. `.xlsx` output calls `write_full_report_xlsx()` to produce a workbook containing the main table, `edge_buckets`, `clv`, and a KPI dashboard (stake totals, win rate, EV, etc.).
- `--start` / `--end` bounds the reporting window; omit both to auto-span every logged bet.
- Reports pull from the same SQLite DB used for schedule + betting pipelines, so make sure your ingest/rank runs targeted the same `--sport/--season` first.

### `backtest` — evaluate models on historical games

Backtesting evaluates model predictions against actual outcomes across a historical period. The backtest runner fits the model on all games before each evaluation date, then generates predictions for that day's games. This produces accuracy metrics (log loss, Brier score, margin MAE) and calibration tables showing how well predicted win probabilities match actual outcomes.

**Basic usage:**

```bash
python -m src.cli.pipeline backtest --model bradley_terry_hfa --csv nba_results.csv --start 2024-11-01 --end 2024-12-01
```

**Options:**

- `--model`: Model to evaluate (default: `bradley_terry_hfa`). Supported: `bradley_terry_hfa`, `bradley_terry_calibrated_hfa`, `elo`, `gssd`, `toor`.
- `--csv`: Path to CSV containing historical games (required). Relative paths are resolved from the repo root.
- `--start` / `--end`: Evaluation window dates (YYYY-MM-DD) (required).
- `--window`: Training window type: `expanding` (default, all games before eval date) or `rolling` (fixed-size lookback).
- `--rolling-days`: Rolling window size in days (required when `--window rolling` is used).
- `--rolling-games`: Rolling window size in games (alternative to `--rolling-days` for rolling windows).
- `--model-params`: JSON string of model parameters (e.g., `'{"k_factor": 20}'`).
- `--model-params-file`: JSON file containing model parameters.
- `--output-dir`: Output directory override (default: `outputs/backtests/<model>/`).
- `--sport` / `--season` / `--db`: Optional; when provided, aggregate metrics and calibrated `win_prob_k` are persisted to the DB for use in projections.

**Output metrics:**

Each backtest produces four CSV files plus an Excel workbook:

1. **`predictions_<run_id>.csv`**: Per-game predictions with actual outcomes
   - Columns: `date`, `home_team`, `away_team`, `p_home_win`, `pred_margin`, `pred_total`, `home_score`, `away_score`, `home_win`, `actual_margin`, plus uncertainty bands and model metadata.

2. **`metrics_by_date_<run_id>.csv`**: Daily aggregate metrics
   - `date`, `games` (count), `log_loss`, `brier_score`, `mae_margin`, `margin_games`

3. **`metrics_overall_<run_id>.csv`**: Summary across the entire evaluation period
   - Single row with: `games`, `log_loss`, `brier_score`, `mae_margin`, `margin_games`, `model_id`
   - **Log loss**: Cross-entropy loss for win probability predictions (lower is better; measures probability calibration).
   - **Brier score**: Mean squared error of win probability predictions (0 = perfect, 0.25 = random guessing).
   - **MAE margin**: Mean absolute error of predicted point margins (lower is better; measures spread accuracy).

4. **`calibration_<run_id>.csv`**: Win probability calibration by bucket
   - Groups predictions by deciles (0-10%, 10-20%, ..., 90-100%) and compares average predicted vs actual win rates.
   - Columns: `bucket`, `count`, `avg_pred`, `avg_actual`. A well-calibrated model has `avg_pred ≈ avg_actual` in each bucket.

**Examples:**

```bash
# Expanding window backtest (default)
python -m src.cli.pipeline backtest \
  --csv data/raw/nba_2024_25.csv \
  --model bradley_terry_hfa \
  --start 2024-11-01 \
  --end 2024-12-01

# Rolling window backtest (last 30 days only)
## Overview

What it does:

1) Ingest Sports-Reference HTML/CSV (or pasted CSV text) and normalize it.
2) Persist games into per-sport/per-season SQLite DBs.
3) Fit power ratings (Bradley-Terry, Elo, GSSD, TOOR, Poisson) and store calibration metrics.
4) Generate matchup projections and daily schedules with spreads/totals/win probabilities.
5) Export rankings and schedule/report workbooks.
6) Backtest and tune models on historical CSVs.
  --rolling-days 30

# Rolling window by games (last 100 games only)
python -m src.cli.pipeline backtest \
  --csv data/raw/nba_2024_25.csv \
  --model toor \
  --start 2024-11-01 \
  --end 2024-12-01 \
  --window rolling \
  --rolling-games 100

  --model-params '{"k_factor": 20, "home_advantage": 60}'
  --csv nba_results.csv \
## Installation

1) Create a virtual environment
  --start 2024-11-01 \
  --end 2024-12-01 \
  --sport nba \
2) Activate it

- Windows (PowerShell): `./.pyenv/Scripts/Activate.ps1`
- macOS/Linux: `source .pyenv/bin/activate`
python -m src.cli.pipeline backtest \
  --csv nba_results.csv \
  --model bradley_terry_calibrated_hfa \
3) Install runtime dependencies

```bash
pip install -r requirements.txt
```
  --end 2024-12-01 \
  --output-dir outputs/custom_backtest_run
```
4) (Optional) Install test/tooling extras

```bash
pip install -r requirements-dev.txt
```

## Data Locations

- Default DB: `data/db/<sport>/<season>.db`
- Default processed outputs: `data/processed/<sport>/<season>/`
- Raw inputs: any path you provide; bare filenames are resolved under `data/raw/`.
- Input CSV/HTML should be straight from Sports-Reference; future games (no scores) are allowed.
- Backtest CSVs must contain `date`, `home_team`, `away_team`, `home_score`, `away_score` (common aliases are auto-detected). Optional: `neutral`, `overtime`, `game_id`.
- The evaluation window (`--start` / `--end`) must overlap the CSV data or no evaluations will run.
## Quickstart

All commands run via `python -m src.cli.pipeline <command> ...`.

Example for NBA 2025-26 with a CSV at `data/raw/nba_2025_26.csv`:

```bash
# 1) Ingest into SQLite (creates data/db/nba/2025-26.db by default)
python -m src.cli.pipeline import --sport nba --season 2025-26 --input data/raw/nba_2025_26.csv

# 2) Build rankings (runs all available models when --model is omitted)
python -m src.cli.pipeline rank --sport nba --season 2025-26

# 3) Export schedule projections (Excel workbook + dashboard by default)
python -m src.cli.pipeline schedule --sport nba --season 2025-26

# 4) Predict a single matchup
python -m src.cli.pipeline matchup --sport nba --season 2025-26 --matchup "Lakers vs Celtics"

# 5) Generate a rankings-only Excel report
python -m src.cli.pipeline report --sport nba --season 2025-26

# 6) Backtest a model on historical games
python -m src.cli.pipeline backtest --csv nba_results.csv --model bradley_terry_hfa --start 2024-11-01 --end 2024-12-01

# 7) Tune model hyperparameters via repeated backtests
python -m src.cli.pipeline tune --model elo --csv nba_results.csv --start 2024-11-01 --end 2024-12-01 --metric log_loss
```
**Notes:**
## Workflows

- Ingest: import Sports-Reference CSV/HTML to SQLite.
- Rank: build power ratings and store calibration.
- Schedule: export played + upcoming games with projections.
- Matchup: predict a single matchup.
- Report: rankings-only Excel workbook.
- Backtest: evaluate models on historical games.
- Tune: grid-search hyperparameters via backtests.
- Daily checklist: step-by-step runbook for ingest → bets → reports in [docs/daily-workflow.md](docs/daily-workflow.md).
```bash
## CLI Reference

See the complete command and option documentation in [docs/CLI.md](docs/CLI.md).
```
- Output directory defaults to `outputs/tuning/<sport>/<season>/<model>/<metric>/` (or `outputs/tuning/<model>/<metric>/` when sport/season are omitted). Artifacts per run: `tuning_results_<run_id>.csv`, `best_params_<run_id>.json`, and per-candidate backtest outputs.
- `--model all` tunes all backtest models; `--metric all` tunes all metrics; use `--fail-fast` to stop on the first failure.
- `--apply-best` reruns the best candidate and persists calibrated metrics when it beats the default-parameter baseline (disable the guard with `--allow-worse`). When tuning multiple metrics, `--apply-metric` chooses which metric becomes the active tuned parameters in the DB.
- Tuned parameters are persisted per metric and loaded automatically by rank/schedule/matchup runs; override with `--model-params` or `--model-params-file`, or select a specific tuned metric with `--tuned-metric`.

## Model parameter overrides (JSON examples)

## Model Parameter Overrides (JSON examples)

- Inline: `--model-params '{"k_factor": 20, "home_advantage": 60}'`
- File: create `params.json`:

```json
{
  "elo": {"k_factor": 20, "home_advantage": 60},
  "bradley-terry": {"max_iter": 500}
}
```

Then run: `python -m src.cli.pipeline rank --sport nba --season 2025-26 --model-params-file params.json`
- Inline: `--model-params '{"k_factor": 20, "home_advantage": 60}'`
## Input Format Notes

- Recognized columns: `Date` / `Game Date`; `Visitor/Neutral` / `Away`; `Home/Neutral` / `Home`; `PTS` or `PTS_away` / `PTS_home`; `OT` / `Overtime`; `Box Score`.
- If a CSV contains an unlabeled start-time column, the parser realigns columns automatically.
- Future games are accepted; rankings and projections require at least one completed game per team to be meaningful.
{
## Outputs At A Glance

- SQLite DB: `data/db/<sport>/<season>.db` (`games`, `model_metrics`, optional `backtest_metrics`).
- Rankings: `data/processed/<sport>/<season>/rankings.csv` (prefixed when multiple models run).
- Schedule: `data/processed/<sport>/<season>/schedule_with_projections.xlsx` (default) or `.csv` (per model).
- Rankings report: `data/processed/<sport>/<season>/report.xlsx` (per model when multiple).
- Backtests: `outputs/backtests/<model>/` CSVs + Excel per run.
- Tuning: `outputs/tuning/<sport>/<season>/<model>/<metric>/` (or `outputs/tuning/<model>/<metric>/`) grid CSV, best params JSON, and per-candidate backtests.

## Configuration

- `src/config.py` holds global defaults such as `DEFAULT_WIN_PROB_K`, calibration sample sizes, and fallback spread/total uncertainty.

- If a CSV contains an unlabeled start-time column, the parser realigns columns automatically.
- Future games are accepted; rankings and projections require at least one completed game per team to be meaningful.


- SQLite DB: `data/db/<sport>/<season>.db` (`games`, `model_metrics`, optional `backtest_metrics`).
- Rankings: `data/processed/<sport>/<season>/rankings.csv` (prefixed when multiple models run).
- Rankings report: `data/processed/<sport>/<season>/report.xlsx` (per model when multiple).
- Backtests: `outputs/backtests/<model>/` CSVs + Excel per run.
- Tuning: `outputs/tuning/<sport>/<season>/<model>/<metric>/` (or `outputs/tuning/<model>/<metric>/`) grid CSV, best params JSON, and per-candidate backtests.

## Testing

Run everything:

```bash
python -m pytest
```

Run only fast tests:

```bash
python -m pytest -q -m "not slow"
```

Coverage:

```bash
coverage run -m pytest -q
coverage report --fail-under=70
```

See [TESTING.md](TESTING.md) for more detail. `make test` is also available as a shortcut.

## Troubleshooting

- Missing or misnamed columns: ensure the CSV has date + home/away + scores; common aliases are auto-mapped.
- Unknown teams during `matchup`: rerun `rank` after new ingests so ratings exist for those teams.
- No completed games: rankings/projections require finished games; ingest more results or allow future games only for schedule exports.
- Overwriting outputs: pass `--overwrite` for rankings; for schedule/report/backtest/tune, specify a new `--output`/`--output-dir` or let the CLI append numeric suffixes when available.
Run everything:
## License

This project is provided as-is for internal analytics workflows.
```bash
python -m pytest
```

Run only fast tests:

```bash
python -m pytest -q -m "not slow"
```

Coverage:

```bash
coverage run -m pytest -q
coverage report --fail-under=70
```

`make test` is also available as a shortcut.

## Troubleshooting

- Missing or misnamed columns: ensure the CSV has date + home/away + scores; common aliases are auto-mapped.
- Unknown teams during `matchup`: rerun `rank` after new ingests so ratings exist for those teams.
- No completed games: rankings/projections require finished games; ingest more results or allow future games only for schedule exports.
- Overwriting outputs: pass `--overwrite` for rankings; for schedule/report/backtest/tune, specify a new `--output`/`--output-dir` or allow the CLI to append numeric suffixes when available.

## License

This project is provided as-is for internal analytics workflows.
