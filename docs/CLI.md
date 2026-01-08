# CLI Reference

This document provides detailed usage for the pipeline CLI covering ingest, rankings, schedules, matchups, reports, backtests, and tuning.

All commands run via:

```bash
python -m src.cli.pipeline <command> [options]
```

## Common Flags

- `--sport`: One of `nba`, `nhl`, `cbb` (as supported).
- `--season`: Season identifier like `2025-26`.
- `--model`: One of `bradley_terry_hfa`, `bradley_terry_calibrated_hfa`, `elo`, `gssd`, `toor`.
- `--model-params`: JSON string of model params, e.g. `'{"k_factor": 20}'`.
  - Example: to suppress small-sigma warnings for `bradley_terry_hfa` (useful for low-scoring sports like `nhl`) pass `--model-params '{"suppress_small_sd_warning": true}'`.
- `--model-params-file`: Path to JSON file with params (supports per-model keys when multiple models run).
- `--output`/`--output-dir`: Override default output paths.
- `--db`: Override default SQLite path.

When multiple models run, outputs are prefixed with abbreviations (`bt`, `elo`, `gssd`, `toor`).

## import — ingest into SQLite

```bash
python -m src.cli.pipeline import --sport nba --season 2025-26 --input data/raw/nba_schedule.html
```

- Required: `--sport`, `--season`, and one of `--input` (CSV/HTML path) or `--input-text` (pasted CSV text).
- Input resolution: if path does not exist, the CLI also checks `data/raw/<value>`.
- Output: `data/db/<sport>/<season>.db` unless `--db` overrides it.

## rank — build power ratings and store calibration

```bash
python -m src.cli.pipeline rank --sport nba --season 2025-26
python -m src.cli.pipeline rank --sport nba --season 2025-26 --model elo --output data/processed/nba/2025-26/custom_rankings.csv
```

- Default runs all available models; use `--model` to select one.
- Default output: `data/processed/<sport>/<season>/rankings.csv`. When multiple models run, each file is prefixed (e.g., `bt_rankings.csv`).
- Stores calibration metrics (`home_advantage`, `win_prob_k`, `base_total`, residual spreads/totals) in the DB.

## schedule — export played + upcoming games with projections

```bash
# Excel workbook (default)
python -m src.cli.pipeline schedule --sport nba --season 2025-26

# CSV (one file per model)
python -m src.cli.pipeline schedule --sport nba --season 2025-26 --output data/processed/nba/2025-26/schedule_with_projections.csv
```

- Default output: `data/processed/<sport>/<season>/schedule_with_projections.xlsx` with one sheet per model plus a `dashboard` sheet.
- If `--output` ends with `.csv`, writes CSV per model (prefixed when multiple models run).
- `--upcoming-only` limits export to games without scores.
- Rows include schedule fields, projections (ratings, spreads, totals, win probabilities), calibration info, and results when scores exist.

## matchup — predict one matchup

```bash
python -m src.cli.pipeline matchup --sport nba --season 2025-26 --matchup "Lakers vs Celtics"
python -m src.cli.pipeline matchup --sport nba --season 2025-26 --home Lakers --away Celtics --model toor
```

- Produces winner, spread, total, win probability, and uncertainty.
- Requires teams to exist in the ingested dataset (run `rank` after new imports).

## report — rankings-only Excel workbook

```bash
python -m src.cli.pipeline report --sport nba --season 2025-26 --models bradley-terry,elo
```

- Default output: `data/processed/<sport>/<season>/report.xlsx` (one sheet per model). Prefixed filenames when multiple models are written.


## betting — market ingestion, review, bets logging, settlement, and reports

Betting commands are grouped under the `betting` command and include OCR ingestion, review workbook generation, logging bets from a workbook, settling bets, and aggregated reports.

Examples:

```bash
# Ingest market screenshots (single image or directory)
python -m src.cli.pipeline betting market-ocr --sport nba --season 2025-26 --images path/to/screenshots --book DraftKings

# Generate a review workbook for a model
python -m src.cli.pipeline betting review-generate --sport nba --season 2025-26 --model elo --output-dir outputs/review

# Log bets from a workbook (write back bet_id/logged_at to the workbook)
python -m src.cli.pipeline betting log-bets --workbook outputs/review/nba-2025-26-review.xlsx --db data/db/nba/2025-26.db --writeback

# Settle bets up to completed games
python -m src.cli.pipeline betting settle-bets --sport nba --season 2025-26 --db data/db/nba/2025-26.db

# Generate aggregated betting reports (daily/weekly/monthly)
python -m src.cli.pipeline betting report --sport nba --season 2025-26 --type weekly --start 2025-11-01 --end 2026-01-01 --output outputs/reports/nba-weekly.xlsx

# Output CSV instead of Excel (use --format or .csv extension)
python -m src.cli.pipeline betting report --sport nba --season 2025-26 --type monthly --format csv --output outputs/reports/nba-monthly.csv
```

Notes:
- `--type` chooses aggregation period: `daily` (default), `weekly`, or `monthly`.
- `--start` / `--end` filter the date range for weekly/monthly aggregations (YYYY-MM-DD).
- `--format` overrides output type; otherwise the CLI infers format from the `--output` extension.
- Betting report workbooks include three sheets: the main period sheet (`daily`, `weekly`, or `monthly`), an `edge_buckets` summary, and a `clv` summary.

## backtest — evaluate models on historical games

Backtesting fits the model on all games before each evaluation date, then predicts that day's games to generate metrics and calibration tables.

```bash
python -m src.cli.pipeline backtest --model bradley_terry_hfa --csv nba_results.csv --start 2024-11-01 --end 2024-12-01
```

Options:

- `--model`: Model to evaluate (default: `bradley_terry_hfa`). Supported: `bradley_terry_hfa`, `bradley_terry_calibrated_hfa`, `elo`, `gssd`, `toor`.
- `--csv`: Path to CSV containing historical games (required). Relative paths resolved from repo root.
- `--start` / `--end`: Evaluation window dates (YYYY-MM-DD) (required).
- `--window`: `expanding` (default) or `rolling`.
- `--rolling-days` or `--rolling-games`: Size for rolling window.
- `--model-params` / `--model-params-file`: As above.
  - Note: model-specific options are supported; for example `bradley_terry_hfa` accepts `suppress_small_sd_warning` (boolean) to silence warnings about small learned margin/total sigmas for low-scoring sports. Default: `false`.
- `--output-dir`: Override default directory (`outputs/backtests/<model>/`).
- `--sport` / `--season` / `--db`: Optional persistence of metrics to DB for use in projections.

Outputs:

1. `predictions_<run_id>.csv`: Per-game predictions and actuals.
2. `metrics_by_date_<run_id>.csv`: Daily aggregate metrics.
3. `metrics_overall_<run_id>.csv`: Summary across the entire period (log loss, Brier, MAE margin).
4. `calibration_<run_id>.csv`: Win probability calibration buckets.

Examples:

```bash
# Expanding window backtest (default)
python -m src.cli.pipeline backtest \
  --csv data/raw/nba_2024_25.csv \
  --model bradley_terry_hfa \
  --start 2024-11-01 \
  --end 2024-12-01

# Rolling window by days
python -m src.cli.pipeline backtest \
  --csv data/raw/nba_2024_25.csv \
  --model elo \
  --start 2024-11-01 \
  --end 2024-12-01 \
  --window rolling \
  --rolling-days 30

# Rolling window by games
python -m src.cli.pipeline backtest \
  --csv data/raw/nba_2024_25.csv \
  --model toor \
  --start 2024-11-01 \
  --end 2024-12-01 \
  --window rolling \
  --rolling-games 100

# Backtest with custom params
python -m src.cli.pipeline backtest \
  --csv nba_results.csv \
  --model elo \
  --start 2024-11-01 \
  --end 2024-12-01 \
  --model-params '{"k_factor": 20, "home_advantage": 60}'

# Persist metrics to DB
python -m src.cli.pipeline backtest \
  --csv nba_results.csv \
  --model bradley_terry_hfa \
  --start 2024-11-01 \
  --end 2024-12-01 \
  --sport nba \
  --season 2024-25 \
  --db data/db/nba/2024-25.db

# Custom output directory
python -m src.cli.pipeline backtest \
  --csv nba_results.csv \
  --model bradley_terry_calibrated_hfa \
  --start 2024-11-01 \
  --end 2024-12-01 \
  --output-dir outputs/custom_backtest_run
```

Input CSV requirements:

- Required: `date`, `home_team`, `away_team`, `home_score`, `away_score`.
- Optional: `neutral`, `overtime`, `game_id`.
- Dates must be YYYY-MM-DD; scores numeric and non-negative.

Notes:

- CSV parsing is lenient; missing/duplicate game IDs are rebuilt when needed.
- Re-running the same model/window overwrites files in the same `--output-dir` unless you change it.
- With `--sport`, `--season`, and `--db`, backtests persist calibrated metrics for downstream projections.

## tune — grid-search hyperparameters via backtests

```bash
python -m src.cli.pipeline tune --model elo --csv nba_results.csv --start 2024-11-01 --end 2024-12-01 --metric brier_score --apply-best --sport nba --season 2025-26 --db data/db/nba/2025-26.db
```

- Grid source: built-in defaults per model; override with `--grid-file` (single grid or per-model object).
- Default output: `outputs/tuning/<sport>/<season>/<model>/<metric>/` (or `outputs/tuning/<model>/<metric>/`).
- Artifacts per run: `tuning_results_<run_id>.csv`, `best_params_<run_id>.json`, and per-candidate backtests.
- `--model all` tunes all backtest models; `--metric all` tunes all metrics; use `--fail-fast` to stop on the first failure.
- `--apply-best` persists best candidate metrics when it beats baseline (disable guard with `--allow-worse`). Use `--apply-metric` when tuning multiple metrics.
- Tuned parameters are auto-loaded by rank/schedule/matchup; override via `--model-params` or choose a tuned metric with `--tuned-metric`.
