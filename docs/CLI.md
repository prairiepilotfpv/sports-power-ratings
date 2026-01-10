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

# Generate a review workbook for a model (requires committed market snapshots)
python -m src.cli.pipeline betting review-generate --sport nba --season 2025-26 --model elo --snapshot-run-id run_20251201 --output-dir outputs/review

# Log bets from a workbook (write back bet_id/logged_at to the workbook)
python -m src.cli.pipeline betting log-bets --workbook outputs/review/nba-2025-26-review.xlsx --db data/db/nba/2025-26.db --writeback

# Settle bets up to completed games
python -m src.cli.pipeline betting settle-bets --sport nba --season 2025-26 --db data/db/nba/2025-26.db

# Import closing lines/odds and backfill CLV
python -m src.cli.pipeline betting clv-csv --sport nba --season 2025-26 --csv data/raw/nba_closing_lines.csv --default-market-type ML

# Generate aggregated betting reports (daily/weekly/monthly)
python -m src.cli.pipeline betting report --sport nba --season 2025-26 --type weekly --start 2025-11-01 --end 2026-01-01 --output outputs/reports/nba-weekly.xlsx

# Output CSV instead of Excel (use --format or .csv extension)
python -m src.cli.pipeline betting report --sport nba --season 2025-26 --type monthly --format csv --output outputs/reports/nba-monthly.csv

# End-to-end OCR ingest to report
python -m src.cli.pipeline betting market-ocr --sport nba --season 2025-26 --images screenshots/ --book DK --captured-at 2025-12-01T14:30:00Z --json-output tmp/lines.json
# Review/accept matches (optional when using DB mode)
python -m src.cli.pipeline market-review --sport nba --season 2025-26 --status all --limit 20
# Commit staging to market_snapshots (DB mode only)
python -m src.cli.pipeline betting market-commit --sport nba --season 2025-26 --snapshot-run-id run_20251201
# Generate weekly report
python -m src.cli.pipeline betting report --sport nba --season 2025-26 --type weekly --start 2025-12-01 --end 2025-12-31 --format xlsx --output outputs/reports/bets-nba-dec.xlsx
```

Notes:
- `--type` chooses aggregation period: `daily` (default), `weekly`, or `monthly`.
- `--start` / `--end` filter the date range for weekly/monthly aggregations (YYYY-MM-DD).
- `--format` overrides output type; otherwise the CLI infers format from the `--output` extension.
- Betting report workbooks include three sheets: the main period sheet (`daily`, `weekly`, or `monthly`), an `edge_buckets` summary, and a `clv` summary.
- `clv-csv` resolves games via `game_id` or `team_home`/`team_away` + `game_date`; invalid rows are skipped and counted. See [docs/market-clv.md](docs/market-clv.md) for the expected CSV schema and flags.
- `review-generate` requires `--snapshot-run-id` (or `--snapshot-date`) so opportunities are built from committed market snapshots.

Manual review workbook flow:
1. Generate a review workbook from committed market snapshots (creates `EV`, `BETS`, and `META` sheets):
   ```bash
   python -m src.cli.pipeline betting review-generate --sport nba --season 2025-26 --model elo --snapshot-run-id run_20251201 --output-dir outputs/review
   ```
2. Fill the `BETS` sheet with `stake`, `book`, and `price` (leave `stake` blank to pass).
3. Log bets and write back `bet_id`/`logged_at` to the workbook:
   ```bash
   python -m src.cli.pipeline betting log-bets --workbook outputs/review/nba-2025-26-review.xlsx --db data/db/nba/2025-26.db --writeback
   ```
4. Run `betting settle-bets` and `betting report` to track results and summaries.

## market-ocr — OCR ingest quick reference

```bash
# Parse images and write JSON only (no DB writes)
python -m src.cli.pipeline betting market-ocr --sport nba --season 2025-26 --images screenshots/ --book DK --captured-at 2025-12-01T14:30:00Z --json-output tmp/lines.json

# Parse and write staging rows to DB (omit --json-output)
python -m src.cli.pipeline betting market-ocr --sport nba --season 2025-26 --images screenshots/ --book DK --captured-at 2025-12-01T14:30:00Z --db data/db/nba/2025-26.db
```

Notes: `--json-output` switches the command to JSON-only mode (no DB writes). Without it, rows go to `market_snapshot_staging` for later review/commit.

## Legacy ingest entry point (deprecated)

`python -m src.cli.ingest` is retained for backward compatibility but is deprecated. Prefer the unified pipeline command:

```bash
python -m src.cli.pipeline import --sport nba --season 2025-26 --input data/raw/nba.csv
```

## market-review — review staging rows

List or resolve OCR/CSV staging rows stored in `market_snapshot_staging`.

```bash
# List pending rows (default: needs_review only)
python -m src.cli.pipeline market-review --sport nba --season 2025-26

# Show all statuses and limit to 20 rows
python -m src.cli.pipeline market-review --sport nba --season 2025-26 --status all --limit 20

# Accept or reject a specific staging row
python -m src.cli.pipeline market-review --sport nba --season 2025-26 --accept 12 --game-id 2024-12-01-lal-lac --match-confidence 0.95
python -m src.cli.pipeline market-review --sport nba --season 2025-26 --reject 12
```

Notes: `--status` accepts comma-separated values (e.g., `matched,needs_review`); `--game-id` is required when accepting.

## market-bets — pivot reviewed staging rows into bets

Convert matched staging rows into `bets` entries with simple stake presets and duplicate detection.

```bash
# Insert bets with default unit stake (1.0) and preset "unit"
python -m src.cli.pipeline market-bets --sport nba --season 2025-26 --status matched

# Use custom review_run_id, double-unit stakes, and fallback book name
python -m src.cli.pipeline market-bets --sport nba --season 2025-26 --review-run-id rr-2025-12-01 --stake-preset double --unit-stake 2.0 --default-book DraftKings

# Dry-run to see counts without writing
python -m src.cli.pipeline market-bets --sport nba --season 2025-26 --dry-run
```

Behaviors:
- Filters by `match_status` (default: `matched`; use `--status all` to include everything).
- Stake presets: `half` (0.5x), `unit` (1.0x), `double` (2.0x) multiplied by `--unit-stake`.
- Auto-hold detection skips duplicate markets from the same image (disable with `--disable-auto-hold`). Held rows are tagged in `hold_reason`.
- If a CLV snapshot exists for the same game/market/selection, the bet is populated with `clv_close_odds` / `clv_close_line`.

See [docs/market-bets.md](docs/market-bets.md) for a fuller walkthrough.

## clv-csv — ingest closing lines and update bets

Import closing lines/odds, store them in `clv_snapshots`, and (by default) apply the latest snapshot to matching bets so reports contain closing numbers.

```bash
python -m src.cli.pipeline betting clv-csv --sport nba --season 2025-26 --csv data/raw/nba_closing_lines.csv --default-market-type ML
```

Flags:
- `--csv`: path to the closing-lines CSV. Required.
- `--default-market-type`: fallback when the CSV omits `market_type`.
- `--captured-at`: override captured_at for all rows.
- `--no-update-bets`: skip backfilling `bets.clv_close_odds` / `bets.clv_close_line`.

CSV expectations (aliases allowed): `market_type`, `selection`, `close_odds` (or `odds`), optional `close_line`; either `game_id` or `team_home`/`team_away` + `game_date` is required. Invalid odds or missing keys are rejected and counted. See [docs/market-clv.md](docs/market-clv.md) for details.

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
