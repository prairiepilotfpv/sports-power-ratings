# CLI Reference

This document provides detailed usage for the pipeline CLI covering ingest, rankings, schedules, matchups, reports, backtests, and tuning.

All commands run via:

```bash
python -m src.cli.pipeline <command> [options]
```

## Common Flags

  - Example: to suppress small-sigma warnings for models that expose this option (useful for low-scoring sports like `nhl`) pass `--model-params '{"suppress_small_sd_warning": true}'`.

### Activating and verifying tuned runs

- After tuning you may optionally promote a tuned run to be the explicit active params used by `rank`/`schedule` with `activate-tuning`:

```bash
python -m src.cli.pipeline activate-tuning --sport <sport> --season <season> --model <model> --market <ML|SPREAD|TOTAL> --run-id <run_id>
```

- To inspect what params/weights `rank`/`schedule` will use without changing DB state, run the new read-only command `tuning-status`:

```bash
python -m src.cli.pipeline tuning-status --sport <sport> --season <season>
```

This prints, per market, each model's status as `ACTIVE` (explicitly promoted), `AUTO-SELECT` (best tuning run auto-selected), or `DEFAULT` (no tuned params found).
When multiple models run, outputs are prefixed with abbreviations (`bt`, `elo`, `gssd`, `toor`).

## Development notes
- `schema_meta` tracks the current schema version; both `src/data/repository.py:init_db` and
- Legacy DB upgrades are additive: migrations check for missing columns/tables and backfill with `ALTER TABLE`
- When adding new migrations, append a new versioned entry in `MIGRATIONS` and keep it idempotent.

## import - ingest into SQLite

```bash
python -m src.cli.pipeline import --sport nba --season 2025-26 --input data/raw/nba_schedule.html
```

- Required: `--sport`, `--season`, and one of `--input` (CSV/HTML path) or `--input-text` (pasted CSV text).
- Input resolution: if path does not exist, the CLI also checks `data/raw/<value>`.
- Output: `data/db/<sport>/<season>.db` unless `--db` overrides it.

## action-paste — convert Action paste blocks into CSV or Excel-paste text

When you have the middle section copied from Action (matchup table), use the deterministic parser to produce either a CSV compatible with the staging workflow or the exact 6-lines-per-game tab-separated block you paste into Excel columns H:I.

CSV (writes one row per required paste line — 6 rows per game):

```bash
python -m src.cli.action_paste --in markettest.txt --out outputs/paste_parsed/mymarkets.csv
```

Excel-paste block (exactly 6 lines per game, each `line<TAB>odds`):

```bash
python tools/action_to_bets_paste.py --in markettest.txt --out bets_paste.txt
# or write to stdout for quick copy:
python tools/action_to_bets_paste.py --in markettest.txt
```

Optional: also persist opens (open spread/total) to JSON for later review:

```bash
python -m src.cli.action_paste --in markettest.txt --out outputs/paste_parsed/mymarkets.csv --include-opens-json opens.json
```

Notes:
- The parser expects header lines of the form `<AWAY> at <HOME> Odds` and the specific positional token layout described in `docs/action_paste.md`.
- The CSV produced matches the staging columns so you can paste or import into the existing workflows.


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
- When multiple models run, the workbook includes a `BETS` sheet with ensemble-aware columns:
  - ML: `home_win_prob`, `away_win_prob`, `win_prob_source`, `ml_ensemble_components_json`
  - SPREAD: `margin_mean`, `margin_sd`, `spread_source`, `spread_ensemble_components_json`
  - TOTAL: `total`, `total_sd`, `total_source`, `total_ensemble_components_json`
  - `market_forecast_source` records which source drove the row.

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
# Generate a formula-enabled review workbook (implied_prob/edge/ev formulas)
python -m src.cli.pipeline betting review-generate --sport nba --season 2025-26 --model elo --snapshot-run-id run_20251201 --output-dir outputs/review --formula-workbook

# Log bets from a workbook (write back bet_id/logged_at to the workbook)
python -m src.cli.pipeline betting log-bets --workbook outputs/review/nba-2025-26-review.xlsx --db data/db/nba/2025-26.db --writeback

# Settle bets up to completed games
python -m src.cli.pipeline betting settle-bets --sport nba --season 2025-26 --db data/db/nba/2025-26.db

**BETS Workflow (lines in workbook → DB)**

- **Generate schedule/workbook with `BETS` sheet:** the schedule/daily-workbook/review workbooks include a `BETS` sheet. Example:

```bash
# Generate the review workbook (schedule export with BETS). `--model` is optional — omit it to include default projections.
python -m src.cli.pipeline schedule --sport nba --season 2025-26 --output outputs/review/nba-2025-26-review.xlsx --db data/db/nba/2025-26.db
```

- **Populate missing `game_id` (optional):** if you have pasted market rows (OCR/paste) you can attempt to fill `game_id` from staging via the helper in `src/utils/review_helpers.py`:

```bash
python -m src.utils.review_helpers --workbook outputs/review/nba-2025-26-review.xlsx --db data/db/nba/2025-26.db
```

- **Edit the `BETS` sheet in Excel:** fill rows you want to log. Recognized columns:
  - **Required for logging:** `game_id`, `market_type`, `selection`, `stake` (non-empty)
  - **Lines/odds:** `line` and either `odds` or `price` (either accepted)
  - **Optional:** `book`, `opportunity_id`, `notes`

- **Dry-run parse the workbook (validate only):**

```bash
python -m src.cli.pipeline betting log-bets --workbook outputs/review/nba-2025-26-review.xlsx --db data/db/nba/2025-26.db --dry-run
```

- **Persist bets and workbook lines to DB (writeback):**

```bash
python -m src.cli.pipeline betting log-bets --workbook outputs/review/nba-2025-26-review.xlsx --db data/db/nba/2025-26.db --writeback
```

Notes on behavior:
- When generating the `BETS` sheet the pipeline will populate `line`, `odds`, and `source_market_snapshot_id` when a `market_snapshots` row exists for the game/market/selection as-of the workbook `as_of_date`. If none exist those cells are blank.
- When you run `log-bets --writeback` and a `BETS` row contains `line`/`odds` that differ from the latest `market_snapshots`, the system will insert a new `market_snapshots` row with `snapshot_run_id = review_run_id` and `book = 'manual'` (this is skipped for `--dry-run`). The `bets` row is then upserted as before and `bet_id`/`logged_at` will be written back into the workbook.

Quick DB checks (sqlite):

```bash
sqlite3 data/db/nba/2025-26.db "SELECT id, snapshot_run_id, book, market_type, selection, line, odds, game_id, captured_at FROM market_snapshots ORDER BY datetime(captured_at) DESC LIMIT 10;"

sqlite3 data/db/nba/2025-26.db "SELECT id, review_run_id, game_id, market_type, selection, line, odds, stake, logged_at FROM bets ORDER BY datetime(logged_at) DESC LIMIT 10;"
```

If you want, I can also add a short smoke-test script that automates: generate workbook → edit a single BETS row → run `log-bets --writeback` → assert DB rows exist. Ask and I will add it.

# Import closing lines/odds and backfill CLV
python -m src.cli.pipeline betting clv-csv --sport nba --season 2025-26 --csv data/raw/nba_closing_lines.csv --default-market-type ML

# Import market lines from CSV (commit matched rows automatically)
python -m src.cli.pipeline betting market-csv --sport nba --season 2025-26 --csv data/raw/nba_markets.csv --default-book dn

# Generate aggregated betting reports (daily/weekly/monthly)
python -m src.cli.pipeline betting report --sport nba --season 2025-26 --type weekly --start 2025-11-01 --end 2026-01-01 --output outputs/reports/nba-weekly.xlsx

# Output CSV instead of Excel (use --format or .csv extension)
python -m src.cli.pipeline betting report --sport nba --season 2025-26 --type monthly --format csv --output outputs/reports/nba-monthly.csv

# Pre-flight validation (DB integrity, predictions, snapshot counts)
python -m src.cli.pipeline betting validate --sport nba --season 2025-26 --model elo --date 2025-12-01 \
  --snapshot-run-id run_20251201 --min-snapshots 10

# End-to-end OCR ingest to report
python -m src.cli.pipeline betting market-ocr --sport nba --season 2025-26 --images screenshots/ --book DK --captured-at 2025-12-01T14:30:00Z --json-output tmp/lines.json
> ⚠️ `market-review` and `market-commit` have been retired and now raise an error directing you to `betting market-csv` + `market_lines`. Rely on `market-csv` for ingestion and skip these commands.

# Generate weekly report
python -m src.cli.pipeline betting report --sport nba --season 2025-26 --type weekly --start 2025-12-01 --end 2025-12-31 --format xlsx --output outputs/reports/bets-nba-dec.xlsx
```

Notes:
- `--type` chooses aggregation period: `daily` (default), `weekly`, or `monthly`.
- `--start` / `--end` filter the date range for weekly/monthly aggregations (YYYY-MM-DD).
- `--format` overrides output type; otherwise the CLI infers format from the `--output` extension.
- Betting report workbooks include three sheets: the main period sheet (`daily`, `weekly`, or `monthly`), an `edge_buckets` summary, and a `clv` summary.
- `clv-csv` resolves games via `game_id` or `team_home`/`team_away` + `game_date`; invalid rows are skipped and counted. See [docs/market-clv.md](docs/market-clv.md) for the expected CSV schema and flags.
- `market-csv` resolves team/date aliases, writes validated rows to `market_lines`, and reports unmatched rows with failure reasons/examples. Use `--date-filter` to scope the import to specific dates.
- `review-generate` requires `--snapshot-run-id`; optionally add `--snapshot-date` to constrain snapshots by captured date.
- `review-generate --formula-workbook` (alias `--formula`) writes formulas in the `EV` and `BETS` sheets for `implied_prob`, `edge`, and `ev`.
  - Formula-driven columns: `implied_prob`, `edge`, `ev`.
  - Editable inputs that drive formulas: `odds`, `line`, `model_prob` (plus other non-formula fields like `selection`/`market_type`).

Manual review workbook flow:
1. Generate a review workbook from committed market snapshots (creates `EV`, `BETS`, and `META` sheets):
   ```bash
   python -m src.cli.pipeline betting review-generate --sport nba --season 2025-26 --model elo --snapshot-run-id run_20251201 --output-dir outputs/review
   ```
   To generate a formula workbook, add `--formula-workbook` (or `--formula`).
   - Review workbooks also include an `EXCLUSIONS` sheet with guardrail reason codes; see [docs/bet-evaluation.md](docs/bet-evaluation.md) for context on why rows are filtered.
2. Fill the `BETS` sheet with `stake`, `book`, and `price` (leave `stake` blank to pass).
   - In formula workbooks, `implied_prob`, `edge`, and `ev` are formula-driven; edit `odds`, `line`, or `model_prob` in `BETS` to refresh the calculations.
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

## market-review — retired

`market-review` no longer supports manual staging review. The command now raises a `ValueError` telling you to use `betting market-csv` and `market_lines` for ingestion/diagnostics. The staging table still exists for compatibility, but every modern workflow should rely on the CSV import path instead.

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

## clv-csv - ingest closing lines and update bets

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

## market-csv - ingest market lines from CSV

Import market lines into the sport/season database. Each row must include `team_home{_raw}`, `team_away{_raw}`, `market_type`, `selection`, `line`, `odds`, and `game_date`. The command resolves aliases/dates, upserts valid rows into the new `market_lines` table, and reports unmatched rows grouped by failure reason while storing their serialized payloads in `market_line_import_errors`.

```bash
python -m src.cli.pipeline betting market-csv --sport nba --season 2025-26 \
  --csv data/raw/nba_markets.csv --default-book dn
```

Flags:
- `--csv`: path to the market CSV. Required.
- `--date-filter`: limit the import to this ISO date (YYYY-MM-DD); repeatable for multiple dates.
- `--default-book`: fallback book name when the CSV omits it.

CSV expectations: `market_type` (ML/spread/total), `selection`, `line` (nullable for ML), `odds`, `team_home{_raw}`, `team_away{_raw}`, `game_date`. Optional: `game_id`, `book`. Invalid or unmatched rows are counted, diagnosed, and stored for inspection. Market types are normalized from common aliases (e.g., `Money Line`, `ML`, `moneyline`), and team aliases are resolved using `data/config/team_aliases.json` during matching.

## backtest — evaluate models on historical games

Backtesting fits the model on all games before each evaluation date, then predicts that day's games to generate metrics and calibration tables.

```bash
python -m src.cli.pipeline backtest --model bradley-terry --csv nba_results.csv --start 2024-11-01 --end 2024-12-01
```

Options:

-- `--model`: Model to evaluate (default: `bradley-terry`). Supported: `bradley-terry`, `elo`, `gssd`, `toor`.
- `--csv`: Path to CSV containing historical games (required). Relative paths resolved from repo root.
- `--start` / `--end`: Evaluation window dates (YYYY-MM-DD) (required).
- `--window`: `expanding` (default) or `rolling`.
- `--rolling-days` or `--rolling-games`: Size for rolling window.
- `--model-params` / `--model-params-file`: As above.
  - Note: model-specific options are supported; check the model's `metadata()` for available params. Some historical wrapper variants exposed `suppress_small_sd_warning`.
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
  --model bradley-terry \
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
  --model bradley-terry \
  --start 2024-11-01 \
  --end 2024-12-01 \
  --sport nba \
  --season 2024-25 \
  --db data/db/nba/2024-25.db

# Custom output directory
python -m src.cli.pipeline backtest \
  --csv nba_results.csv \
  --model bradley-terry \
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

## init-ensemble-config — create default per-market configs

```bash
python -m src.cli.pipeline init-ensemble-config --sport nba --season 2025-26
```

- Writes per-market defaults to `outputs/ensembles/<sport>/<season>/<market>/default.json`.
- Use `--overwrite` to replace existing defaults.

## tune-ensemble — optimize ensemble weights for a market

```bash
python -m src.cli.pipeline tune-ensemble --sport nba --season 2025-26 --market ML --ensemble ensemble_ml_v1 \
  --start-date 2020-01-01 --end-date 2024-12-31 --csv data/raw/nba_history.csv
```

- Stores the tuning run in the DB and writes weights to `outputs/ensembles/<sport>/<season>/<market>/<ensemble_id>.json`.
- `tuning-status` reports whether active weights come from explicit DB actives, best runs, or defaults.

## calibrate — fit a probability calibrator for an ML source

```bash
python -m src.cli.pipeline calibrate --sport nba --season 2025-26 --market ML --source ensemble_ml_v1 \
  --start-date 2020-01-01 --end-date 2024-12-31 --csv data/raw/nba_history.csv
```

- Calibrators are written to `outputs/calibrators/<sport>/<season>/<source_id>/<market>/`.
