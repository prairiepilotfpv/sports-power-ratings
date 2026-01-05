# Sports Power Ratings

Local-first tooling to turn Sports-Reference schedules/results into SQLite databases, power ratings, projections, and reports. Everything runs through a single CLI and works entirely offline once inputs are downloaded.

## What it does

1) Ingest Sports-Reference HTML/CSV (or pasted CSV text) and normalize it. 
2) Persist games into per-sport/per-season SQLite DBs. 
3) Fit power ratings (Bradley-Terry, Elo, GSSD, TOOR) and store calibration metrics. 
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

## Requirements

- Python 3.11+ (tested locally)
- Tesseract OCR if you plan to ingest screenshots/images (optional)
- `OPENAI_API_KEY` / `OPENAI_MODEL` only if you want OCR assistance; not required otherwise

## Setup (explicit steps)

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

## Data locations and expectations

- Default DB: `data/db/<sport>/<season>.db`
- Default processed outputs: `data/processed/<sport>/<season>/`
- Raw inputs: any path you provide; bare filenames are resolved under `data/raw/`.
- Input CSV/HTML should be straight from Sports-Reference; future games (no scores) are allowed.
- Backtest CSVs must contain `date`, `home_team`, `away_team`, `home_score`, `away_score` (common aliases are auto-detected). Optional: `neutral`, `overtime`, `game_id`.

## Standard workflow (pipeline CLI)

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

### `backtest` — evaluate models on historical games

```bash
python -m src.cli.pipeline backtest --model bradley_terry_hfa --csv nba_results.csv --start 2024-11-01 --end 2024-12-01 --window expanding
```

- Supported models: bradley_terry_hfa, bradley_terry_calibrated_hfa, elo, gssd, toor.
- CSV parsing is lenient: common Sports-Reference column aliases are detected; missing/duplicate game IDs are rebuilt when needed.
- Output directory defaults to `outputs/backtests/<model>/`. Artifacts per run: `predictions_<run_id>.csv`, `metrics_by_date_<run_id>.csv`, `metrics_overall_<run_id>.csv`, `calibration_<run_id>.csv`, plus an Excel workbook.
- If `--sport`, `--season`, and `--db` (or default DB) are provided, aggregate metrics and calibrated `win_prob_k` are persisted to the DB.

### `tune` — grid-search hyperparameters via backtests

```bash
python -m src.cli.pipeline tune --model elo --csv nba_results.csv --start 2024-11-01 --end 2024-12-01 --metric brier_score --apply-best --sport nba --season 2025-26 --db data/db/nba/2025-26.db
```

- Grid source: built-in defaults per model; override with `--grid-file` (either a single grid or a per-model object).
- Output directory defaults to `outputs/tuning/<model>/`. Artifacts per run: `tuning_results_<run_id>.csv`, `best_params_<run_id>.json`, and per-candidate backtest outputs.
- `--apply-best` reruns the best candidate and persists calibration metrics when it beats the default-parameter baseline (disable the guard with `--allow-worse`).

## Model parameter overrides (JSON examples)

- Inline: `--model-params '{"k_factor": 20, "home_advantage": 60}'`
- File: create `params.json`:

```json
{
  "elo": {"k_factor": 20, "home_advantage": 60},
  "bradley-terry": {"max_iter": 500}
}
```

Then run: `python -m src.cli.pipeline rank --sport nba --season 2025-26 --model-params-file params.json`

## Input format notes

- Recognized columns: `Date` / `Game Date`; `Visitor/Neutral` / `Away`; `Home/Neutral` / `Home`; `PTS` or `PTS_away` / `PTS_home`; `OT` / `Overtime`; `Box Score`.
- If a CSV contains an unlabeled start-time column, the parser realigns columns automatically.
- Future games are accepted; rankings and projections require at least one completed game per team to be meaningful.

## Outputs at a glance

- SQLite DB: `data/db/<sport>/<season>.db` (`games`, `model_metrics`, optional `backtest_metrics`).
- Rankings: `data/processed/<sport>/<season>/rankings.csv` (prefixed when multiple models run).
- Schedule: `data/processed/<sport>/<season>/schedule_with_projections.xlsx` (default) or `.csv` (per model).
- Rankings report: `data/processed/<sport>/<season>/report.xlsx` (per model when multiple).
- Backtests: `outputs/backtests/<model>/` CSVs + Excel per run.
- Tuning: `outputs/tuning/<model>/` grid CSV, best params JSON, and per-candidate backtests.

## Configuration

- `src/config.py` holds global defaults such as `DEFAULT_WIN_PROB_K`, calibration sample sizes, and fallback spread/total uncertainty.

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

`make test` is also available as a shortcut.

## Troubleshooting

- Missing or misnamed columns: ensure the CSV has date + home/away + scores; common aliases are auto-mapped.
- Unknown teams during `matchup`: rerun `rank` after new ingests so ratings exist for those teams.
- No completed games: rankings/projections require finished games; ingest more results or allow future games only for schedule exports.
- Overwriting outputs: pass `--overwrite` for rankings; for schedule/report/backtest/tune, specify a new `--output`/`--output-dir` or allow the CLI to append numeric suffixes when available.

## License

This project is provided as-is for internal analytics workflows.
