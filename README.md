# Sports Power Ratings

A lightweight, local-first pipeline for turning Sports-Reference schedules/results into:

- SQLite game databases (one DB per sport + season)
- Power ratings (Bradley-Terry, Elo, GSSD, TOOR) and calibration metrics
- Matchup projections (spread, total, win probability)
- Schedule exports with projections
- Excel reports for daily review

The project is intentionally minimal: everything runs locally with CSV/HTML inputs and a
single command-line interface.

## What this project does

1. **Ingest** Sports-Reference schedules/results (HTML, CSV, or pasted CSV text)
2. **Normalize + validate** the dataset into a consistent schema
3. **Store** games in a SQLite database keyed by `(game_id, sport, season)`
4. **Run rankings** using the Bradley-Terry model
5. **Generate projections** for matchups and schedules
6. **Export** results to CSV or Excel

## Repository layout

```
src/
  cli/                 Command-line entry points
  data/                SQLite schema, read/write helpers, validation
  ingest/              Sports-Reference parsing + normalization helpers
  models/              Power rating implementations + model interfaces
  pipelines/           Ranking, projections, reports, matchup logic
  utils/               Small shared utilities

data/
  raw/                 Drop raw HTML/CSV inputs here
  processed/           Output rankings/schedules/reports
  db/<sport>/<season>.db  SQLite database per sport+season
```

## Models and capabilities

- **Ranking models**: `bradley-terry` (default), `elo`, `gssd` (requires `ssat`), `toor`
- **Backtest models**: `bradley_terry_hfa`, `bradley_terry_calibrated_hfa`, `toor`, `elo`, `gssd`
- All backtest models output win probabilities and margins and emit calibration extras (`win_prob_k`, home-advantage/scale/error terms) for persistence.

## Setup

### 1) Create a virtual environment

```bash
python -m venv .pyenv
```

Activate it:

- **Windows (PowerShell)**: `./.pyenv/Scripts/Activate.ps1`
- **macOS/Linux**: `source .pyenv/bin/activate`

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

Optional dev dependencies (tests):

```bash
pip install -r requirements-dev.txt
```

## Documentation

The codebase uses module/class/function docstrings as the source of truth for internal
documentation. You can generate a quick HTML reference locally with `pydoc` so the docs
stay in sync with the code comments:

```bash
# Example: generate HTML docs for the ingest schema module
python -m pydoc -w src.ingest.schema
```

For an interactive view, you can also run `python -m pydoc` and browse the rendered pages
locally.

### 3) Optional OCR dependencies

If you plan to ingest screenshots/images, install Tesseract and set:

- `OPENAI_API_KEY` (optional, improves parsing)
- `OPENAI_MODEL` (optional override)

## Quickstart

Assume you have a Sports-Reference CSV file at `data/raw/nba_2025_26.csv`.

```bash
# 1) Import results into SQLite
python -m src.cli.pipeline import --sport nba --season 2025-26 --input data/raw/nba_2025_26.csv

# 2) Run rankings
python -m src.cli.pipeline rank --sport nba --season 2025-26 --model bradley-terry
#    (other ranking models: elo, gssd, toor; gssd requires the `ssat` dependency)

# 3) Export schedule with projections
python -m src.cli.pipeline schedule --sport nba --season 2025-26 --model bradley-terry

# 4) Predict a single matchup
python -m src.cli.pipeline matchup --sport nba --season 2025-26 --matchup "Lakers vs Celtics"

# 5) Generate an Excel report
python -m src.cli.pipeline report --sport nba --season 2025-26

# 6) Run a backtest (Excel + CSV outputs)
python -m src.cli.pipeline backtest --csv nba_results.csv --model bradley_terry_hfa --start 2024-11-01 --end 2024-12-01
#
# Supported backtest models:
#   bradley_terry_hfa, bradley_terry_calibrated_hfa, toor, elo, gssd
```

## CLI reference

All commands are available via `python -m src.cli.pipeline`.

### `import` (ingest into SQLite)

```bash
python -m src.cli.pipeline import --sport nba --season 2025-26 --input data/raw/nba_schedule.html
```

Options:

- `--sport` / `--season`: required identifiers used to select the DB
- `--input`: HTML or CSV file path
- `--input-text`: pasted CSV text (for quick copy/paste ingestion)
- `--db`: custom SQLite path (defaults to `data/db/<sport>/<season>.db`)

Notes:

- If `--input` is a bare filename, it is first resolved under `data/raw/`.
- Ingestion accepts **future games with no scores** so the schedule can include upcoming matchups.

### `rank` (run model + save rankings)

```bash
python -m src.cli.pipeline rank --sport nba --season 2025-26 --model bradley-terry
```

Options:

- `--output`: output CSV path (defaults to `data/processed/<sport>/<season>/rankings.csv`)
- `--overwrite`: overwrite the output if it already exists
- `--model-params`: JSON string of model parameters to override defaults
- `--model-params-file`: JSON file containing model parameters or per-model overrides
- `--db`: custom SQLite path

This step also stores calibration metrics (`home_advantage`, `win_prob_k`, `base_total`) in the
SQLite database for downstream projection steps.

### `schedule` (export schedule with projections)

```bash
python -m src.cli.pipeline schedule --sport nba --season 2025-26 --model bradley-terry
```

Options:

- `--output`: output CSV path (defaults to `data/processed/<sport>/<season>/schedule_with_projections.csv`)
- `--upcoming-only`: include only games without scores
- `--model-params`: JSON string of model parameters to override defaults
- `--model-params-file`: JSON file containing model parameters or per-model overrides

Output includes:

- Base schedule: `date`, `home_team`, `away_team`, `home_score`, `away_score`, `neutral`, `overtime`, `game_id`
- Status: `status` (`final` or `scheduled`)
- Ratings/projections: `home_rating`, `away_rating`, `projected_winner`, `projected_spread`,
  `projected_win_prob`, `projected_home_score`, `projected_away_score`, `projected_total`
- Calibration: `home_advantage` (team-specific residual or fallback)
- Results for completed games: `result_margin`, `result_total`

### `matchup` (predict a single matchup)

```bash
python -m src.cli.pipeline matchup --sport nba --season 2025-26 --matchup "Lakers vs Celtics"
# or
python -m src.cli.pipeline matchup --sport nba --season 2025-26 --home Lakers --away Celtics
```

Output includes a projected winner, spread, total points, and win probability when calibrated.

Options:

- `--model-params`: JSON string of model parameters to override defaults
- `--model-params-file`: JSON file containing model parameters or per-model overrides

### `report` (Excel output)

```bash
python -m src.cli.pipeline report --sport nba --season 2025-26
```

Options:

- `--models`: comma-separated list of models (default: `bradley-terry`)
- `--output`: output Excel path (defaults to `data/processed/<sport>/<season>/report.xlsx`)
- `--model-params`: JSON string of model parameters to override defaults
- `--model-params-file`: JSON file containing model parameters or per-model overrides

### `backtest` (model accuracy on historical games)

```bash
python -m src.cli.pipeline backtest --csv nba_results.csv --model bradley_terry_hfa --start 2024-11-01 --end 2024-12-01
```

Options:

- `--model`: backtest model to run (default: `bradley_terry_hfa`)
- `--model-params`: JSON string of model parameters to override defaults
- `--model-params-file`: JSON file containing model parameters or per-model overrides
- `--start` / `--end`: evaluation window (YYYY-MM-DD)
- `--window`: training window type (`expanding` or `rolling`)
- `--rolling-days` / `--rolling-games`: rolling window size for `rolling` runs
- `--output-dir`: output directory override (default: `outputs/backtests/<model>`)
- `--sport` / `--season`: dataset selection
- `--db`: custom SQLite path
- `--csv`: required path to a CSV of completed games (relative paths are resolved from the repo root)

Notes:

- Re-running the same model/window overwrites CSV/Excel files in the same `--output-dir`. Point `--output-dir` to a new folder if you want to keep multiple runs side-by-side.

Input CSV for backtests:

- Place the file anywhere (e.g., repo root) and pass it via `--csv my_results.csv`
- Required columns: `date`, `home_team`, `away_team`, `home_score`, `away_score` (aliases like `Date`, `Visitor/Neutral`, `Home/Neutral`, `PTS_away`, `PTS_home` are auto-detected)
- Optional columns: `neutral` (or `neutral_site`), `overtime` (`ot`), `game_id` (`box score`) (auto-generated if missing)
- Dates must parse as YYYY-MM-DD; scores must be numeric and non-negative
- The backtest window (`--start` / `--end`) must overlap the CSV data or no evaluations will run

### `tune` (hyperparameter tuning via backtests)

Run a grid search of model-specific hyperparameters and select the best
configuration based on log loss, Brier score, or margin MAE.

```bash
python -m src.cli.pipeline tune \
  --model elo \
  --csv nba_results.csv \
  --start 2024-11-01 \
  --end 2024-12-01 \
  --metric log_loss
```

Options:

- `--model`: backtest model to tune (e.g., `elo`, `gssd`, `toor`, `bradley_terry_hfa`)
- `--start` / `--end`: evaluation window (YYYY-MM-DD)
- `--window`: training window type (`expanding` or `rolling`)
- `--rolling-days` / `--rolling-games`: rolling window size for `rolling` runs
- `--metric`: optimization target (`log_loss`, `brier_score`, `mae_margin`)
- `--output-dir`: output directory override (default: `outputs/tuning/<model>`)
- `--csv`: required path to a CSV of completed games (relative paths are resolved from the repo root)
- `--grid-file`: JSON file defining custom parameter grids
- `--apply-best`: run a final backtest with best params and persist metrics to the DB
- `--allow-worse`: allow tuning results worse than the default parameters
- `--sport` / `--season`: dataset selection for persisting best metrics
- `--db`: custom SQLite path for persisting best metrics

Output files (per run) include:

- `tuning_results_<run_id>.csv`: every candidate’s metric results and output directory
- `best_params_<run_id>.json`: the best-performing parameter set and score
- Individual backtest artifacts for each candidate under the run directory

Notes:

- Tuning uses the same CSV schema and parsing rules as the backtest command.
- When `--apply-best` is set, the tuning run compares the best candidate against
  the default-parameter baseline and only persists calibration metrics if the
  candidate improves on the chosen metric. Use `--allow-worse` to skip this guard.

#### Custom grids (JSON)

Provide a JSON file to override the default tuning grid. Two supported formats:

**Per-model grid**

```json
{
  "elo": {
    "k_factor": [10, 20, 40],
    "home_advantage": [0, 50, 100],
    "initial_rating": [1500],
    "min_rating": [1]
  }
}
```

**Single-grid override** (applies to the specified `--model`)

```json
{
  "k_factor": [10, 20, 40],
  "home_advantage": [0, 50, 100],
  "initial_rating": [1500],
  "min_rating": [1]
}
```

## Input formats

The ingest layer is designed for Sports-Reference exports. It supports:

- **HTML tables** from the schedule/results pages
- **CSV exports** from Sports-Reference
- **Pasted CSV text** via `--input-text`

Important columns recognized:

- `Date` / `Game Date`
- `Visitor/Neutral` / `Away`
- `Home/Neutral` / `Home`
- `PTS` or `PTS_away` / `PTS_home`
- `OT` / `Overtime`
- `Box Score`

If the CSV contains an unlabeled start-time column, the parser will realign columns automatically.

## Output files

- **SQLite DB**: `data/db/<sport>/<season>.db`
  - `games` table
  - `model_metrics` table
- **Rankings CSV**: `data/processed/<sport>/<season>/rankings.csv`
- **Schedule CSV**: `data/processed/<sport>/<season>/schedule_with_projections.csv`
- **Excel report**: `data/processed/<sport>/<season>/report.xlsx`
- **Backtest outputs**: `outputs/backtests/<model>/` (CSV + Excel workbook per run)
- **Tuning outputs**: `outputs/tuning/<model>/` (grid search CSV, best params JSON, and per-candidate backtests)

## Configuration

Edit `src/config.py` to adjust defaults such as:

- `DEFAULT_WIN_PROB_K`: logistic scale for converting spreads to win probabilities

## Tests

```bash
python -m pytest
```

Focused checks:

```bash
python -m pytest tests/cli/test_pipeline_import.py -q
python -m pytest tests/pipelines/test_run_rankings.py -q
python -m pytest tests/pipelines/test_schedule.py -q
```

## Troubleshooting

- **Missing columns**: the ingest parser expects date + home/away columns. Check CSV headers.
- **Unknown teams in matchup**: run `rank` after new imports to update ratings.
- **No completed games**: rankings/projections require at least one finished game.

## License

This project is provided as-is for internal analytics workflows.
