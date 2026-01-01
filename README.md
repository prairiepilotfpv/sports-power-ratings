# Sports Power Ratings

A lightweight, local-first pipeline for turning Sports-Reference schedules/results into:

- SQLite game databases (one DB per sport + season)
- Bradley-Terry power ratings and calibration metrics
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
  models/              Bradley-Terry implementation + model interfaces
  pipelines/           Ranking, projections, reports, matchup logic
  utils/               Small shared utilities

data/
  raw/                 Drop raw HTML/CSV inputs here
  processed/           Output rankings/schedules/reports
  db/<sport>/<season>.db  SQLite database per sport+season
```

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

# 3) Export schedule with projections
python -m src.cli.pipeline schedule --sport nba --season 2025-26 --model bradley-terry

# 4) Predict a single matchup
python -m src.cli.pipeline matchup --sport nba --season 2025-26 --matchup "Lakers vs Celtics"

# 5) Generate an Excel report
python -m src.cli.pipeline report --sport nba --season 2025-26

# 6) Run a backtest (Excel + CSV outputs)
python -m src.cli.pipeline backtest --model bradley_terry_hfa --start 2024-11-01 --end 2024-12-01
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

### `report` (Excel output)

```bash
python -m src.cli.pipeline report --sport nba --season 2025-26
```

Options:

- `--models`: comma-separated list of models (default: `bradley-terry`)
- `--output`: output Excel path (defaults to `data/processed/<sport>/<season>/report.xlsx`)

### `backtest` (model accuracy on historical games)

```bash
python -m src.cli.pipeline backtest --model bradley_terry_hfa --start 2024-11-01 --end 2024-12-01
```

Options:

- `--model`: backtest model to run (default: `bradley_terry_hfa`)
- `--start` / `--end`: evaluation window (YYYY-MM-DD)
- `--window`: training window type (`expanding` or `rolling`)
- `--rolling-days` / `--rolling-games`: rolling window size for `rolling` runs
- `--output-dir`: output directory override (default: `outputs/backtests/<model>`)
- `--sport` / `--season`: dataset selection
- `--db`: custom SQLite path

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
