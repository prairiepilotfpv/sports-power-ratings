# Sports Power Ratings

Lightweight pipeline for turning Sports-Reference schedules/results into SQLite data and Bradley-Terry power ratings (projections, matchup sims, Excel reports).

## Setup
- python -m venv .pyenv
- .\.pyenv\Scripts\Activate.ps1
- pip install -r requirements.txt
- For OCR/image ingestion, install Tesseract and set `OPENAI_API_KEY` (optional: `OPENAI_MODEL`).
- Tests use pytest; install it alongside runtime deps: `pip install pytest`

## Usage

### Convert Sports-Reference schedules to CSV
- HTML/CSV/Excel: `python -m src.pipelines.ingest_game_results data/raw/nba_schedule.html --mode auto --overwrite`
- Screenshots: `python -m src.pipelines.ingest_game_results data/raw/nba_schedule.png --mode image` (uses Tesseract + OpenAI parsing)
- Defaults: resolves missing paths under `data/raw/`, writes to `data/processed/<input_stem>.csv`, and requires `--overwrite` if the output exists.

### Import games into SQLite
- `python -m src.cli.pipeline import --sport nba --season 2025-26 --input data/processed/nba_schedule.csv`
- `--sport` and `--season` are required and determine which SQLite database is used. The pipeline does not infer these from the CSV.
- Accepts HTML/CSV files or `--input-text` for pasted CSV content. If `--input` is a bare filename, it is resolved under `data/raw/` first. Override the DB with `--db` (default `data/db/<sport>/<season>.db`).
- Rows without scores are kept so the schedule can include future games.
- Example for another league: `python -m src.cli.pipeline import --sport nfl --season 2024 --input data/raw/nfl_2024.csv`

### Update a database with a new CSV drop
- Drop the latest Sports-Reference CSV in `data/raw/` (or point directly to the file path).
- Re-run the import for the same `--sport`/`--season` to upsert new or updated games:
  - `python -m src.cli.pipeline import --sport nba --season 2025-26 --input data/raw/nba_2025_26.csv`
- The import uses `INSERT OR REPLACE` keyed on `(game_id, sport, season)`, so re-importing the full CSV refreshes existing rows and adds new games.

### Run rankings
- `python -m src.cli.pipeline rank --sport nba --season 2025-26 --model bradley-terry --output data/processed/nba/2025-26/rankings.csv --overwrite`
- `--sport` and `--season` are required and must match the database you imported into.
- Without `--output`, rankings write to `data/processed/<sport>/<season>/rankings.csv`. Uses the same DB as import unless `--db` is set. If the target file exists and `--overwrite` is omitted, a suffixed path (e.g., `rankings-1.csv`) is used instead.
- The rankings step also saves `home_advantage`, `win_prob_k`, and `base_total` into the database for the selected model; those metrics are reused by projections/matchups to calibrate spreads, totals, and win probabilities.

### Export schedule with projections
- `python -m src.cli.pipeline schedule --sport nba --season 2025-26 --model bradley-terry --output data/processed/nba/2025-26/schedule_with_projections.csv`
- Outputs played games and upcoming games in one calendar. Uses the same per-sport/per-season DB unless `--db` is set; defaults to writing `data/processed/<sport>/<season>/schedule_with_projections.csv` when `--output` is omitted.
- Data columns:
  - Schedule core: `date`, `home_team`, `away_team`, `home_score`, `away_score`, `neutral`, `overtime`, `game_id`
  - Status: `status` is `final` for completed games and `scheduled` otherwise; rerunning after new scores moves rows to `final` and carries the latest points.
  - Ratings/projections: `home_rating`/`away_rating` (model-derived point-scale ratings, expected neutral-court margin vs average), `projected_winner`, `projected_spread` (betting-format home spread, negative means home favored), `projected_win_prob` (home win probability), `projected_home_score`, `projected_away_score`, `projected_total` (league base total)
  - Model calibration: `home_advantage`, `win_prob_k`, `base_total` pulled from the rankings step
  - Results for completed games: `result_margin`, `result_total`
- Use `--upcoming-only` to show just future games. Notes/model columns are intentionally omitted from the output.

### Predict a matchup
- `python -m src.cli.pipeline matchup --sport nba --season 2025-26 --matchup "Lakers vs Celtics"`
- `python -m src.cli.pipeline matchup --sport nfl --season 2023 --home Eagles --away Cowboys`
- Uses the per-sport/per-season database and Bradley-Terry rankings to estimate a winner, spread (betting-format home spread), total points, and win probability (when `win_prob_k` is available from the latest rankings run).

### Rankings output columns
- team
- rating (Bradley-Terry strength)
- points (log-scaled rating mapped to point-spread units; expected neutral-court margin vs an average team)
- games

### Generate Excel worksheet output
- `python -m src.cli.pipeline report --sport nba --season 2025-26`
- Writes to `data/processed/<sport>/<season>/report.xlsx` by default. Use `--output` to pick a different file or directory.
- Pass `--models` with a comma-separated list to create one sheet per model (currently only `bradley-terry` is registered).
- The daily report spreadsheet (see `src/pipelines/report.py`) includes a "Spreads" section where `home_rating`/`away_rating` are the model-derived power ratings for each team.

## Automated tests
- Activate the venv and install deps (including pytest), then run: `python -m pytest`
- Quick checks:
  - Import path resolution: `python -m pytest tests/cli/test_pipeline_import.py -q`
  - Rankings pipeline: `python -m pytest tests/pipelines/test_run_rankings.py -q`
  - Schedule projections: `python -m pytest tests/pipelines/test_schedule.py -q`
- Tests use local fixtures only; no network calls are required.
