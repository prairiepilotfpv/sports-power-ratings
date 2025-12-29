# Sports Power Ratings

Lightweight pipeline for turning Sports-Reference schedules/results into SQLite data and Bradley-Terry power ratings.

## Setup
- python -m venv .pyenv
- .\.pyenv\Scripts\Activate.ps1
- pip install -r requirements.txt
- For OCR/image ingestion, install Tesseract and set `OPENAI_API_KEY` (optional: `OPENAI_MODEL`).

## Usage

### Convert Sports-Reference schedules to CSV
- HTML/CSV/Excel: `python -m src.pipelines.ingest_game_results data/raw/nba_schedule.html --mode auto --overwrite`
- Screenshots: `python -m src.pipelines.ingest_game_results data/raw/nba_schedule.png --mode image`
- Defaults: resolves missing paths under `data/raw/`, writes to `data/processed/<input_stem>.csv`, and requires `--overwrite` if the output exists.

### Import games into SQLite
- `python -m src.cli.pipeline import --sport nba --season 2024-25 --input data/processed/nba_schedule.csv`
- `--sport` and `--season` are required and determine which SQLite database is used. The pipeline does not infer these from the CSV.
- Accepts HTML/CSV files or `--input-text` for pasted CSV content. If `--input` is a bare filename, it is resolved under `data/raw/` first. Override the DB with `--db` (default `data/db/<sport>/<season>.db`).
- Rows without scores are kept so the schedule can include future games.
- Example for another league: `python -m src.cli.pipeline import --sport nfl --season 2024 --input data/raw/nfl_2024.csv`

### Update a database with a new CSV drop
- Drop the latest Sports-Reference CSV in `data/raw/` (or point directly to the file path).
- Re-run the import for the same `--sport`/`--season` to upsert new or updated games:
  - `python -m src.cli.pipeline import --sport nba --season 2024-25 --input data/raw/nba_2024_25.csv`
- The import uses `INSERT OR REPLACE` keyed on `(game_id, sport, season)`, so re-importing the full CSV refreshes existing rows and adds new games.

### Run rankings
- `python -m src.cli.pipeline rank --sport nba --season 2024-25 --model bradley-terry --output data/processed/nba/2024-25/rankings.csv --overwrite`
- `--sport` and `--season` are required and must match the database you imported into.
- Without `--output`, rankings write to `data/processed/<sport>/<season>/rankings.csv`. Uses the same DB as import unless `--db` is set.

### Export schedule with projections
- `python -m src.cli.pipeline schedule --sport nba --season 2024-25 --model bradley-terry --output data/processed/nba/2024-25/schedule_with_projections.csv`
- Outputs played games and upcoming games in one calendar, with `projected_winner`, `projected_spread`, `projected_total`, and current `home_rating`/`away_rating` from the selected model. Use `--upcoming-only` to show just future games.
- Uses the same per-sport/per-season DB unless `--db` is set; defaults to writing `data/processed/<sport>/<season>/schedule_with_projections.csv` when `--output` is omitted.

### Predict a matchup
- `python -m src.cli.pipeline matchup --sport nba --season 2024-25 --matchup "Lakers vs Celtics"`
- `python -m src.cli.pipeline matchup --sport nfl --season 2023 --home Eagles --away Cowboys`
- Uses the per-sport/per-season database and Bradley-Terry rankings to estimate a winner, spread, and total points.

### Rankings output columns
- team
- rating (Bradley-Terry strength)
- points (log-scaled rating mapped to point-spread units)
- games

### Generate Excel worksheet output
- `python -m src.cli.pipeline report --sport nba --season 2024-25`
- Writes to `data/processed/<sport>/<season>/report.xlsx` by default. Use `--output` to pick a different file or directory.
- To include multiple models in separate sheets, pass a comma-separated list:
  - `python -m src.cli.pipeline report --sport nba --season 2024-25 --models bradley-terry,elo`
