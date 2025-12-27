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
- Accepts HTML/CSV files or `--input-text` for pasted CSV content. Override the DB with `--db` (default `data/db/<sport>/<season>.db`).

### Run rankings
- `python -m src.cli.pipeline rank --sport nba --season 2024-25 --model bradley-terry --output data/processed/nba/2024-25/rankings.csv --overwrite`
- `--sport` and `--season` are required and must match the database you imported into.
- Without `--output`, rankings write to `data/processed/<sport>/<season>/rankings.csv`. Uses the same DB as import unless `--db` is set.

### Predict a matchup
- `python -m src.cli.pipeline matchup --sport nba --season 2024-25 --matchup "Lakers vs Celtics"`
- `python -m src.cli.pipeline matchup --sport nfl --season 2023 --home Eagles --away Cowboys`
- Uses the per-sport/per-season database and Bradley-Terry rankings to estimate a winner, spread, and total points.

### Rankings output columns
- team
- rating (Bradley-Terry strength)
- points (log-scaled rating mapped to point-spread units)
- games
