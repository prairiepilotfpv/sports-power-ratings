# CLAUDE.md

Concise reference for AI assistants working in this repository.

## Project Overview

Sports Power Ratings is a local-first CLI toolchain that:
1. Ingests Sports-Reference schedules/results (CSV/HTML)
2. Persists games into per-sport/per-season SQLite DBs
3. Fits power rating models (Bradley-Terry, Elo, GSSD, TOOR, Poisson)
4. Generates projections with win probabilities, spreads, and totals
5. Supports backtesting, hyperparameter tuning, and ensemble forecasting
6. Includes a bet tracking suite with OCR ingestion, logging, settlement, and reporting

## Repository Structure

```
src/
  cli/                 CLI entry points (pipeline.py is primary)
  data/                SQLite helpers, paths, migrations, team aliases
  ingest/              Sports-Reference parsing + normalization
  models/              Power rating implementations + registry
  pipelines/           Orchestration (ingest, rank, schedule, backtest, tune, betting)
  ensemble/            Ensemble configuration and implementations
  calibration/         Probability calibration (Platt, isotonic)
  backtest/            Backtest runner + exports
  eval/                Evaluator and validation
  ocr/                 Tesseract OCR wrapper
  parsers/             Paste/table parsers

data/
  raw/                 Drop input CSV/HTML files here
  processed/           Rankings, schedules, reports (per sport/season)
  db/<sport>/<season>.db   SQLite DB per sport/season
  config/              Configuration files

outputs/
  backtests/           Backtest results per model
  tuning/              Tuning results per sport/season/model/metric
  ensembles/           Ensemble configs and weights
  calibrators/         Calibrator artifacts
  reports/             Betting reports

tests/                 pytest test suite with fixtures
```

## Quick Commands

All commands use `python -m src.cli.pipeline <command>`.

```bash
# Ingest games into SQLite
python -m src.cli.pipeline import --sport nba --season 2025-26 --input data/raw/nba.csv

# Build power ratings (all models)
python -m src.cli.pipeline rank --sport nba --season 2025-26

# Build ratings (single model)
python -m src.cli.pipeline rank --sport nba --season 2025-26 --model elo

# Export schedule with projections (Excel workbook)
python -m src.cli.pipeline schedule --sport nba --season 2025-26

# Predict a matchup
python -m src.cli.pipeline matchup --sport nba --season 2025-26 --matchup "Lakers vs Celtics"

# Backtest a model
python -m src.cli.pipeline backtest --model bradley_terry_hfa --csv data/raw/nba.csv --start 2024-11-01 --end 2024-12-01

# Tune model hyperparameters
python -m src.cli.pipeline tune --model elo --csv data/raw/nba.csv --start 2024-11-01 --end 2024-12-01 --metric log_loss

# Check tuning status
python -m src.cli.pipeline tuning-status --sport nba --season 2025-26
```

## Testing

```bash
# Run all tests
make test
# or
python -m pytest -q

# Run only fast tests (skip slow)
pytest -q -m "not slow"

# Run specific test file
pytest tests/test_interface_contract.py

# Coverage
coverage run -m pytest -q
coverage report --fail-under=70
```

## Key Conventions

### Path Conventions
- Default DB: `data/db/<sport>/<season>.db`
- Processed outputs: `data/processed/<sport>/<season>/`
- Raw inputs: `data/raw/` (CLI resolves bare filenames here)
- Path helpers: `src/data/paths.py`

### Model Registry
- Ranking models: `src/models/registry.py` → `list_models()`
- Backtest models: `src/models/registry.py` → `list_backtest_models()`
- Names differ between contexts (e.g., `bradley-terry` vs `bradley_terry_hfa`)

Available ranking models: `bradley-terry`, `elo`, `gssd`, `poisson`, `toor`
Available backtest models: `bradley_terry_hfa`, `bradley_terry_calibrated_hfa`, `elo`, `gssd`, `poisson`, `toor`

### Game ID Generation
- `ensure_game_id()` in `src/contracts.py` builds deterministic IDs: `{date}_{home_team}_{away_team}`
- Always call this when game_id might be missing

### Model Parameters
- Inline JSON: `--model-params '{"k_factor": 20}'`
- File (single or per-model): `--model-params-file params.json`
- Tuned params auto-load when present; override with `--tuned-metric`

### Data Validation
- `src/contracts.py`: Core validation helpers and dataclasses
- `validate_game_records()`: Validates game datasets
- `validate_predictions()`: Validates model outputs
- Required CSV columns for backtest: `date`, `home_team`, `away_team`, `home_score`, `away_score`

### Configuration
- Global defaults: `src/config.py` (e.g., `DEFAULT_WIN_PROB_K = 6.566...`)
- Guardrails for prediction bounds defined there

## Database Schema (Key Tables)

- `games`: Game records with scores, teams, dates
- `teams`: Canonical team names
- `team_aliases`: Team name mappings
- `model_metrics`: Calibration metrics per model
- `model_tuned_params`: Per-model, per-metric tuned parameters
- `market_lines`: Market odds/lines (modern ingestion)
- `market_snapshots`: Historical market snapshots
- `bets`: Logged bets with settlement info
- `clv_snapshots`: Closing line value snapshots

Migrations are idempotent and run on DB init (`src/data/migrations.py`).

## Ensemble System

Ensembles combine multiple model outputs per market (ML, SPREAD, TOTAL).

Config resolution order:
1. Custom: `outputs/ensembles/<sport>/<season>/<market>/<ensemble_id>.json`
2. Season default: `outputs/ensembles/<sport>/<season>/<market>/default.json`
3. Global default: `src/ensemble/default_configs/<market>.json`
4. Fallback: equal weights

```bash
# Initialize ensemble defaults
python -m src.cli.pipeline init-ensemble-config --sport nba --season 2025-26

# Tune ensemble weights
python -m src.cli.pipeline tune-ensemble --sport nba --season 2025-26 --market ML \
  --ensemble ensemble_ml_v1 --csv data/raw/nba_history.csv --start-date 2020-01-01 --end-date 2024-12-31
```

## Betting Workflow

```bash
# Import market lines from CSV
python -m src.cli.pipeline betting market-csv --sport nba --season 2025-26 --csv data/raw/markets.csv

# Log bets from workbook
python -m src.cli.pipeline betting log-bets --workbook schedule.xlsx --db data/db/nba/2025-26.db --writeback

# Settle bets
python -m src.cli.pipeline betting settle-bets --sport nba --season 2025-26

# Generate report
python -m src.cli.pipeline betting report --sport nba --season 2025-26 --type weekly --output report.xlsx
```

Note: `market-review` and `market-commit` are retired; use `betting market-csv` instead.

## Code Style & Linting

Pre-commit hooks configured (`.pre-commit-config.yaml`):
- Black (formatting)
- Ruff (linting)

## Important Files to Know

| File | Purpose |
|------|---------|
| `src/cli/pipeline.py` | Main CLI entry point |
| `src/models/registry.py` | Model registration and lookup |
| `src/contracts.py` | Data validation and contracts |
| `src/config.py` | Global configuration constants |
| `src/data/repository.py` | SQLite persistence layer |
| `src/data/paths.py` | Path resolution helpers |
| `src/data/migrations.py` | Database schema migrations |
| `tests/conftest.py` | Shared pytest fixtures |

## Development Guidelines

1. **Read before editing**: Always read existing code before modifying
2. **Prefer additive changes**: Add new modules + small integration points
3. **Run tests**: `make test` before committing
4. **Don't refactor unrelated code**: Keep changes focused
5. **Keep backward compatibility**: Don't break existing CLI interfaces
6. **Use the registry**: Register models via `register_model()` for tests
7. **Validate inputs**: Use contract helpers for data validation
8. **Document assumptions**: Add comments when making design decisions

## Common Patterns

### Loading games from DB
```python
from data.repository import load_games
games = load_games(db_path, sport="nba", season="2025-26")
```

### Running a model
```python
from models.registry import get_model
ModelClass = get_model("elo")
model = ModelClass()
model.fit(games_df)
predictions = model.predict(upcoming_df)
```

### Path resolution
```python
from data.paths import db_path_for, processed_dir
db = db_path_for("nba", "2025-26")  # data/db/nba/2025-26.db
out = processed_dir(sport="nba", season="2025-26")  # data/processed/nba/2025-26/
```

## External Dependencies

- `pandas`: Data manipulation
- `scikit-learn`: Calibration (optional for some tests)
- `openpyxl`: Excel workbook generation
- `pytesseract`: OCR (requires Tesseract installed)
- `ssat`: Required for GSSD model
- `beautifulsoup4`: HTML parsing

## Documentation

- `docs/CLI.md`: Complete CLI reference
- `docs/daily-workflow.md`: Day-to-day operations runbook
- `docs/ensembles.md`: Ensemble configuration guide
- `docs/calibration.md`: Calibration methodology
- `docs/market-bets.md`: Betting workflow details
- `docs/market-clv.md`: Closing line value ingestion
