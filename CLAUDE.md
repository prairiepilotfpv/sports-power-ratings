# CLAUDE.md

Concise reference for AI coding agents working in this repository.

## Project Overview

**Sports Power Ratings** is a local-first Python CLI for sports analytics:
- Ingest Sports-Reference schedules/results into per-sport/per-season SQLite databases
- Fit power-rating models (Bradley-Terry, Elo, GSSD, TOOR, Poisson)
- Generate projections, rankings, and schedule workbooks with spreads/totals/win probabilities
- Run backtests and hyperparameter tuning
- Track bets via OCR/CSV ingestion, settlement, and reporting

All commands run via `python -m src.cli.pipeline <command> ...`.

## Quick Reference Commands

```bash
# Setup
python -m venv .pyenv
source .pyenv/bin/activate  # Linux/macOS
pip install -r requirements.txt

# Core workflow
python -m src.cli.pipeline import --sport nba --season 2025-26 --input data/raw/nba.csv
python -m src.cli.pipeline rank --sport nba --season 2025-26
python -m src.cli.pipeline schedule --sport nba --season 2025-26
python -m src.cli.pipeline matchup --sport nba --season 2025-26 --matchup "Lakers vs Celtics"

# Testing
pytest -q                    # All tests
pytest -q -m "not slow"      # Fast tests only
coverage run -m pytest -q && coverage report --fail-under=70
```

## Repository Structure

```
src/
  cli/                 CLI entry points (pipeline.py is primary)
  data/                SQLite helpers, paths, validation, migrations
  ingest/              Sports-Reference parsing + normalization
  models/              Power rating implementations + registry
  pipelines/           Orchestration (ingest, rank, schedule, backtest, tune, betting)
  ensemble/            Ensemble aggregation for ML/SPREAD/TOTAL markets
  calibration/         Probability calibration (Platt, Isotonic)
  backtest/            Backtest runner + exports
  eval/                Bet evaluation and guardrails
  utils/               Odds conversion, game IDs, team mapping
  ocr/                 Tesseract OCR wrapper

data/
  raw/                 Drop input HTML/CSV files here
  processed/           Rankings, schedules, reports output
  db/<sport>/<season>.db  SQLite databases

outputs/
  backtests/           Backtest artifacts
  tuning/              Tuning results and best params
  ensembles/           Ensemble weight configs
  calibrators/         Fitted calibrator objects
  reports/             Betting reports

tests/
  fixtures/            Test data (CSV, JSON samples)
  models/              Model-specific tests
  pipelines/           Pipeline integration tests
  contracts/           Contract/interface tests
```

## Key Architectural Patterns

### Model Registry
Models are registered in `src/models/registry.py`:
- **Ranking models**: `bradley-terry`, `elo`, `gssd`, `poisson`, `toor`
- **Backtest models**: `bradley_terry_hfa`, `bradley_terry_calibrated_hfa`, `elo`, `gssd`, `toor`

Use `list_models()` and `list_backtest_models()` to enumerate available models.

### Path Conventions
- Default DBs: `data/db/<sport>/<season>.db`
- Processed output: `data/processed/<sport>/<season>/`
- Bare filenames resolve under `data/raw/` (see `resolve_input_path`)
- Betting artifacts: `outputs/` subdirectories

### Game ID Generation
`ensure_game_id()` in `src/contracts.py` builds deterministic IDs from `{date}_{home_team}_{away_team}` when missing. Tests rely on stable game IDs.

### Database Schema
Core tables in SQLite (see `src/data/repository.py`):
- `games` - schedule/results data
- `teams` / `team_aliases` - canonical team names and mappings
- `model_metrics` - calibration parameters per model
- `model_tuned_params` - hyperparameter tuning results
- `market_lines` - imported market odds
- `bets` - logged wagers
- `clv_snapshots` - closing line values

Migrations are additive and idempotent (see `src/data/migrations.py`).

## Development Workflow

### Running the CLI
Always use `python -m src.cli.pipeline <command>` for integration behavior:

```bash
# Ingest
python -m src.cli.pipeline import --sport nba --season 2025-26 --input data/raw/nba.csv

# Rank (all models by default)
python -m src.cli.pipeline rank --sport nba --season 2025-26
python -m src.cli.pipeline rank --sport nba --season 2025-26 --model elo

# Schedule export (Excel workbook with BETS sheet)
python -m src.cli.pipeline schedule --sport nba --season 2025-26

# Backtest
python -m src.cli.pipeline backtest --model bradley_terry_hfa --csv nba_results.csv \
  --start 2024-11-01 --end 2024-12-01

# Tune
python -m src.cli.pipeline tune --model elo --csv nba_results.csv \
  --start 2024-11-01 --end 2024-12-01 --metric log_loss --apply-best \
  --sport nba --season 2025-26

# Betting workflow
python -m src.cli.pipeline betting market-csv --sport nba --season 2025-26 \
  --csv data/raw/nba_markets.csv --default-book dn
python -m src.cli.pipeline betting settle-bets --sport nba --season 2025-26
python -m src.cli.pipeline betting report --sport nba --season 2025-26 --type weekly
```

### Model Parameters
- Inline: `--model-params '{"k_factor": 20, "home_advantage": 60}'`
- File: `--model-params-file params.json` (supports per-model keys)
- Tuned params auto-load when present; override with `--tuned-metric`

### Testing

```bash
# Run all tests
pytest -q

# Fast tests only (skip @pytest.mark.slow)
pytest -q -m "not slow"

# Specific test file/function
pytest -q tests/test_interface_contract.py
pytest -q -k "test_elo"

# Coverage
coverage run -m pytest -q
coverage report --fail-under=70
```

Test organization:
- `tests/models/` - model-specific tests
- `tests/pipelines/` - pipeline integration tests
- `tests/contracts/` - interface contract tests
- `tests/fixtures/` - test data files

### Code Quality
Pre-commit hooks run `black` and `ruff`:
```bash
pip install -r requirements-dev.txt
pre-commit install
pre-commit run --all-files
```

## Important Conventions

### DO
- Use `python -m src.cli.pipeline` for integration testing
- Prefer `--model-params-file` for complex parameter overrides
- Run `pytest -q` before committing changes
- Add tests for new model or pipeline behavior
- Use `register_model()` in tests to inject test doubles
- Keep migrations idempotent (check for existence before altering)

### DON'T
- Refactor unrelated modules
- Change existing model fitting/training logic unless explicitly required
- Assume model names are interchangeable between ranking and backtest contexts
- Change DB schema without updating validation functions
- Commit large OCR screenshots or generated reports (keep in `outputs/`, `tmp-*`)
- Use `src/cli/ingest.py` directly (deprecated; use `src/cli/pipeline import`)

### Output Safety
When `--output` exists and `--overwrite` is not provided, CLI appends numeric suffixes (`-1`, `-2`, etc.) via `_next_available_path`.

## Calibration & Market-Specific Provenance Tags

### Overview
The `schedule` command applies market-specific probability calibrators (ML, SPREAD, TOTAL) to predicted values. When calibration succeeds, provenance tags are appended to the `win_prob_source` column to provide auditability.

### Implementation Details
- **Location**: `src/pipelines/schedule.py::_apply_calibration_to_schedule_df()` (lines 333-573)
- **Tracking**: Maintains a `calibrated_markets` set that accumulates which markets had successful calibrations
- **Tag Appending**: After all calibrations complete, tags are appended to `win_prob_source` with '+' separator
- **Tag Format**: 
  - ML market → appends `calibrated_ml`
  - SPREAD market → appends `calibrated_spread`
  - TOTAL market → appends `calibrated_total`
  - Example: `"elo+calibrated_ml+calibrated_spread"`

### Idempotency
Tags are appended idempotently—calling the function multiple times on the same DataFrame won't create duplicates:
- Uses case-insensitive checking: if tag already exists (anywhere in string), it's skipped
- Sorts tags alphabetically for deterministic output
- Safe to call multiple times on production data

### Logging
When tags are appended, an INFO-level log is emitted:
```
[_apply_calibration_to_schedule_df] Appended calibration provenance tags to win_prob_source: calibrated_ml, calibrated_spread
```
Enable with `--log-cli-level=INFO` in pytest or check application logs.

### Testing
Comprehensive test suite in `tests/test_calibration_bets_integration.py`:
- **test_calibration_provenance_tags_ml_market**: Single market tag appending
- **test_calibration_provenance_tags_spread_market**: Different market tag
- **test_calibration_provenance_tags_total_market**: Third market type
- **test_calibration_provenance_tags_multiple_markets**: Multiple markets simultaneously
- **test_calibration_provenance_tags_idempotent**: Idempotency verification (no duplicates)
- **test_calibration_provenance_tags_no_column**: Graceful handling when column is missing

Run with: `pytest -q tests/test_calibration_bets_integration.py -k "provenance_tags" -v`

### Design Rationale
Provenance tags enable:
1. **Auditability**: Downstream consumers can see which calibrators were applied
2. **Filtering**: Group/analyze predictions by calibration status (e.g., "calibrated_ml" vs "uncalibrated")
3. **Debugging**: Identify missing calibrations or mismatches
4. **Reproducibility**: Track model configuration changes across seasons

### Deprecated: BETS Sheet Secondary Calibration

**Note (2025-01-27)**: The following functions in `src/pipelines/schedule.py` are deprecated and no longer called:
- `_apply_spread_total_calibrators()` (lines ~1628-1718)
- `_apply_market_calibrator()` (lines ~1721-1757)
- `_load_market_calibrators()` (lines ~1760-1800)

**Why deprecated**: These functions had a fundamental interface mismatch:
1. `_load_market_calibrators()` loaded `MarginalDistributionCalibrator` objects (for SPREAD/TOTAL)
2. `MarginalDistributionCalibrator.transform()` expects a DataFrame with `pred_mean` and `pred_sd` columns
3. But `_apply_market_calibrator()` called `.transform(pd.Series([raw_prob]))` expecting a probability calibrator
4. The silent `except Exception` block caused all calibration to return `pd.NA`

**Current architecture**: Distribution parameters (margin_mean, margin_sd, total, total_sd) are calibrated upstream by `_apply_calibration_to_schedule_df()` before being passed to `_build_bets_dataframe()`. The probabilities computed by `_calculate_model_prob()` already use calibrated distribution parameters.

### Testing Calibration: Proper Patching Pattern

When mocking `load_latest_calibrator` in tests, patch it **where it's used** (in schedule.py), not where it's defined (in calibration/io.py):

```python
# CORRECT: Patch where the function is USED
import src.pipelines.schedule as schedule_module
orig_load = schedule_module.load_latest_calibrator
schedule_module.load_latest_calibrator = mock_load
try:
    result = _apply_calibration_to_schedule_df(...)
finally:
    schedule_module.load_latest_calibrator = orig_load

# INCORRECT: This won't work because schedule.py imports directly
import src.calibration.io as cal_io
cal_io.load_latest_calibrator = mock_load  # schedule.py still uses its own reference
```

See `tests/test_calibration_bets_integration.py` for working examples.

## CSV Input Requirements

### Game data (ingest/backtest)
Required columns: `date`, `home_team`, `away_team`, `home_score`, `away_score`
Optional: `neutral`, `overtime`, `game_id`
- Dates must be parseable (YYYY-MM-DD preferred)
- Scores must be numeric and non-negative
- Column aliases are auto-detected (e.g., `Visitor/Neutral` → `away_team`)

### Market lines
Required: `market_type`, `selection`, `line`, `odds`, `team_home{_raw}`, `team_away{_raw}`, `game_date`
Optional: `game_id`, `book`, `captured_at`

## Configuration

### Global defaults (`src/config.py`)
- `DEFAULT_WIN_PROB_K = 6.566641...` - logistic scale for spread→probability
- `CALIBRATION_RESIDUAL_GAMES = 300` - rolling window for residual calibration
- `DEFAULT_MARGIN_SD_FALLBACK = 12.0` - fallback spread uncertainty
- `DEFAULT_TOTAL_SD_FALLBACK = 20.0` - fallback total uncertainty

### Guardrails
- `MARGIN_SD_GUARDRAIL_MIN/MAX` - bounds for spread uncertainty
- `TOTAL_SD_GUARDRAIL_MIN/MAX` - bounds for total uncertainty
- `PROJECTED_SCORE_MIN/MAX` - bounds for score projections

## Dependencies

### Required
- Python 3.11+
- pandas, openpyxl, beautifulsoup4, pydantic
- ssat (required for GSSD model)

### Optional
- Tesseract OCR (for screenshot ingestion)
- scikit-learn (for calibration)
- openai (for OCR assistance)

## File References

| Purpose | Location |
|---------|----------|
| CLI entry point | `src/cli/pipeline.py` |
| Model registry | `src/models/registry.py` |
| DB schema | `src/data/repository.py` |
| Migrations | `src/data/migrations.py` |
| Validation helpers | `src/contracts.py` |
| Path utilities | `src/data/paths.py` |
| Global config | `src/config.py` |
| CLI documentation | `docs/CLI.md` |
| Daily workflow | `docs/daily-workflow.md` |

## Common Tasks

### Add a new model
1. Implement in `src/models/<name>.py` inheriting from `BaseModel` or `PowerRatingModel`
2. Register in `src/models/registry.py` (add to `_MODEL_SPECS` and/or `_BACKTEST_REGISTRY`)
3. Add tests in `tests/models/`
4. Update CLI docs if needed

### Add a new pipeline command
1. Implement in `src/pipelines/<name>.py`
2. Wire into `src/cli/pipeline.py`
3. Add integration tests in `tests/pipelines/`
4. Document in `docs/CLI.md`

### Modify DB schema
1. Add migration in `src/data/migrations.py` (append to `MIGRATIONS` dict)
2. Keep idempotent (check for column/table existence)
3. Update `src/data/repository.py` if adding new queries
4. Add migration tests in `tests/`

## Troubleshooting

- **Missing columns**: Ensure CSV has date + home/away + scores; aliases auto-map
- **Unknown teams in matchup**: Rerun `rank` after new ingests
- **No completed games**: Rankings require finished games; ingest more results
- **Model requires dependency**: GSSD needs `ssat`; calibration needs `scikit-learn`
- **OCR not working**: Install Tesseract; check `src/ocr/ocr.py` for path detection
