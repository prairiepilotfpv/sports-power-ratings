# CLAUDE.md - AI Assistant Guide

This document provides context for AI assistants working on the sports-power-ratings codebase.

## Project Overview

A local-first Python CLI tool that transforms Sports-Reference schedules/results into SQLite databases, power ratings, projections, and betting analysis reports. The system is designed to work entirely offline once inputs are downloaded.

**Core Workflow:** Ingest → Rank → Schedule → Matchup/Report → (Optional: Backtest/Tune/Bet)

## Quick Reference

### Essential Commands

```bash
# Run all tests
python -m pytest -q

# Fast tests only (skip slow-marked tests)
python -m pytest -q -m "not slow"

# Coverage check (70% minimum)
coverage run -m pytest -q && coverage report --fail-under=70

# Format code
black .

# Lint
ruff check . --fix
```

### Main CLI Entry Point

All commands run via `python -m src.cli.pipeline <command>`:

```bash
# Import games
python -m src.cli.pipeline import --sport nba --season 2025-26 --input data/raw/games.csv

# Build power ratings
python -m src.cli.pipeline rank --sport nba --season 2025-26

# Export schedule with projections
python -m src.cli.pipeline schedule --sport nba --season 2025-26

# Single matchup prediction
python -m src.cli.pipeline matchup --sport nba --season 2025-26 --matchup "Lakers vs Celtics"

# Backtest a model
python -m src.cli.pipeline backtest --csv games.csv --model elo --start 2024-01-01 --end 2024-12-31

# Tune model hyperparameters
python -m src.cli.pipeline tune --model elo --csv games.csv --start 2024-01-01 --end 2024-12-31 --metric log_loss
```

## Repository Structure

```
sports-power-ratings/
├── src/                          # Main source code
│   ├── cli/                      # CLI entry points (pipeline.py is primary)
│   ├── ingest/                   # Sports data parsing & normalization
│   ├── models/                   # Power rating implementations
│   │   ├── bradley_terry.py      # Logistic win-probability ratings
│   │   ├── bradley_terry_hfa.py  # Bradley-Terry with home advantage
│   │   ├── elo.py                # Classic Elo rating system
│   │   ├── gssd.py               # Scoring system analysis via ssat
│   │   ├── toor.py               # OLS regression on margins
│   │   ├── poisson.py            # Attack/defense scoring model
│   │   ├── calibration.py        # Probability calibration
│   │   └── registry.py           # Model registration
│   ├── pipelines/                # Orchestration & workflows (32 modules)
│   ├── data/                     # SQLite persistence & path helpers
│   │   ├── repository.py         # DB operations & queries
│   │   ├── migrations.py         # Schema evolution
│   │   └── contracts.py          # Pydantic data models
│   ├── ensemble/                 # Multi-model combinations
│   ├── calibration/              # Probability calibration (Platt, Isotonic)
│   ├── markets/                  # Market types (ML, SPREAD, TOTAL)
│   ├── eval/                     # Model evaluation & EV calculations
│   ├── ocr/                      # OCR for sportsbook screenshots
│   ├── utils/                    # Team mapping, odds, identity helpers
│   ├── backtest/                 # Backtest runner & exports
│   └── config.py                 # Global configuration constants
├── tests/                        # Test suite (83+ files)
│   ├── fixtures/                 # Test data (CSV, JSON, images)
│   └── conftest.py               # Shared pytest fixtures
├── data/                         # Data directories
│   ├── db/<sport>/<season>.db    # SQLite databases
│   ├── raw/                      # Input files
│   └── processed/                # Output files (rankings, schedules)
├── outputs/                      # Generated artifacts
│   ├── backtests/                # Backtest results
│   ├── tuning/                   # Hyperparameter search results
│   ├── ensembles/                # Ensemble weights
│   └── calibrators/              # Fitted calibrators
└── docs/                         # Documentation
    ├── CLI.md                    # Detailed command reference
    ├── daily-workflow.md         # Operational runbook
    └── archived/                 # Historical design docs
```

## Code Conventions

### Python Style

- **Python 3.11+** required
- **Black** for formatting (line length: 88)
- **Ruff** for linting
- **Type hints** throughout using `from __future__ import annotations`
- Pre-commit hooks enforce formatting and linting

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Modules | snake_case | `bradley_terry_hfa.py` |
| Classes | PascalCase | `BradleyTerryHFA`, `GamePrediction` |
| Functions | snake_case | `normalize_games()`, `load_rankings()` |
| Constants | UPPER_CASE | `DEFAULT_WIN_PROB_K`, `MIN_CALIBRATION_SAMPLES` |
| Test files | `test_*.py` | `test_elo.py`, `test_interface_contract.py` |

### Import Order

1. Standard library
2. Third-party packages
3. Relative imports

### Key Patterns

- **Dataclasses** with `frozen=True` for immutable data
- **Pydantic models** for validated inputs/outputs (see `src/data/contracts.py`)
- **Abstract Base Classes** define contracts for models/ensembles
- **Registry pattern** for model/source discovery (`src/models/registry.py`)
- **Factory pattern** for CLI model/source instantiation

## Database Schema (SQLite)

Each sport/season has its own database at `data/db/<sport>/<season>.db`.

### Core Tables

| Table | Purpose |
|-------|---------|
| `games` | Game records (teams, scores, dates, neutral, overtime) |
| `teams` | Team registry with canonical names |
| `team_aliases` | Maps alternate team names to canonical |
| `model_metrics` | Fitted model parameters (home_advantage, win_prob_k, etc.) |
| `model_tuned_params` | Hyperparameter tuning results |
| `model_market_active_params` | Active tuned params per market (ML/SPREAD/TOTAL) |
| `market_lines` | Sportsbook betting lines |
| `market_line_import_errors` | Import failure logs |

### Betting Tables

| Table | Purpose |
|-------|---------|
| `market_snapshot_staging` | OCR-parsed staging rows |
| `market_snapshots` | Committed market snapshots |
| `bets` | Recorded wagers with CLV and settlement |
| `opportunities` | Identified betting opportunities |

## Power Rating Models

| Model | Description | Key Parameters |
|-------|-------------|----------------|
| `bradley_terry` | Logistic win-probability ratings | `max_iter`, `tol`, `temp`, `l2_lambda` |
| `bradley_terry_hfa` | Bradley-Terry with home advantage | Same as above + HFA term |
| `elo` | Incremental Elo ratings | `k_factor`, `home_advantage`, `initial_rating` |
| `gssd` | Scoring system analysis | Uses `ssat` library |
| `toor` | OLS regression on margins | Lightweight alternative |
| `poisson` | Attack/defense with simulation | `n_simulations`, `reg_strength` |

## Testing

### Test Organization

```bash
tests/
├── models/          # Model implementation tests
├── ingest/          # Parsing and normalization
├── pipelines/       # Workflow orchestration
├── contracts/       # Data contract validation
├── data/            # Database and persistence
├── cli/             # CLI interface tests
└── fixtures/        # Golden test data
```

### Test Markers

- `@pytest.mark.slow` - Long-running tests (skipped with `-m "not slow"`)
- Default: Run fast tests only in development

### Coverage Requirements

- Minimum **70%** coverage enforced
- Run: `coverage run -m pytest -q && coverage report --fail-under=70`

### Writing Tests

```python
import pytest
from src.models.elo import Elo

def test_elo_initial_rating():
    model = Elo(initial_rating=1500)
    # ... assertions

@pytest.mark.slow
def test_elo_full_season():
    # Tests that take >1s should be marked slow
    pass
```

## Key Configuration (src/config.py)

```python
DEFAULT_WIN_PROB_K = 6.566641127986305    # Logistic scale for spread → win prob
CALIBRATION_RESIDUAL_GAMES = 300          # Rolling window for calibration
MIN_CALIBRATION_SAMPLES = 25              # Minimum samples for estimation
DEFAULT_MARGIN_SD_FALLBACK = 12.0         # Conservative margin uncertainty
DEFAULT_TOTAL_SD_FALLBACK = 20.0          # Conservative total uncertainty
MARGIN_SD_GUARDRAIL_MIN = 5.0             # Lower bound for SD clamping
MARGIN_SD_GUARDRAIL_MAX = 30.0            # Upper bound for SD clamping
```

## Working with This Codebase

### Before Making Changes

1. **Read existing code** before modifying - understand the current patterns
2. **Run tests** to ensure baseline passes: `python -m pytest -q -m "not slow"`
3. **Check for related tests** in `tests/` before adding features

### Adding a New Model

1. Create model class in `src/models/` implementing `PowerRatingModel` protocol
2. Register in `src/models/registry.py`
3. Add tests in `tests/models/test_<model>.py`
4. Update calibration support if model provides margins/totals

### Adding a New CLI Command

1. Add command handler in `src/cli/pipeline.py`
2. Wire up argument parser
3. Add CLI tests in `tests/cli/`
4. Document in `docs/CLI.md`

### Common Gotchas

- **Game ordering**: Must be consistent across pipelines (ingest → staging → models)
- **Team aliases**: Always resolve team names through the alias system
- **DB initialization**: Call `init_db()` before any operations
- **Neutral games**: Handle `neutral=True` games specially (no home advantage)
- **Future games**: Can be ingested but won't have scores; handle gracefully

## Documentation Index

| File | Content |
|------|---------|
| `README.md` | Project overview, quickstart, CLI reference |
| `docs/CLI.md` | Detailed command documentation |
| `docs/daily-workflow.md` | Day-to-day operational runbook |
| `docs/calibration.md` | Calibration methodology |
| `docs/ensembles.md` | Ensemble configuration |
| `docs/bet-tracking.md` | Betting workflow guide |
| `TESTING.md` | Test setup and running |
| `TODO.md` | Current task list and backlog |

## Environment Setup

```bash
# Create virtual environment
python -m venv .pyenv

# Activate (Linux/macOS)
source .pyenv/bin/activate

# Activate (Windows PowerShell)
./.pyenv/Scripts/Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development

# Install pre-commit hooks
pre-commit install
```

## Dependencies

### Runtime (requirements.txt)
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `pydantic` - Data validation
- `scikit-learn` - ML utilities (calibration)
- `openpyxl` - Excel workbook generation
- `beautifulsoup4` - HTML parsing
- `requests` - HTTP client
- `pytesseract` / `pillow` - OCR (optional)
- `openai` - OCR assistance (optional)

### Development (requirements-dev.txt)
- `pytest` - Testing framework
- `coverage` - Code coverage
- `black` - Code formatting
- `ruff` - Linting
- `pre-commit` - Git hooks

## Tips for AI Assistants

1. **Always read files before editing** - Use the Read tool first
2. **Run tests after changes** - `python -m pytest -q -m "not slow"`
3. **Preserve existing patterns** - Match the surrounding code style
4. **Check contracts** - Data models are in `src/data/contracts.py`
5. **Use type hints** - This codebase uses comprehensive typing
6. **Don't over-engineer** - Keep changes minimal and focused
7. **Test coverage matters** - Add tests for new functionality
8. **Check TODO.md** - Be aware of known issues and planned work
