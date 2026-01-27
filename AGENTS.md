# AGENTS.md

Concise, actionable guidance for AI agents working in this repository. Keep changes focused and consistent with CLAUDE.md and .github/copilot-instructions.md.

## Project snapshot
Sports Power Ratings is a local-first Python CLI for sports analytics:
- Ingest Sports-Reference schedules/results into per-sport/per-season SQLite DBs.
- Fit power-rating models (Bradley–Terry, Elo, GSSD, TOOR, Poisson).
- Generate projections, rankings, schedule workbooks, and betting reports.
- Run backtests and tuning for model/market performance.

All commands run via `python -m src.cli.pipeline <command> ...`.

## Repo map (high level)
- `src/cli/` — CLI entry points (pipeline.py is primary)
- `src/pipelines/` — orchestration for ingest/rank/schedule/backtest/tune/betting
- `src/models/` — model implementations + registry
- `src/data/` — SQLite helpers, paths, migrations
- `src/ingest/` — Sports-Reference parsing/normalization
- `tests/` — model, pipeline, and contract tests
- `data/` — raw inputs, processed outputs, per-season DBs
- `outputs/` — backtests, tuning artifacts, reports

## Quick commands
```bash
# Setup
python -m venv .pyenv
source .pyenv/bin/activate  # Linux/macOS
pip install -r requirements.txt

# Core workflow
python -m src.cli.pipeline import --sport nba --season 2025-26 --input data/raw/nba.csv
python -m src.cli.pipeline rank --sport nba --season 2025-26
python -m src.cli.pipeline schedule --sport nba --season 2025-26

# Tests
pytest -q
pytest -q -m "not slow"
```

## Guardrails (do / don’t)
**Do:**
- Use `python -m src.cli.pipeline` for integration behavior.
- Keep changes additive and tightly scoped; avoid unrelated refactors.
- Add/adjust tests when behavior changes.
- Keep migrations idempotent (check for existence before altering).
- Use `make_game_id()` / `ensure_game_id()` for game IDs (never roll your own format).
- Keep outputs/temporary artifacts in `outputs/` or `tmp-*` (do not commit large artifacts).

**Don’t:**
- Change model fitting/training logic or math unless explicitly requested.
- Change DB schema without updating validation and repository helpers.
- Assume model names are interchangeable between ranking and backtest contexts.

## Architecture notes
- Model registry lives in `src/models/registry.py`.
- Default DB: `data/db/<sport>/<season>.db`.
- Processed output: `data/processed/<sport>/<season>/`.
- Output safety: when `--output` exists and `--overwrite` is not provided, the CLI appends numeric suffixes.

## Testing expectations
- Prefer `pytest -q` for full runs; `pytest -q -m "not slow"` for fast suites.
- Use `register_model()` in tests to inject model test doubles.
- Run `tests/test_pipeline_canonization.py` after architectural changes.

## When you’re done
- Summarize changes and reference relevant files in your final response.
- Include tests run and their results.
