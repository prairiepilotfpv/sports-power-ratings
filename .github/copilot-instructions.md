# Copilot instructions for sports-power-ratings

You are the lead programmer for this repo. Do not wait for me to specify every detail.

Your job:
- Implement the feature end-to-end with correct architecture and tests.
- Fill in conceptual gaps I missed by proposing sensible defaults and documenting assumptions.
- If something is ambiguous, make the smallest reasonable assumption that preserves existing behavior, and write it down in docs + code comments.
- Only ask me questions if the choice would materially change behavior or require external data I must provide.

Constraints:
- Do NOT refactor unrelated modules.
- Do NOT change existing model fitting/training logic unless explicitly required.
- Prefer additive changes: new modules + small, clearly-scoped integration points.
- Add tests and a short README for the new feature.
- Update CLI/output schemas only if necessary; if you do, keep backward compatibility.

Deliverable format:
1) Implementation plan (bullets, file-level changes)
2) Code changes (actual implementation)
3) Tests + how to run
4) Notes: assumptions, risks, follow-ups

Summary: concise reference to help an AI code agent be productive in this repository (architecture, workflows, conventions, and examples).

## Big picture
- Purpose: ingest Sports-Reference schedules/results, persist per-sport/per-season SQLite DBs, fit power-rating models, produce projections/reports, run bet-tracking OCR, and support backtests/tuning.
- Key areas: `src/cli` (entrypoint), `src/ingest` (parsers/normalizers), `src/data` (paths + DB helpers), `src/models` (model implementations + registry), `src/pipelines` (orchestration: ingest, rank, schedule, backtest, tune, bet tracking), `outputs/` and `data/processed/` (artifacts).
- Bet tracking modules: `src/pipelines/market_ocr.py` (OCR ingestion), `src/cli/betting.py` (CLI wiring for market OCR + reports), `src/data/reporting.py` (weekly/monthly aggregations + Excel writer), `src/data/betting_repository.py` (staging + bets tables).

## Quick dev workflows (use these exact commands)
- Install: `python -m venv .pyenv && ./.pyenv/Scripts/Activate.ps1 && pip install -r requirements.txt`
- Tests: `make test` (runs `pytest -q`). Use `-k` to run specific tests.
- Run CLI (preferred entrypoint):
  - Ingest: `python -m src.cli.pipeline import --sport nba --season 2025-26 --input data/raw/nba.csv`
  - Rank (all models): `python -m src.cli.pipeline rank --sport nba --season 2025-26`
  - Rank (single model): `python -m src.cli.pipeline rank --sport nba --season 2025-26 --model elo`
  - Matchup: `python -m src.cli.pipeline matchup --sport nba --season 2025-26 --matchup "Lakers vs Celtics"`
  - Backtest: `python -m src.cli.pipeline backtest --model bradley-terry --csv nba_results.csv --start 2024-11-01 --end 2024-12-01`
  - Tune: `python -m src.cli.pipeline tune --model elo --csv nba_results.csv --start 2024-11-01 --end 2024-12-01 --metric log_loss`
  - Market OCR: `python -m src.cli.pipeline market-ocr --sport nba --season 2025-26 --input screenshots/ --book dn --captured-at 2024-12-01T14:30:00Z --json-output tmp/lines.json`
  - Bet report: `python -m src.cli.pipeline bet-report --sport nba --season 2025-26 --type weekly --start 2024-12-01 --end 2024-12-31 --format xlsx --output outputs/reports/bets-nba-dec.xlsx`

## Important conventions & patterns
- Paths: default DBs are `data/db/<sport>/<season>.db`. Processed output: `data/processed/<sport>/<season>/` (`src/data/paths.py`). Betting artifacts (reports, OCR JSON) default to `outputs/`.
-- Models: registry in `src/models/registry.py`. Forecast model names (e.g., `bradley-terry`, `elo`, `gssd`, `poisson`, `toor`) map to power-rating classes; backtest runners use the canonical names from `list_backtest_models()`.
- Model params: overrides accepted as JSON string (`--model-params '{"k_factor":20}'`) or file (`--model-params-file params.json`). Tuned params are persisted and auto-loaded when present.
- Input resolution: CLI resolves bare filenames under `data/raw/` when a provided path is missing (see `resolve_input_path`). Ingest supports `--input`, `--input-dir`, or `--input-text` (mutually exclusive).
- Required CSV schema for backtests: `date`, `home_team`, `away_team`, `home_score`, `away_score`. Parsing is lenient (column aliases handled) but missing/invalid dates or negative scores raise errors (see `src/contracts.py` validation helpers).
- OCR requirements: local Tesseract install. `src/ocr/ocr.py` attempts to auto-detect Windows paths when the binary is missing from `PATH`.
- Game ID: `ensure_game_id()` builds deterministic fallback IDs from date/home/away when missing; tests rely on stable game ids (see `tests/`).
- Optional deps: `ssat` is required for `gssd`; Tesseract tools (pytesseract) are optional for OCR ingestion. `OPENAI_API_KEY` is optional for OCR assistance.
- Output conventions: backtests write CSVs and an Excel workbook to `outputs/backtests/<model>/`; tuning writes `outputs/tuning/<...>/`; betting reports land in `outputs/reports/` (Excel via `write_full_report_xlsx`). When `market-ocr` runs with `--json-output`, expect JSON files under `tmp/` or `outputs/market_ocr/` per workflow.
- CLI safety: when `--output` exists and `--overwrite` is not provided, CLI writes to a `-N` suffixed filename (see `_next_available_path`).

## Tests & contracts to reference
- Contract tests ensure models implement expected interfaces: `tests/test_interface_contract.py` and model-specific tests under `tests/models/` and `tests/pipelines/`.
- Add/modify tests to cover any new model or pipeline behavior. Use `register_model()` in tests to inject test doubles.

## Where to look for specifics
- CLI reference: `docs/CLI.md` (detailed flags and examples).
- CLI reference: `docs/CLI.md` (detailed flags and examples). Ensure any new bet-tracking flags (`market-ocr`, `bet-report`, etc.) are mirrored there.
- Model implementations: `src/models/*.py` and `src/models/registry.py`.
- Ingest/parsing: `src/ingest/` (especially `sports_reference` source).
- Validation rules and canonical columns: `src/contracts.py`.
- Global defaults: `src/config.py` (e.g., `DEFAULT_WIN_PROB_K`).
- Reporting helpers: `src/data/reporting.py` for weekly/monthly aggregations and Excel formatting.

## Do / Don't guidance (project-specific)
- Do use `python -m src.cli.pipeline` for integration behavior instead of invoking internals directly unless writing unit tests.
- Do prefer `--model-params-file` for complex parameter overrides (multi-model JSON supported).
- Do group bet-tracking changes with appropriate tests (OCR parsing -> `tests/parsers/` or new folder; reporting -> `tests/pipelines/`).
- Don't assume model names are interchangeable between ranking and backtest contexts — consult `models.registry`.
- Don't change DB schema without updating the validation functions and backtest persistence logic.
- Don't commit large OCR screenshots or generated reports; keep them in ignored `outputs/` / `tmp-` folders.

---
If anything here looks incomplete or unclear (model naming, a missing example, or a workflow you want covered), tell me which section to expand and I’ll iterate.  
(Author: GitHub Copilot — using Raptor mini (Preview))

“When working on models, follow docs/model_canonization_playbook.md”

“Never change model math unless explicitly requested”

“Prefer dedicated projection engine registration over branching in default engine”

“Add regression tests for semantic drift + metric dropout”
"System invariants are protected by tests/test_pipeline_canonization.py - run after any architectural change"

"Game IDs must use make_game_id() - never create custom formats"

"Ensembles: ML/SPREAD/TOTAL classes must all be imported in schedule.py"