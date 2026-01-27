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
## Calibration Provenance Tags (Recent Feature)

### What it does
When the `schedule` command runs, calibration is applied to market-specific probabilities (ML, SPREAD, TOTAL). After calibration, provenance tags are appended to `win_prob_source` to track which markets were calibrated.

### How it works (implementation detail)
- **File**: `src/pipelines/schedule.py` lines 333-573
- **Key code segment**: Function `_apply_calibration_to_schedule_df()` maintains a `calibrated_markets` set
- **Set tracking**: 
  - After ML calibration succeeds → `calibrated_markets.add("ML")`
  - After SPREAD calibration succeeds → `calibrated_markets.add("SPREAD")`
  - After TOTAL calibration succeeds → `calibrated_markets.add("TOTAL")`
- **Tag appending**: At the end (lines 524-551), tags are appended via `_append_calibration_tags()` helper
- **Idempotency**: Function checks case-insensitively if tag exists before appending; sorts tags for deterministic order

### Tag format examples
- Single market: `"elo"` → `"elo+calibrated_ml"`
- Multiple markets: `"elo"` → `"elo+calibrated_ml+calibrated_spread"`
- Idempotent: if you run the function twice, tags don't duplicate

### Testing this feature
```bash
# Run all provenance tag tests
pytest -q tests/test_calibration_bets_integration.py -k "provenance_tags" -v

# Run with logging enabled
pytest -q tests/test_calibration_bets_integration.py -k "provenance_tags" --log-cli-level=INFO
```

### Key test cases
- `test_calibration_provenance_tags_ml_market` — single market tag
- `test_calibration_provenance_tags_multiple_markets` — multiple tags appended
- `test_calibration_provenance_tags_idempotent` — verify no duplicates on repeated calls

### Important: Idempotency guarantee
The `_append_calibration_tags()` helper (lines 533-548) ensures tags are never duplicated even if called multiple times. Use case: if a schedule DataFrame is processed by multiple pipelines or if code retries, tags will remain clean.

### Recent Calibration & Ensemble Notes
- When extending calibration helpers, remember `_apply_calibration_to_schedule_df` and `MarginalDistributionCalibrator.transform` now accept alias columns like `margin_mean`/`total_sd` in addition to `pred_*`. Keep tests aligned with that flexibility so the standalone calibration suite keeps running.
- The total-market ensemble falls back to uniform weights when no tuned weights with positive mass exist; only warn again when configuration includes explicit zero-weight models. Revisit docs/tests if you touch `TotalWeightedAverageEnsemble`.
- Parsing `ml_ensemble_components_json` should tolerate both legacy `prob` and normalized `value` keys so downstream assertions mirror production output.

### Deprecated: BETS Sheet Secondary Calibration (2025-01-27)

The following functions in `src/pipelines/schedule.py` are deprecated and **no longer called**:
- `_apply_spread_total_calibrators()` — tried to apply probability calibration to BETS sheet
- `_apply_market_calibrator()` — helper that had interface mismatch with distribution calibrators
- `_load_market_calibrators()` — loaded calibrators for the deprecated flow

**Reason**: Interface mismatch. `MarginalDistributionCalibrator.transform()` expects a DataFrame with `(pred_mean, pred_sd)` columns, but `_apply_market_calibrator()` passed a `pd.Series([raw_prob])`. The distribution parameters are now calibrated upstream by `_apply_calibration_to_schedule_df()`.

### Testing: Proper Patching Pattern

When mocking `load_latest_calibrator` in tests, patch it **where it's used**:

```python
# CORRECT — patch where imported
import src.pipelines.schedule as schedule_module
orig = schedule_module.load_latest_calibrator
schedule_module.load_latest_calibrator = mock_fn
try:
    result = _apply_calibration_to_schedule_df(...)
finally:
    schedule_module.load_latest_calibrator = orig

# WRONG — schedule.py has its own reference
import src.calibration.io as cal_io
cal_io.load_latest_calibrator = mock_fn  # Won't affect schedule.py
```

See `tests/test_calibration_bets_integration.py` for working examples.

## When you're done
- Summarize changes and reference relevant files in your final response.
- Include tests run and their results.
