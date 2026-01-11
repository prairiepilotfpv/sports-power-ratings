# Bet Tracking (Market OCR + Betting Reports)

This document explains how to log a bet (from OCR/CSV -> review workbook -> DB) and how to exercise the behavior with the test suite.
For a concise daily checklist, see [docs/daily-workflow.md](docs/daily-workflow.md).

**Overview**
- Purpose: ingest sportsbook screenshots/CSV, stage parsed market rows, resolve them to canonical games, generate a review workbook, and log bets from the workbook into `bets` table for later backtests/settlement.
- Key code: `src/data/betting_repository.py` (DB schema + helpers), `src/pipelines/staging_bets.py` (pivoting), `src/pipelines/bets.py` (log/settle), `src/cli/betting.py` (betting subcommands).

**Daily checklist (step-by-step)**
1. Ingest schedule + market lines (OCR or CSV).
2. Review OCR matches (`market-review`) and accept/reject as needed.
3. Commit matched staging rows into `market_snapshots` (so snapshots are queryable).
4. (Optional) Run pre-flight validation (DB integrity, predictions, snapshot counts).
5. Generate a review run + opportunities for the day.
6. Build the unified daily workbook (projections + snapshots + OCR + EV/BETS):

```powershell
python -m src.cli.pipeline betting daily-workbook --sport <sport> --season <season> --date <YYYY-MM-DD>
```

7. Open the workbook, decide stakes in `BETS`, and log bets with `log-bets`.
8. Settle and run daily reports after results are in.

**Prerequisites**
- Create and activate a virtualenv and install deps:

```powershell
python -m venv .pyenv
.\.pyenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- Tesseract (optional for live OCR): install via Chocolatey `choco install tesseract` or download from upstream. The code will try to auto-detect common Windows paths, but `pytesseract` must be importable for live OCR.

**DB path**
- Default DB: `data/db/<sport>/<season>.db` (helpers in `src/data/paths.py`). Many CLI commands accept `--db` to override.

**End-to-end flow (commands)**
1. Ingest screenshots or a CSV of market lines.

Top-level pipeline supports both a top-level `market-ocr` command and a nested `betting market-ocr` command; either form is acceptable depending on your workflow. Example (top-level):

```powershell
python -m src.cli.pipeline market-ocr --sport nba --season 2025-26 \
  --images screenshots/ --book dn \
  --json-output tmp/lines.json
```

Or via the betting subcommand:

```powershell
python -m src.cli.pipeline betting market-ocr --sport nba --season 2025-26 \
  --images screenshots/ --book dn \
  --json-output tmp/lines.json
```

Note: `--json-output` writes parsed market lines to JSON without touching the DB (useful for dry-runs).

If you already have a CSV of market lines, use the CSV import command instead of OCR:

```powershell
python -m src.cli.pipeline betting market-csv --sport nba --season 2025-26 \
  --csv data/raw/nba_markets.csv --snapshot-run-id run_20251201 \
  --default-book dn
```

Expected CSV columns: `market_type`, `selection`, `line`, `odds`, `team_home` (or `home_team`), `team_away` (or `away_team`), `game_date` (YYYY-MM-DD). Optional: `game_id`, `book`, `source`, `captured_at`. Invalid rows are rejected and counted.

2. Commit matched staging rows into `market_snapshots` (optional, used when snapshots are desired):

```powershell
python -m src.cli.pipeline betting market-commit --sport nba --season 2025-26 \
  --snapshot-run-id snap-20250101T000000Z
```

3. Generate a review workbook for a model (this creates `META` + `BETS` sheets used by `log-bets`):

```powershell
python -m src.cli.pipeline betting review-generate --sport nba --season 2025-26 --model elo \
  --snapshot-run-id snap-20250101T000000Z
```

To generate a formula workbook (EV + BETS formulas for `implied_prob`, `edge`, `ev`), add `--formula-workbook` (or `--formula`).
In formula workbooks, `implied_prob`, `edge`, and `ev` are formula-driven, while `odds`, `line`, and `model_prob` remain editable inputs.
Note: guardrails can filter predictions before they reach the workbook; see `docs/bet-evaluation.md` for details.
`review-generate` requires `--snapshot-run-id`; optionally add `--snapshot-date` to constrain snapshots to a specific captured date.

**OCR_RAW sheet (audit trail)**
- The review workbook now includes an `OCR_RAW` sheet by default. It joins `market_snapshots` to `market_snapshot_staging` so you can trace any EV/BETS row back to the OCR source.
- Key columns: `source_market_snapshot_id`, `image_path`, `raw_text`, `team_home_raw`, `team_away_raw`, `match_status`, `match_confidence`, `hold_reason`, `captured_at`, `book`, `market_type`, `selection`, `line`, `odds`.
- Use `source_market_snapshot_id` to link `EV.source_market_snapshot_id` or `BETS.source_market_snapshot_id` back to the OCR row for audit/review.
- To disable the sheet, pass `--no-include-ocr-raw` to `review-generate`.

4. Edit the `BETS` sheet in the produced workbook. Required and supported columns (case-sensitive names expected by the `log_bets` parser):
- `game_id` (preferred) - canonical `game_id` from the DB. If missing, the row cannot be logged idempotently.
- `market_type` - e.g., `ML`, `spread`, `total`.
- `selection` - team name or `home`/`away` depending on workbook conventions.
- `line` - numeric point spread or total (float) where applicable.
- `odds` or `price` - American odds as integer (e.g., `+120`, `-150`). `price` is supported as alternative to `odds`.
- `stake` — stake amount; blank or zero means PASS (row skipped).
- Optional: `book`, `opportunity_id` (to trace source opportunity), `log_status`, `bet_id`, `logged_at` (these may be written back by `--writeback`).

Important behavior:
- Blank `stake` is treated as PASS and will be skipped by `log_bets`.
- Idempotency is enforced by the UNIQUE key `(review_run_id, game_id, market_type, selection)`. Re-running `log_bets` will update an existing row rather than duplicating.
- If `review_run_id` is not provided to `log_bets`, the function will attempt to read it from the `META` sheet where a row with `key == review_run_id` and `value == <id>` is expected.
- Formula workbooks include formulas in `EV` and `BETS` for `implied_prob`, `edge`, and `ev`. Edit `selection`, `line`, `odds`, or `model_prob` in `BETS` to refresh the EV calculations.
  - Formula-driven columns: `implied_prob`, `edge`, `ev`.
  - Editable inputs: `odds`, `line`, `model_prob` (plus other non-formula columns).

5. Log bets from the workbook into the DB (dry-run first):

```powershell
# Dry-run, parse workbook but do not write
python -m src.cli.pipeline betting log-bets --workbook outputs/review-nba.xlsx --db data/db/nba/2025-26.db --dry-run

# Real run: write bets to DB and (optionally) write back bet_id/logged_at into workbook
python -m src.cli.pipeline betting log-bets --workbook outputs/review-nba.xlsx --db data/db/nba/2025-26.db --writeback
```

Notes on `--writeback`:
- When `--writeback` is used, `log_bets` writes `bet_id`, `logged_at`, and `log_status` back to the `BETS` sheet for each logged row. The workbook is updated in-place.

**Logging from pivoted staging rows into bets**
- If you prefer to pivot reviewed staging rows into bets (instead of editing a workbook), use the `market-bets` pipeline which converts matched staging rows into `bets` using stake presets:

```powershell
python -m src.cli.pipeline market-bets --sport nba --season 2025-26 --stake-preset unit --unit-stake 1.0 --dry-run

# To actually insert:
python -m src.cli.pipeline market-bets --sport nba --season 2025-26 --stake-preset unit --unit-stake 1.0
```

The pivot respects `match_status` filters (default: `matched`) and will auto-hold duplicates found in the same image unless `--disable-auto-hold` is passed. The summary printed shows inserted/held/skipped counts.

**Table/column notes and helper behavior**
- Table `bets` columns: `id, review_run_id, game_id, market_type, selection, line, odds, stake, book, logged_at, clv_close_odds, clv_close_line, status, outcome, profit, source_opportunity_id`.
- The system will attach the latest CLV snapshot (closing line/odds) to a bet when available.

**Tests to run**
- Run the focused betting tests that validate `log_bets`, pivoting, and repository behavior:

```powershell
pytest -q tests/test_cli_log_bets.py tests/test_cli_log_bets_integration.py \
  tests/test_bets_pipeline.py tests/test_betting_repository.py
```

- For a faster smoke-check during doc edits, run just the log-bets tests:

```powershell
pytest -q tests/test_cli_log_bets.py tests/test_cli_log_bets_integration.py -q
```

**Troubleshooting & tips**
- Pre-flight validation can catch missing predictions or snapshots before you generate a workbook:

```powershell
python -m src.cli.pipeline betting validate --sport nba --season 2025-26 --model elo \
  --date 2025-12-01 --snapshot-run-id snap-20251201 --min-snapshots 10
```

- If validation reports "No predictions found", check that rankings exist for the date/model and rerun `python -m src.cli.pipeline rank` before `review-generate`.
- If validation reports "Insufficient market snapshots", confirm you committed staging rows for the snapshot run id or pass the correct `--snapshot-date`.
- High-risk staging rows (e.g., odds outside typical ranges or matched rows missing `game_id`) are logged as warnings when inserted. Resolve by reviewing OCR matches (`market-review`) and re-ingesting or correcting source data if needed.
- If `log_bets` raises `ValueError: review_run_id not provided`, ensure the workbook has a `META` sheet with `key == review_run_id` or pass `--review-run-id` (the CLI currently reads `review_run_id` from `META` automatically when absent).
- If DB path resolution fails, pass `--db data/db/<sport>/<season>.db` or call the CLI with `--sport` and `--season` so the CLI can infer the path.
- Use `--dry-run` liberally to confirm parsing before writing.

If you'd like, I can add an example review workbook template (minimal `META` + `BETS` sheets) and a small helper to populate `game_id` values from `market_snapshot_staging` for easier logging.
