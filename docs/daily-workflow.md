# Daily Operations Runbook

Practical, copy-paste checklist to go from raw schedules to logged bets and daily reconciliation. Commands are ordered the way we actually work: tune once per season, then daily ingest → rank → schedule workbook → log bets → settle/report. Replace the example sport/season with yours.

## Conventions used below

- Example variables: `SPORT=nba`, `SEASON=2025-26`, `DB=data/db/nba/2025-26.db`.
- All commands run from the repo root: `python -m src.cli.pipeline ...`.
- The schedule workbook produced by `schedule` contains a `BETS` sheet; that is the source of truth you log into SQLite.

## One-time setup (per machine)

```bash
python -m venv .pyenv
./.pyenv/Scripts/Activate.ps1
pip install -r requirements.txt
```

## Seasonal prep (do once when a new season starts)

1) Import the schedule/results into the season DB (skip if already present):

```bash
python -m src.cli.pipeline import --sport nba --season 2025-26 --input data/raw/nba_schedule.csv
```

2) Tune individual models on history (persist best params for the season):

```bash
python -m src.cli.pipeline tune --model elo --csv data/raw/nba_history.csv \
  --start 2020-01-01 --end 2024-12-31 --metric log_loss \
  --apply-best --sport nba --season 2025-26 --db data/db/nba/2025-26.db
```

3) (Optional) Tune per-market and ensemble weights if you use ensembles:

```bash
# Per-model, per-market tuning (ML example)
python -m src.cli.pipeline tune-model --sport nba --season 2025-26 --model elo --market ML \
  --csv data/raw/nba_history.csv --start 2020-01-01 --end 2024-12-31 \
  --output-dir outputs/tuning/nba/2025-26/elo

# Ensemble weight tuning
python -m src.cli.pipeline tune-ensemble --sport nba --season 2025-26 --market ML \
  --start-date 2020-01-01 --end-date 2024-12-31 --ensemble ensemble_ml_v1 \
  --csv data/raw/nba_history.csv
```

4) Activate tuned params/weights so `rank` and `schedule` pick them up:

```bash
# Promote best runs to actives (all models/markets)
python -m src.cli.pipeline bootstrap-market-actives --sport nba --season 2025-26 --model all

# Verify what is active
python -m src.cli.pipeline tuning-status --sport nba --season 2025-26
```

## Daily flow (repeat every day)

1) Ingest the latest schedule/results into the DB:

```bash
python -m src.cli.pipeline import --sport nba --season 2025-26 --input data/raw/nba_schedule.csv
```

2) Rebuild power ratings (uses active tuned params automatically):

```bash
python -m src.cli.pipeline rank --sport nba --season 2025-26
```

3) Export the schedule workbook (with BETS tab) for today:

```bash
python -m src.cli.pipeline schedule --sport nba --season 2025-26 \
  --as-of-date 2025-01-13 --output data/processed/nba/2025-26/schedule_with_projections.xlsx
```

4) Enter bets in the workbook

- Open `schedule_with_projections.xlsx` and fill the `BETS` sheet (odds, stake, book, market type, game_id already populated). Save the file.

5) Log BETS sheet rows into SQLite (dry-run first, then write):

```bash
# Sanity check without DB writes
python -m src.cli.pipeline betting log-bets --workbook data/processed/nba/2025-26/schedule_with_projections.xlsx \
  --db data/db/nba/2025-26.db --dry-run

# Commit bets to the DB and write back normalized fields into the workbook
python -m src.cli.pipeline betting log-bets --workbook data/processed/nba/2025-26/schedule_with_projections.xlsx \
  --db data/db/nba/2025-26.db --writeback
```

6) Wrap up the day (once games are final):

```bash
# Settle recorded bets against final scores
python -m src.cli.pipeline betting settle-bets --sport nba --season 2025-26

# Produce a daily report workbook (edge buckets, CLV, PnL scenarios)
python -m src.cli.pipeline betting report --sport nba --season 2025-26 --type daily \
  --output outputs/reports/nba-daily-2025-01-13.xlsx
```

## Optional: market snapshot → bets (OCR/CSV) path

Use this when you start from screenshots or market CSVs instead of the BETS sheet.

```bash
# 1) OCR screenshots (or import CSV) into staging
python -m src.cli.pipeline betting market-ocr --sport nba --season 2025-26 \
  --images screenshots/2025-01-13 --book dn --captured-at 2025-01-13T14:30:00Z

# 2) Review any unresolved matches
python -m src.cli.pipeline market-review --sport nba --season 2025-26

# 3) Commit matched rows into market_snapshots (enforces snapshot_run_id format)
python -m src.cli.pipeline betting market-commit --sport nba --season 2025-26 \
  --snapshot-run-id snap-20250113

# 4) Pivot snapshots into candidate bets using stake presets
python -m src.cli.pipeline market-bets --sport nba --season 2025-26 --snapshot-run-id snap-20250113 \
  --stake-preset unit --unit-stake 1.0

# 5) Generate a daily workbook tied to that snapshot
python -m src.cli.pipeline betting daily-workbook --sport nba --season 2025-26 \
  --date 2025-01-13 --snapshot-run-id snap-20250113
```

From here, continue with step 5 above (`betting log-bets`) and the day-end settle/report commands.

## Example day (all-in-one, replace paths/dates)

```bash
python -m src.cli.pipeline import --sport nba --season 2025-26 --input data/raw/nba_schedule.csv
python -m src.cli.pipeline rank --sport nba --season 2025-26
python -m src.cli.pipeline schedule --sport nba --season 2025-26 --as-of-date 2025-01-13 \
  --output data/processed/nba/2025-26/schedule_with_projections.xlsx
# Fill BETS tab manually here
python -m src.cli.pipeline betting log-bets --workbook data/processed/nba/2025-26/schedule_with_projections.xlsx \
  --db data/db/nba/2025-26.db --writeback
python -m src.cli.pipeline betting settle-bets --sport nba --season 2025-26
python -m src.cli.pipeline betting report --sport nba --season 2025-26 --type daily \
  --output outputs/reports/nba-daily-2025-01-13.xlsx
```

## Tips

- Use `--strict` on `schedule` to fail fast if tuned params or ensemble weights are missing.
- `betting log-bets` accepts `--dry-run` for validation and `--writeback` to normalize odds/lines in the workbook after a successful import.
- Keep snapshot_run_id consistent (`snap-YYYYMMDD...`) so market-commit and market-bets runs are traceable.
