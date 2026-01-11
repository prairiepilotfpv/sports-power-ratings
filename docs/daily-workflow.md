# Daily Workflow Checklist

Use this daily checklist to go from fresh schedules and market data to logged bets and reports. Each step includes an example command you can paste and adjust.

## 1) Ingest the latest schedule/results

```bash
python -m src.cli.pipeline import --sport nba --season 2025-26 --input data/raw/nba_schedule.csv
```

## 2) Build rankings (power ratings)

```bash
python -m src.cli.pipeline rank --sport nba --season 2025-26
```

## 3) Generate schedule projections

```bash
python -m src.cli.pipeline schedule --sport nba --season 2025-26 --output data/processed/nba/2025-26/schedule_with_projections.xlsx
```

## 4) Capture market lines (OCR or CSV)

**OCR screenshots:**

```bash
python -m src.cli.pipeline market-ocr --sport nba --season 2025-26 \
  --images screenshots/2025-01-12 --book dn
```

**CSV import (if you already have a lines file):**

```bash
python -m src.cli.pipeline betting market-csv --sport nba --season 2025-26 \
  --csv data/raw/nba_markets.csv --snapshot-run-id snap-20250112 --default-book dn
```

## 5) Review OCR matches

```bash
python -m src.cli.pipeline market-review --sport nba --season 2025-26
```

## 6) Commit matched staging rows into market snapshots (optional)

```bash
python -m src.cli.pipeline betting market-commit --sport nba --season 2025-26 \
  --snapshot-run-id snap-20250112
```

## 7) Validate prerequisites (optional but recommended)

```bash
python -m src.cli.pipeline betting validate --sport nba --season 2025-26 --model elo \
  --date 2025-01-12 --snapshot-run-id snap-20250112 --min-snapshots 10
```

## 8) Generate the daily workbook

```bash
python -m src.cli.pipeline betting daily-workbook --sport nba --season 2025-26 --date 2025-01-12
```

## 9) Log bets from the workbook

```bash
python -m src.cli.pipeline betting log-bets --workbook outputs/daily-nba-2025-01-12.xlsx \
  --db data/db/nba/2025-26.db --writeback
```

## 10) Settle bets and report

```bash
python -m src.cli.pipeline betting settle-bets --sport nba --season 2025-26
```

```bash
python -m src.cli.pipeline betting report --sport nba --season 2025-26 --type daily \
  --output outputs/reports/nba-daily-2025-01-12.xlsx
```
