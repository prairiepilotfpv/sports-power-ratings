# Daily Workflow

Concise daily checklist for schedule + market ingestion, OCR review, bet logging, and reporting.

## Checklist

1. Data ingest (schedule + market lines)

```powershell
# Schedule ingest
python -m src.cli.pipeline import --sport <sport> --season <season> --input <schedule_csv_or_html>

# Market lines via OCR (JSON-only dry run)
python -m src.cli.pipeline market-ocr --sport <sport> --season <season> \
  --input <screenshots_dir_or_file> --book <book> --json-output <lines_json>

# Or import a market CSV
python -m src.cli.pipeline betting market-csv --sport <sport> --season <season> \
  --csv <market_csv> --snapshot-run-id <run_id> --default-book <book>
```

2. OCR review

```powershell
python -m src.cli.pipeline market-review --sport <sport> --season <season>
```

3. Review workbook generation

```powershell
python -m src.cli.pipeline betting review-generate --sport <sport> --season <season> --model <model> \
  --snapshot-run-id <run_id>
```

4. Manual entry in BETS sheet

```powershell
start <review_workbook_path>
```

5. Log bets

```powershell
python -m src.cli.pipeline betting log-bets --workbook <review_workbook_path> --db <db_path> --writeback
```

6. Settle and report

```powershell
python -m src.cli.pipeline betting settle-bets --sport <sport> --season <season>
python -m src.cli.pipeline betting report --sport <sport> --season <season> --type daily --output <report_path>
```
