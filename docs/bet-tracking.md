# Bet Tracking (Market OCR + Betting Reports)

This page documents the bet-tracking subsystem: OCR ingestion, staging, snapshots, opportunities, and bet logging.

Overview
- Purpose: ingest sportsbook screenshots/CSV, parse market lines via OCR (or CSV), persist staging rows, resolve to games, surface opportunities, and record logged bets.
- Key code: `src/data/betting_repository.py` (DB schema + helpers), `src/ocr/ocr.py` (OCR utility), and pipelines under `src/pipelines` (market OCR, review, reports).

Prerequisites
- Python virtualenv with project requirements installed (`pip install -r requirements.txt`).
- Tesseract OCR is optional for tests (the code will attempt to import `pytesseract` and auto-detect common Windows install paths). If you want live OCR, install Tesseract on Windows (e.g., `choco install tesseract`) or download from: https://github.com/tesseract-ocr/tesseract.

Testing the bet-tracking stack
- Run the betting-related unit tests (these validate OCR parsing, staging resolution, repository behavior, CLI wiring):

```bash
pytest -q tests/test_bets_pipeline.py tests/test_betting_repository.py \
  tests/test_betting_resolve.py tests/test_cli_betting.py \
  tests/test_cli_log_bets.py tests/test_market_ocr.py tests/test_market_ocr_bundling.py -q
```

DB and schema
- Per-sport/per-season DB path: `data/db/<sport>/<season>.db` (helpers in `src/data/paths.py`).
- The betting schema and idempotent initializer live in `src/data/betting_repository.py` (run `init_db(db_path)` to create tables).
- Betting tables include `market_snapshot_staging`, `market_snapshots`, `forecast_snapshots`, `opportunities`, `bets`, and `clv_snapshots`.

CLI usage (examples)
- Market OCR (ingest screenshots or JSON output):

```bash
python -m src.cli.pipeline market-ocr --sport nba --season 2025-26 \
  --input screenshots/ --book dn --captured-at 2024-12-01T14:30:00Z \
  --json-output tmp/lines.json
```

- Produce a bet report (Excel):

```bash
python -m src.cli.pipeline bet-report --sport nba --season 2025-26 \
  --type weekly --start 2024-12-01 --end 2024-12-31 --format xlsx \
  --output outputs/reports/bets-nba-dec.xlsx
```

Notes & next steps
- Unit tests currently mock or exercise OCR behavior; if you plan to run live OCR ingestion, ensure `pytesseract` is installed and `tesseract.exe` is available on PATH (or installed in one of the common Windows locations). `src/ocr/ocr.py` will attempt common Windows locations automatically.
- If you'd like, I can add a short CLI example that runs a full end-to-end dry-run using sample screenshots/fixtures.

