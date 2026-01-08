# TODO

## Bet Tracking Suite — OCR & Parsing (Active)
- [ ] **Group parsed lines into team bundles**: update `src/pipelines/market_ocr.py` so JSON mode emits exactly three ordered rows per team (moneyline, spread, total) and flags gaps for manual review.
- [ ] **Confidence heuristics + tagging**: add per-line metadata (font row index, odds confidence, spread/total keywords) to the JSON to speed up downstream QA.
- [ ] **Golden sample tests**: create fixture images + expected JSON in `tests/fixtures/ocr/` and add regression tests that exercise the OCR parser sans Tesseract (use stored text dumps).

## Bet Tracking Suite — Logging & Review
- [ ] **Review CLI**: build `python -m src.cli.pipeline market-review` to list `market_snapshot_staging` rows filtered by `match_status` and allow accepting/rejecting matches (persisting `game_id`).
- [ ] **Bet logger**: implement a CLI that pivots reviewed staging rows into `bets` entries (stake, book, market metadata) with simple stake presets.
- [ ] **Auto-hold detection**: add rules that detect duplicate markets from the same screenshot and tag them before logging bets.

## Bet Tracking Suite — Reporting & Analytics
- [ ] **Weekly/monthly workbook polish**: add sparkline columns and highlight rules to the `edge_buckets` + `clv` sheets in `write_full_report_xlsx()`.
- [ ] **CLV ingestion**: extend the DB schema helper to store close odds/line snapshots and surface them in the reports.
- [ ] **PnL scenarios**: create an optional worksheet that simulates Kelly/unit sizing scenarios using the aggregated bets.

## Core Pipeline & Housekeeping
- [ ] Sunset or document the legacy `src/cli/ingest.py` entry point.
- [ ] Wire `market-ocr` + bet-report commands into `docs/CLI.md` with end-to-end examples.
- [ ] Add smoke tests for the new CLI flags (`--json-output`, bet-report `--type/--format`).

## ✅ Recently Completed
- [x] Weekly/monthly aggregations + formatted Excel writer.
- [x] OCR fallback path detection for Windows installs.
- [x] `market-ocr` CLI accepts `--json-output` to skip DB writes.