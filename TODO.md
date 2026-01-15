# TODO

make sure the BETs sheet match for all sports, must not be sports agnostic right now and needs to be

Make ingest easier/more organized
 - one big import? like everything is one folder and it all gets nabbed every morning? NO, probably not, that's a like of error checking

 - clean up the Dashboard sheet in the schedule output - hasn't been updated to represent the ensembles
 


## Bet Tracking Suite — OCR & Parsing (Active)
- [ ] (no open items)

## Bet Tracking Suite — Logging & Review
- [ ] (no open items)

## Bet Tracking Suite — Reporting & Analytics
- [ ] (no open items)

## Core Pipeline & Housekeeping
- [ ] (no open items)

## ✅ Recently Completed
- **Bet Tracking Suite — OCR & Parsing**
	- [x] Group parsed lines into team bundles (three rows per team, gap flags)
	- [x] Confidence heuristics + tagging for OCR lines
	- [x] Golden sample tests for OCR parser fixtures
- **Bet Tracking Suite — Logging & Review**
	- [x] Review CLI for staging rows
	- [x] Bet logger with stake presets
	- [x] Auto-hold detection for duplicate markets
- **Bet Tracking Suite — Reporting & Analytics**
	- [x] Weekly/monthly workbook polish (sparklines + highlights)
	- [x] CLV ingestion and reporting surface
	- [x] PnL scenarios worksheet
- **Core Pipeline & Housekeeping**
	- [x] Document legacy `src/cli/ingest.py` entry point
	- [x] Wire `market-ocr` + bet-report docs with examples
	- [x] Smoke tests for new CLI flags (`--json-output`, bet-report `--type/--format`)
	- [x] Weekly/monthly aggregations + formatted Excel writer
	- [x] OCR fallback path detection for Windows installs
	- [x] `market-ocr` CLI accepts `--json-output` to skip DB writes