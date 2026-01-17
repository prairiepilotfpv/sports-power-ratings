# TODO



 - clean up the Dashboard sheet in the schedule output - hasn't been updated to represent the ensembles

## Immediate / Merge-blockers
- [ ] Fix BETS sheet sport-specific matching: ensure bet-sheet parsing and matching is sport-aware (NBA, NHL discrepancies reported). High priority for merging.
- [ ] Verify game ordering is stable across pipelines (ingest -> staging -> models). Several reports show mismatched ordering for NBA/NHL.

## Near-term (address shortly after merge)
- [ ] Revisit ingest workflow organization: document recommended flows (`--input`, `--input-dir`, `--input-text`) and consolidate helpers in `src/ingest`.
- [ ] Clean `Dashboard` sheet in schedule outputs to reflect current ensemble outputs.

## Backlog (nice-to-have)
- [ ] Evaluate a daily ingestion orchestration (idempotent folder pickup) — design only; do not auto-run by default.

## Bet Tracking Suite (status)
- OCR & Parsing: no open critical items; maintain tests and golden fixtures.
- Logging & Review: no open critical items; consider UX polish for CSV writeback flags.
- Reporting & Analytics: stable; keep weekly/monthly workbook formatting tests passing.

## Recently Completed (high level)
- Grouped parsed OCR lines into team bundles, added confidence heuristics and golden tests.
- Bet logger, stake presets, and auto-hold duplicate detection implemented.
- Weekly/monthly workbook polish, CLV ingestion, and PnL worksheet completed.
- CLI: documented legacy `src/cli/ingest.py` as deprecated; `market-ocr` supports `--json-output`.

## Notes & Follow-ups for reviewers
- I archived `REFACTOR_SUMMARY.md` to `docs/archived/REFACTOR_SUMMARY.md` (it looked like a design doc). If you'd prefer deletion, say so.
- `docs/CLI.md` contains a clear deprecated note for the legacy ingest entry point — no change made.
- Run `pytest -q` on CI to ensure no new failures; I did not change code logic.

If you want, I can:
- run a repo-wide grep for lingering `TODO`/`DEPRECATED` instances and create PR-ready edits.
- update `docs/README` or top-level README to point to `docs/archived` for historical notes.
 
 make sure the game order is being observed and followed throughout the suite, it's not matched in NBA or NHL




## Bet Tracking Suite — OCR & Parsing (Active)

## Bet Tracking Suite — Logging & Review

## Bet Tracking Suite — Reporting & Analytics

## Core Pipeline & Housekeeping

## ✅ Recently Completed
	- [x] Group parsed lines into team bundles (three rows per team, gap flags)
	- [x] Confidence heuristics + tagging for OCR lines
	- [x] Golden sample tests for OCR parser fixtures
	- [x] Review CLI for staging rows
	- [x] Bet logger with stake presets
	- [x] Auto-hold detection for duplicate markets
	- [x] Weekly/monthly workbook polish (sparklines + highlights)
	- [x] CLV ingestion and reporting surface
	- [x] PnL scenarios worksheet
	- [x] Document legacy `src/cli/ingest.py` entry point
	- [x] Wire `market-ocr` + bet-report docs with examples
	- [x] Smoke tests for new CLI flags (`--json-output`, bet-report `--type/--format`)
	- [x] Weekly/monthly aggregations + formatted Excel writer
	- [x] OCR fallback path detection for Windows installs
	- [x] `market-ocr` CLI accepts `--json-output` to skip DB writes