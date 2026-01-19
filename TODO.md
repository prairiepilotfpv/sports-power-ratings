# TODO

## Immediate / Merge blockers
- [ ] Fix the `BETS` sheet sport-specific matching: ensure the bet-sheet parsing + matching honors NBA/NHL differences and that the signed integers on the sheet match the ingested bets.
- [ ] Verify game ordering is stable across the pipelines (ingest → staging → models → `schedule`). Several reports still show NBA/NHL order mismatches, so confirm the ingesters generate deterministically sorted games and the downstream exports sort by `(date, game_id, away_team)`.
- [ ] Clean the `Dashboard` + `BETS` sheets in the schedule workbook so they reflect the ensemble outputs instead of legacy single-model layouts before merging.

## Pipeline & output sanity
- [ ] Audit the market ingestion/bets workflow to ensure the retired `market-review` path no longer blocks production and that any manual steps are documented or removed.
- [ ] Confirm every tuning/ensemble flow clearly logs when tuned params are used (`rank`/`schedule` should not print `Missing active params...` when tuned values exist, and `tuning-status` should match what is applied).
- [ ] Keep the `docs/cli`/`docs/daily-workflow` guide synchronized with what the CLI actually supports, especially where the old `market-review`/`market-commit` commands were retired.

## Documentation & housekeeping
- [x] Remove the duplicated Quickstart/Installation sections from `README.md` (now a single Quickstart + matching Workflows section describes the end-to-end flow).
- [ ] Keep this TODO list structured and current—move resolved work down to “Done” or archive it, and keep future tasks grouped by area so reviewers immediately see what is outstanding.
- [x] Link the new `docs/archived` location from the README (the `REFACTOR_SUMMARY.md` move is complete and the README link points directly there).

## Recently Completed
- [x] Cleaned the README’s duplicate sections, kept only the canonical Quickstart, and inserted a dedicated Workflows summary so the table of contents stays accurate.
- [x] Archived `REFACTOR_SUMMARY.md` under `docs/archived` and removed the root copy so reviewers only see one source of truth.

## Backlog (nice-to-have)
- [ ] Evaluate a daily ingestion orchestration (idempotent folder pickup) for future automation (design only; do not auto-run by default).
- [ ] Continue checking bet-tracking analytics formatting, guardrails, and logging UX to keep weekly/monthly workbook formatting tests green.

## Notes & follow-ups for reviewers
- `REFACTOR_SUMMARY.md` now exists only under `docs/archived/REFACTOR_SUMMARY.md`; the root copy was removed so reviewers can rely on a single canonical doc.
- `docs/CLI.md` already contains a notice that `src/cli/ingest.py` is deprecated; no further change was made here.
- Run `pytest -q` (covered in `TESTING.md`) to be confident there are no new failures before releasing.
