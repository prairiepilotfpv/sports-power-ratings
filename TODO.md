# TODO

## Immediate / Merge blockers
- [ ] **REGRESSION**: `tests/pipelines/test_schedule_calibration.py::test_spread_weights_filter_out_models_without_spread_outputs` is failing. Expected `{"gssd"}` but got `{'elo', 'gssd'}`. This is related to weight filtering logic (unrelated to BETS gating changes), but needs investigation to determine if it's a pre-existing bug or a side effect of ensemble logic changes.
- [ ] Fix the `BETS` sheet sport-specific matching: ensure the bet-sheet parsing + matching honors NBA/NHL differences and that the signed integers on the sheet match the ingested bets.
- [ ] Verify game ordering is stable across the pipelines (ingest → staging → models → `schedule`). Several reports still show NBA/NHL order mismatches, so confirm the ingesters generate deterministically sorted games and the downstream exports sort by `(date, game_id, away_team)`.
- [ ] Clean the `Dashboard` + `BETS` sheets in the schedule workbook so they reflect the ensemble outputs instead of legacy single-model layouts before merging. Dashboard should be model summaries, BETS is just for determinig bet value for the days, or a specified days, games. 

## Future enhancements
- [ ] **Start Time column**: Parse and persist game start times from Sports Reference CSVs. The parser currently handles rows with/without Time columns but doesn't store start_time in the database. Would enable time-based filtering for betting workflows.
- [ ] Add optional dynamic ensemble weights (per-date/per-game) via a weight provider or combine-time override; keep existing fixed-weight behavior as default.

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

---

## Code Review Findings (2026-01-22)

### Critical: game_id Contract Unification
| Priority | Task | File(s) | Status |
|----------|------|---------|--------|
| 🔴 | Deprecate `build_game_id`, use `make_game_id` everywhere | `src/contracts.py` | ✅ Done |
| 🔴 | Update `ensure_game_id` to use canonical `make_game_id` when sport/season available | `src/contracts.py` | ✅ Done |
| 🔴 | Align `_build_game_key` in ensemble_tuning with canonical format | `src/pipelines/ensemble_tuning.py` | [ ] |
| 🟡 | Add UNIQUE index `games(sport, season, game_id)` via migration | `src/data/migrations.py` | [ ] |
| 🟢 | Add test `test_game_id_canonical_format_matches_db` | `tests/` | [ ] |

### High: Spread Sign Convention & Odds Parsing
| Priority | Task | File(s) | Status |
|----------|------|---------|--------|
| 🔴 | Document `projected_spread = -margin_mean` inversion explicitly | `src/pipelines/projections.py` | [ ] |
| 🟡 | Consolidate `_american_odds_to_prob` to import from `src/utils/odds.py` | `src/parsers/paste_parser.py` | [ ] |
| 🟡 | Add spread sign validation in market ingestion | `src/data/validation.py` | [ ] |
| 🟢 | Add test `test_spread_sign_convention_consistency` | `tests/` | [ ] |

### Medium: Reporting Layer & Business Logic Separation
| Priority | Task | File(s) | Status |
|----------|------|---------|--------|
| 🟡 | Extract edge bucket classification to `src/eval/edge.py` | `src/data/reporting.py` | [ ] |
| 🟡 | Replace silent `except Exception: pass` with logging | `src/data/migrations.py` | [ ] |
| 🟢 | Add `_unknown` sport validation config (permissive guardrails) | `src/eval/validation.py` | [ ] |
| 🟢 | Add test `test_excel_spread_formula_matches_python` | `tests/` | [ ] |

### Low: Model Output Completeness
| Priority | Task | File(s) | Status |
|----------|------|---------|--------|
| 🟢 | Ensure Poisson emits `margin_sd` or documents exclusion | `src/models/poisson.py` | ✅ Done |
| 🔵 | Document streaming-compatible models in STREAMING_BACKTEST.md | `docs/` | ✅ Done |
| 🔵 | Cap tuning `n_jobs` to avoid memory exhaustion | `src/pipelines/tuning.py` | ✅ Done |

### Data Contract Summary
```
game_id formats (to unify):
  ✅ make_game_id: {sport}:{season}:{date}:{hash12}  <- CANONICAL
  ❌ build_game_id: {date}_{home}_{away}             <- DEPRECATED
  ❌ ensure_game_id: {date}_{home}_{away}            <- UPDATE NEEDED
  ❌ _build_game_key: {date}-{home}-{away}           <- UPDATE NEEDED

Spread sign convention:
  • projected_spread = away_minus_home (negative = home favored)
  • margin_mean = home_minus_away
  • Relationship: projected_spread = -margin_mean
```

---

## Notes & follow-ups for reviewers
- `REFACTOR_SUMMARY.md` now exists only under `docs/archived/REFACTOR_SUMMARY.md`; the root copy was removed so reviewers can rely on a single canonical doc.
- `docs/CLI.md` already contains a notice that `src/cli/ingest.py` is deprecated; no further change was made here.
- Run `pytest -q` (covered in `TESTING.md`) to be confident there are no new failures before releasing.
