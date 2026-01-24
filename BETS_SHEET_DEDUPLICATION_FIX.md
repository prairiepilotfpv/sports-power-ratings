# BETS Sheet Canonical Row Assembly Refactor

## Critical Issue: Duplicate Game Rows
**ROOT CAUSE**: The BETS sheet was displaying **TWO complete sets** of 6 rows per game because:
- Input `bets_schedule_df` contained duplicate `game_id` entries (from multiple market runs or model iterations)
- The original code created 6 rows per input row, so 2 duplicate inputs → 12 rows per game
- Result: Games appeared twice (once with model probs but no lines, once with full line/odds/EV data)

## Solution: Input Deduplication + Canonical Assembly

Refactored `_build_bets_dataframe()` in [src/pipelines/schedule.py](src/pipelines/schedule.py#L727) with two critical fixes:

### 1. Input Deduplication (lines 800-810)
**Added `drop_duplicates(subset=["game_id"], keep="first")`** before row creation to ensure each game appears exactly once in the input, regardless of how many times it was present in the upstream schedule data.

```python
# CRITICAL DEDUPLICATION: Ensure each game_id appears exactly once in input.
# If the input schedule contains duplicate game_ids (e.g., from multiple market runs),
# keep only the first occurrence to prevent creating multiple 6-row blocks per game.
if "game_id" in df.columns:
    initial_count = len(df)
    df = df.drop_duplicates(subset=["game_id"], keep="first")
    if len(df) < initial_count:
        _LOG.warning(
            f"Deduplicated {initial_count - len(df)} duplicate game_id(s) in BETS input. "
            f"This prevents creating multiple 6-row blocks per game."
        )
```

### 2. Canonical Row Assembly (lines 850-1026)

**Initialize all forecast fields** in base row (ensures ML rows don't have spread fields, etc.)
   - 2 × ML (away, home)
   - 2 × spread (away, home)  
   - 2 × total (over, under)
   - Created via a single loop with explicit specifications

3. **Enrich each canonical row in-place** (lines 963-1017)
   - Populate market-specific forecast fields
   - Left-join market lines (missing = blank cells, never separate rows)
   - No additional rows appended after canonical rows complete

4. **Enforce invariant** (lines 1019-1026)
   - Assert each `game_id` appears exactly 6 times
   - Log warning if invariant violated

### Invariant Guarantees

```
✓ Each game_id appears exactly 6 times
✓ BETS row count = 6 × number_of_games_on_date  
✓ Rows created once per game/market/selection (no duplication)
✓ Missing market lines = blank cells (not separate rows)
✓ Market type order preserved: ML, spread, total
✓ Selection order within market type: away before home
```

## Files Changed
- [src/pipelines/schedule.py](src/pipelines/schedule.py#L820-L1026): Complete refactor of `_build_bets_dataframe()` to canonical assembly pattern
- **Removed**: 50+ lines of implicit, scattered row creation logic  
- **Added**: Explicit, documented canonical row assembly with invariant checks

## Tests Added/Updated
- [tests/test_bets_canonical_assembly.py](tests/test_bets_canonical_assembly.py) (NEW): Tests for 6-rows-per-game invariant
  - `test_bets_invariant_exactly_six_rows_per_game`: Verifies all games have exactly 6 rows
  - `test_bets_invariant_no_rows_appended_after_canonical`: Confirms market type order
- [tests/test_bets_sheet_deduplication.py](tests/test_bets_sheet_deduplication.py): Game ordering with/without market lines
- [tests/test_schedule_bets_market_scoping.py](tests/test_schedule_bets_market_scoping.py): Scoping of forecast fields per market
- All 25 existing BETS tests pass unchanged

## Verification

```bash
# Run all BETS tests (25 total)
python -m pytest tests/ -k "bets" -v

# Run canonical assembly tests
python -m pytest tests/test_bets_canonical_assembly.py -v

# Run schedule pipeline tests  
python -m pytest tests/pipelines/test_schedule.py -v
```

## Example: BETS Sheet Structure

```
Game: Lakers vs Celtics (game_id=g1, date=2025-01-24)
─────────────────────────────────────────────────────
Row 1: ML / Lakers / -115 / 0.520 win prob / ...
Row 2: ML / Celtics / +105 / 0.480 win prob / ...
Row 3: SPREAD / Lakers / -3.5 / 3.2 ± 1.1 / ...
Row 4: SPREAD / Celtics / +3.5 / 3.2 ± 1.1 / ...
Row 5: TOTAL / Over / 210.5 / 210.0 ± 5.0 / ...
Row 6: TOTAL / Under / 210.5 / 210.0 ± 5.0 / ...

Game: Warriors vs 76ers (game_id=g2, date=2025-01-24)
─────────────────────────────────────────────────────
Row 7: ML / Warriors / (blank) / 0.550 win prob / ...  
Row 8: ML / 76ers / (blank) / 0.450 win prob / ...
Row 9: SPREAD / Warriors / (blank) / 2.5 ± 1.0 / ...
Row 10: SPREAD / 76ers / (blank) / 2.5 ± 1.0 / ...
Row 11: TOTAL / Over / (blank) / 208.0 ± 5.5 / ...
Row 12: TOTAL / Under / (blank) / 208.0 ± 5.5 / ...

Total rows = 6 × 2 games = 12
─────────────────────────────────────────────────────
```

## Benefits

1. **Clarity**: Explicit canonical assembly pattern is easy to audit and understand
2. **Safety**: Invariant checks prevent accidental row duplication
3. **Correctness**: Missing market lines handled uniformly (blanks, not extra rows)
4. **Maintainability**: Clear separation between row creation and enrichment phases
5. **Backward compatible**: No API changes; all existing tests pass
