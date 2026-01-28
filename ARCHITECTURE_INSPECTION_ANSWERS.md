# Architecture Inspection: Calibration, Columns, Direct Rows, and Market Lookups

Date: 2026-01-28  
Scope: Answers to 5 detailed architectural questions about sports-power-ratings

---

## 1. Calibration Wiring: Where Spread & Total Calibrators Modify Schedule DF

### Current Implementation (Active)
**File**: [src/pipelines/schedule.py](src/pipelines/schedule.py)  
**Function**: `_apply_calibration_to_schedule_df()` [lines 430-700]

**Flow**:
1. Called from `get_schedule()` at [line 336](src/pipelines/schedule.py#L336)
2. Executes **before** `_build_bets_dataframe()` is called
3. Modifies these columns **in-place** for SPREAD and TOTAL:
   - SPREAD: `margin_mean`, `margin_sd` → calibrated values
   - TOTAL: `total_mean`, `total_sd` → calibrated values
4. Appends provenance tags to `win_prob_source` (e.g., `"+calibrated_spread"`)

**Implementation Details**:
- ML calibration [lines 475-540]: Uses `home_win_prob` / `away_win_prob`
- SPREAD calibration [lines 542-614]: Loads `VarianceCalibrator`, applies to `margin_mean`/`margin_sd`
- TOTAL calibration [lines 616-683]: Loads `VarianceCalibrator`, applies to `total_mean`/`total_sd`
- All three use `load_latest_calibrator(sport, season, model, market)` pattern
- Guardrail protection: SD values clipped to `MARGIN_SD_GUARDRAIL_MIN` (0.5) and `TOTAL_SD_GUARDRAIL_MIN` (0.5)

### Unfinished/Deprecated Functions (NOT Used)
**Three deprecated functions exist but are never called** (removed from main flow 2025-01-27):

1. **`_apply_spread_total_calibrators()`** [lines 2182-2245]
   - **Status**: Deprecated, not called
   - **Why**: Interface mismatch — designed for probability calibration but was being passed to distribution calibrators
   - **Issue**: Tried to apply calibration to BETS sheet rows *after* they were already created, which is redundant

2. **`_apply_market_calibrator()`** [lines 2275-2313]
   - **Status**: Deprecated, kept for reference only
   - **Why**: Called only by `_apply_spread_total_calibrators()` (itself deprecated)
   - **Problem**: Attempted `calibrator.transform(pd.Series([raw_prob]))` but distribution calibrators expect `DataFrame` with `(pred_mean, pred_sd)`
   - **Result**: Silent failure via try/except returning `pd.NA`

3. **`_load_market_calibrators()`** [lines 2314-2330+]
   - **Status**: Deprecated, never called
   - **Why**: Only called by the now-defunct `_apply_spread_total_calibrators()`

**Key Design Insight**: Calibration happens **upstream** at the schedule DF level, not downstream at the BETS row level. Distribution parameters are calibrated once, then probabilities are computed from them in `_calculate_model_prob()`.

---

## 2. Column Contract: Canonical Names for Distribution Parameters

### Canonical Column Names

| Parameter | Canonical Name | Used By | Notes |
|-----------|---|---|---|
| **Spread Mean** | `margin_mean` | SPREAD market predictions | Home team advantage in points |
| **Spread SD** | `margin_sd` | SPREAD market predictions | Uncertainty in margin (guardrail min: 0.5) |
| **Total Mean** | `total` | TOTAL market predictions | Predicted game total score |
| **Total SD** | `total_sd` | TOTAL market predictions | Uncertainty in total (guardrail min: 0.5) |
| **Spread Pre-Guardrail SD** | `margin_sd_pre_guardrail` | Debugging/audit trail | Original SD before guardrail clipping (optional tracking) |
| **Total Pre-Guardrail SD** | `total_sd_pre_guardrail` | Debugging/audit trail | Original SD before guardrail clipping (optional tracking) |

### Internal Calibration Contract (Calibrator Input/Output)
When calibrators transform distributions, they always use:
- **Input DataFrame columns**: `pred_mean`, `pred_sd`
- **Output DataFrame columns**: `calibrated_mean`, `calibrated_sd`

This is **different** from the schedule DF columns! Translation happens in `_apply_calibration_to_schedule_df()`:

```python
# SPREAD calibration example
calib_input = pd.DataFrame({
    "pred_mean": margin_mean.loc[valid_mask],      # From schedule's margin_mean
    "pred_sd": margin_sd.loc[valid_mask],          # From schedule's margin_sd
})
calib_result = spread_calibrator.transform(calib_input)
df.loc[valid_mask, "margin_mean"] = calib_result["calibrated_mean"].values
df.loc[valid_mask, "margin_sd"] = calib_result["calibrated_sd"].values
```

### Used by `_calculate_model_prob()`?

**YES, directly consumed**.

**SPREAD market** [lines 2411-2432]:
```python
def _spread_raw_probability(row: Mapping[str, Any]) -> float | None:
    line = _coerce_float(row.get("line"))
    mean = _coerce_float(row.get("margin_mean"))      # ← canonical name
    sd = _coerce_float(row.get("margin_sd"))          # ← canonical name
    raw_sd = _coerce_float(row.get("margin_sd_pre_guardrail"))  # ← debug column
    # ... uses these to compute CDF probability for cover
```

**TOTAL market** [lines 2455-2475]:
```python
def _total_raw_probability(row: Mapping[str, Any]) -> float | None:
    line = _coerce_float(row.get("line"))
    mean = _coerce_float(row.get("total"))            # ← canonical name
    sd = _coerce_float(row.get("total_sd"))           # ← canonical name
    raw_sd = _coerce_float(row.get("total_sd_pre_guardrail"))  # ← debug column
    # ... uses these to compute CDF probability for over/under
```

**Result**: Calibrated `margin_mean`/`margin_sd`/`total`/`total_sd` flow directly into probability calculations. The calibration is **baked into** the `model_prob` values.

---

## 3. Direct Row Source: Which Function Generates Spread/Total Rows?

### Source Function: `_build_bets_dataframe()` 
**File**: [src/pipelines/schedule.py](src/pipelines/schedule.py)  
**Lines**: [1582-2075]

### Generation Strategy: Unified, NOT Market-Specific

**All 6 canonical rows (2×ML, 2×SPREAD, 2×TOTAL) are generated from a SINGLE game in one loop** [lines 1896-2010]:

```python
# CANONICAL 6 ROWS: Construct exactly once per game, then enrich in-place
canonical_specs = [
    # 2 × ML
    ("ML", away_team, ml_source_label, ...),
    ("ML", home_team, ml_source_label, ...),
    # 2 × spread
    ("spread", away_team, spread_source_label, ...),
    ("spread", home_team, spread_source_label, ...),
    # 2 × total
    ("total", "Over", total_source_label, ...),
    ("total", "Under", total_source_label, ...),
]

for market_type, selection, source_label, prob_col1, prob_col2, source_col, ensemble_col in canonical_specs:
    canonical_row = dict(base_row)
    # Enrich with market-specific forecast data
    if market_type == "ML":
        # ML-specific columns
    elif market_type == "spread":
        # SPREAD-specific columns
    else:  # total
        # TOTAL-specific columns
    rows.append(canonical_row)
```

### Key Characteristics

1. **Shared with ML**: YES — all three markets use the same base row template, difference is purely which columns are populated
2. **Market-Specific Enrichment**: 
   - **ML rows**: `home_win_prob`, `away_win_prob`, `win_prob_source`, `ml_ensemble_components_json`
   - **SPREAD rows**: `margin_mean`, `margin_sd`, `spread_source`, `spread_ensemble_components_json`
   - **TOTAL rows**: `total`, `total_sd`, `total_source`, `total_ensemble_components_json`
3. **Invariant**: Exactly 6 rows per game (verified at [line 2039-2048])

### Assembly Process Order
1. **Base row template** [lines 1840-1895]: Initialize all columns to blank
2. **Canonical loop** [lines 1896-2010]: For each market/selection:
   - Create dict copy of base_row
   - Populate market-specific forecast fields
   - LEFT-JOIN market lines from `betting_repository` (if available)
   - Append to rows list
3. **DataFrame construction** [line 2048]: `pd.DataFrame(rows, columns=bets_columns)`
4. **Probability calculation** [line 2061]: `_calculate_model_prob(bets_df)`

---

## 4. Best Insertion Point: Gating Direct Rows on `ensemble_applied`

### Current Status: NO GATING EXISTS
The code currently generates **all 6 rows unconditionally** for every game.

### Where to Gate?

Three candidate locations with **increasing complexity**:

#### Option A: **Row Assembly Time** (CLEANEST)
**Location**: Inside `_build_bets_dataframe()` canonical loop [lines 1896-2010]  
**Mechanism**: Skip appending rows for markets where `ensemble_applied=False`

**Pseudocode**:
```python
for market_type, selection, source_label, ... in canonical_specs:
    # Gate: Skip non-ensemble markets if ensemble_applied is False
    if market_type in ("spread", "total") and not ensemble_applied:
        continue  # Don't generate direct rows
    
    canonical_row = dict(base_row)
    # ... populate, lookup market lines, append
    rows.append(canonical_row)
```

**Pros**:
- Prevents creating rows that will never be used
- Cleanest memory/performance profile
- No complex filtering downstream

**Cons**:
- Requires passing `ensemble_applied` flags into `_build_bets_dataframe()` as parameters
- Must handle per-market flags (ML, SPREAD, TOTAL may differ)

#### Option B: **Forecast Merge Time** (MODERATE)
**Location**: Before `_build_bets_dataframe()` is called, in `get_schedule()` [lines 3680+]  
**Mechanism**: Filter `bets_schedule_df` before passing it to BETS builder

**Pseudocode**:
```python
# After ensemble application (line 3690+), before _build_bets_dataframe():
if not spread_ensemble_applied:
    bets_schedule_df = bets_schedule_df.drop(
        bets_schedule_df[bets_schedule_df["market_type"] == "spread"].index
    )
if not total_ensemble_applied:
    bets_schedule_df = bets_schedule_df.drop(
        bets_schedule_df[bets_schedule_df["market_type"] == "total"].index
    )
```

**Pros**:
- Cleaner contract: BETS builder only sees games it should process
- Separation of concerns (ensemble logic ≠ BETS builder logic)

**Cons**:
- Requires modifying both `get_schedule()` AND `_build_bets_dataframe()` signature
- Less obvious why rows disappear to someone reading the builder function

#### Option C: **BETS Build Entry** (NOT RECOMMENDED)
**Location**: Inside `_build_bets_dataframe()` after row creation [lines 2048-2075]  
**Mechanism**: Filter rows DataFrame before return

**Pseudocode**:
```python
bets_df = pd.DataFrame(rows, columns=bets_columns)
if bets_df.empty:
    return bets_df

# Remove direct rows if ensemble wasn't applied
if not spread_ensemble_applied:
    bets_df = bets_df[bets_df["market_type"] != "spread"]
if not total_ensemble_applied:
    bets_df = bets_df[bets_df["market_type"] != "total"]
```

**Pros**: Works, least invasive to current code

**Cons**:
- Creates rows then throws them away (wasteful)
- Breaks the invariant "6 rows per game" (some games will have <6 rows)
- Hard to debug

### Recommendation: **Option A (Row Assembly Time)**
This is the cleanest architecture:

1. Pass `spread_ensemble_applied` and `total_ensemble_applied` as optional params to `_build_bets_dataframe()`
2. Skip canonical_row append for gated markets
3. Update invariant check to account for conditional row counts
4. Document the gating logic in BETS function docstring

**Signature**:
```python
def _build_bets_dataframe(
    schedule_df: pd.DataFrame,
    *,
    model_name: str,
    as_of_date: str,
    review_run_id: str,
    db_path: str,
    sport: str,
    season: str,
    spread_ensemble_applied: bool = True,  # NEW
    total_ensemble_applied: bool = True,   # NEW
) -> pd.DataFrame:
```

---

## 5. Market Lookup Usage: `_build_bets_dataframe` Only?

### Direct Consumers of `betting_repository`

**Answer**: NO, `_build_bets_dataframe()` is NOT the only consumer.

### All Current Consumers

| Function | File | Purpose | Lookup Type |
|----------|------|---------|------------|
| **`_build_bets_dataframe()`** | [schedule.py:1960-1990](src/pipelines/schedule.py#L1960) | LEFT-JOIN market lines to rows | `get_latest_market_line()` |
| **`staging_bets.py`** | [staging_bets.py:18](src/pipelines/staging_bets.py#L18) | Unknown (import exists) | TBD |
| **`review_runs.py`** | [review_runs.py:22](src/pipelines/review_runs.py#L22) | Unknown (import exists) | TBD |
| **`opportunities.py`** | [opportunities.py:14](src/pipelines/opportunities.py#L14) | Unknown (import exists) | TBD |
| **`market_ocr.py`** | [market_ocr.py:54](src/pipelines/market_ocr.py#L54) | Write staging rows from OCR | `add_staging_row()` |
| **`market_review.py`** | [market_review.py:14](src/pipelines/market_review.py#L14) | Unknown (import exists) | TBD |
| **`daily_workbook.py`** | [daily_workbook.py:15, 22](src/pipelines/daily_workbook.py#L15) | Get opportunities with game info | `get_opportunities_with_game_info()` |
| **`betting_validation.py`** | [betting_validation.py:10](src/pipelines/betting_validation.py#L10) | Unknown (import exists) | TBD |
| **`bets.py`** | [bets.py:21](src/pipelines/bets.py#L21) | Unknown (import exists) | TBD |
| **`action_import.py`** | [action_import.py:11](src/pipelines/action_import.py#L11) | Import action (bet placement) | Unknown |
| **`pipeline.py` (CLI)** | [cli/pipeline.py:2391, 2460](src/cli/pipeline.py#L2391) | CLI commands | Multiple |

### Key Findings

1. **`get_latest_market_line()` is ONLY used in `_build_bets_dataframe()`**
   - Called 3 times: [lines 1960, 1969, 1978](src/pipelines/schedule.py#L1960)
   - Purpose: Look up market line by game_id + market_type + selection
   - Returns dict with `line`, `odds`, `id` (source_market_snapshot_id)

2. **Other betting_repository functions are used elsewhere**:
   - `add_staging_row()` — OCR ingestion
   - `get_opportunities_with_game_info()` — daily workbook, bet tracking
   - `create_review_run()`, `commit_market_snapshots()`, etc. — bet staging pipeline

3. **NOT hidden consumers** — all are explicit imports at module level

### Definition of `get_latest_market_line()`

**Location**: [src/data/betting_repository.py:1046](src/data/betting_repository.py#L1046)

```python
def get_latest_market_line(
    db_path: str,
    *,
    sport: str,
    season: str,
    game_id: str,
    market_type: str,
    selection_team_id: str | None = None,
    selection: str | None = None,
) -> dict | None:
    """Get the latest market line for a game/market/selection.
    
    For ML/SPREAD: requires selection_team_id (team abbreviation)
    For TOTAL: requires selection ('over'/'under')
    
    Returns dict with keys: line, odds, id, ... or None if not found
    """
```

---

## Summary: Architecture Snapshot

| Aspect | Finding |
|--------|---------|
| **Calibration Locus** | `_apply_calibration_to_schedule_df()` — upstream, before BETS builder |
| **Unfinished Spread/Total Path** | `_apply_spread_total_calibrators()` + helpers (DEPRECATED, never called) |
| **Canonical Spread Columns** | `margin_mean`, `margin_sd` (guardrail min: 0.5) |
| **Canonical Total Columns** | `total`, `total_sd` (guardrail min: 0.5) |
| **Internals (Calibrator)** | `pred_mean`, `pred_sd` (different from schedule columns) |
| **Used by Probability Calc** | YES — `_calculate_model_prob()` consumes these directly |
| **Direct Row Generator** | `_build_bets_dataframe()` — unified for all markets |
| **Shared with ML** | YES — same base row, market-specific enrichment |
| **Ensemble Gating** | NO GATING EXISTS — all rows generated unconditionally |
| **Best Gate Point** | Row assembly time (inside canonical loop) — cleanest option |
| **Market Lookup Scope** | `get_latest_market_line()` **only** called from `_build_bets_dataframe()` |
| **Other BR Consumers** | `market_ocr.py`, `daily_workbook.py`, etc. use different functions |

---

## Test Coverage

Key tests to reference:
- Calibration: [tests/test_calibration_bets_integration.py](tests/test_calibration_bets_integration.py)
- BETS builder: [tests/test_pipeline_canonization.py](tests/test_pipeline_canonization.py) (invariant check)
- Probability calc: Same suite (verify columns used)

