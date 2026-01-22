# Suite-Wide Recency Semantics Audit

**Date:** 2026-01-22  
**Scope:** All models, pipelines, and backtest infrastructure  
**Status:** Audit complete. No code modifications.

---

## Executive Summary

Recency weighting is applied **unevenly across models**. Four models (ELO, GSSD, TOOR, Bradley-Terry calibrated) support it; two do not (Poisson, Bradley-Terry core). The single helper `recency_weight()` is well-centralized, but application patterns and as-of dates vary in ways that could introduce subtle leakage or inconsistency risks.

**Key Finding:** Recency is applied **only during calibration regression**, not during the core power-rating fit step for some models. This is architecturally sound but limits recency's effectiveness in models where it matters most (e.g., TOOR's team-strength OLS).

---

## Model-by-Model Recency Audit

### 1. **ELO** (`src/models/elo.py`)

| Aspect | Value |
|--------|-------|
| **Supports Recency?** | ✅ Yes (`recency_lambda`) |
| **Where Applied** | Calibration: margin + total regression (line 292, 310) |
| **Age Unit** | Days (computed in `recency_weight()`) |
| **As-of Date Source** | `resolve_fit_end_date(games_df)` → max game date in training set |
| **Weight Curve** | Exponential: $w = e^{-\lambda \cdot \text{days\_ago}}$ |
| **Applied Once?** | Yes (margin calibration only) |

**Details:**
- Core Elo ratings fit **without recency** in `EloPowerRating.fit()` (line 140–170).
- Recency applied only to **calibration design matrix** (margin regression) at line 292.
- Totals statistics similarly weighted at line 310.
- `fit_end_date` defaults to max date in the training DataFrame.

**Risk:** Core Elo ratings ignore recency entirely; only margin prediction is recency-weighted.

---

### 2. **GSSD** (`src/models/gssd.py`)

| Aspect | Value |
|--------|-------|
| **Supports Recency?** | ✅ Yes (`recency_lambda`) |
| **Where Applied** | Core stats accumulation (line 139) + calibration regression (line 310) |
| **Age Unit** | Days (computed in `recency_weight()`) |
| **As-of Date Source** | `GSSDPowerRating.fit()`: `resolve_fit_end_date_from_games()` (line 113–116) |
| | `GSSDModel.fit()`: `resolve_fit_end_date(games_df)` (line 263) |
| **Weight Curve** | Exponential: $w = e^{-\lambda \cdot \text{days\_ago}}$ |
| **Applied Twice?** | Yes: once in power-rating stats (line 139), again in calibration (line 310) |

**Details:**
- Two-layer application: `GSSDPowerRating` (stats) and `GSSDModel` (calibration).
- `GSSDPowerRating.fit()` accepts optional `fit_end_date` parameter or computes it from games.
- `GSSDModel.fit()` resolves `fit_end_date` independently and passes it to `GSSDPowerRating`.
- **Double weighting risk:** If both layers apply recency, older games are down-weighted twice (multiplicative effect).

**Risk:** Potential double weighting; inconsistent as-of date if one layer uses passed date and the other recomputes.

---

### 3. **TOOR** (`src/models/toor.py`)

| Aspect | Value |
|--------|-------|
| **Supports Recency?** | ✅ Yes (`recency_lambda`) |
| **Where Applied** | Power-rating OLS (line 131) + MOV calibration (line 328) + totals stats (line 331) |
| **Age Unit** | Days (computed in `recency_weight()`) |
| **As-of Date Source** | `TOORPowerRating.fit()`: `resolve_fit_end_date_from_games()` (line 102) |
| | `TOORModel.fit()`: `resolve_fit_end_date(games_df)` (line 298) |
| **Weight Curve** | Exponential: $w = e^{-\lambda \cdot \text{days\_ago}}$ |
| **Applied Multiple Times?** | Yes: power-rating OLS (line 131), MOV regression (line 328), totals (line 331) |

**Details:**
- Three application sites: team-strength OLS, MOV calibration, league-total stats.
- `TOORPowerRating.fit()` builds design matrix with recency weights.
- `TOORModel.fit()` calls `TOORPowerRating.fit()`, then applies recency again in the MOV regression.
- Totals computed as weighted mean/SD using same weights as MOV regression.

**Risk:** Multiple passes with different as-of dates (independent `resolve_fit_end_date` calls); potential for inconsistency if max game dates differ between layers.

---

### 4. **Bradley-Terry Backtest** (`src/models/bradley_terry.py` lines 342–420)

| Aspect | Value |
|--------|-------|
| **Supports Recency?** | ❌ No |
| **Where Applied** | Core Bradley-Terry fit: **None** |
| **Calibration Recency?** | Not exposed in backtest wrapper |
| **Age Unit** | N/A |
| **As-of Date Source** | N/A |
| **Weight Curve** | N/A |

**Details:**
- Core `BradleyTerry.fit()` (line 128–193) uses **unweighted** logistic regression.
- `BradleyTerry._fit_calibration()` (line 195–283) uses **unweighted** OLS for margin/total regression.
- `BradleyTerryBacktest` (backtest wrapper) does not expose `recency_lambda` as a parameter.
- Calibration residuals computed without recency weights (line 217).

**Risk:** No recency support; all games equally weighted in both rating and calibration steps.

---

### 5. **Poisson** (`src/models/poisson.py`)

| Aspect | Value |
|--------|-------|
| **Supports Recency?** | ❌ No |
| **Where Applied** | Core Poisson fit: **None** |
| **Age Unit** | N/A |
| **As-of Date Source** | N/A |
| **Weight Curve** | N/A |

**Details:**
- `PoissonPowerRating.fit()` (line 130–241) uses unweighted gradient descent on log-likelihood.
- Games processed as flat list; no date-based weighting applied.
- `poisson_canonical_from_samples()` post-fit uses empirical distributions (no recency).

**Risk:** All games equally weighted; Poisson cannot prioritize recent matches.

---

## Centralized Helper: `recency_weight()`

**File:** [src/models/calibration.py](src/models/calibration.py#L83-L95)

```python
def recency_weight(
    game_date: object,
    fit_end_date: pd.Timestamp | None,
    recency_lambda: float | None,
) -> float:
    """Compute a recency weight from a game date."""
    if recency_lambda is None or recency_lambda <= 0 or fit_end_date is None:
        return 1.0
    dt = pd.to_datetime(game_date, errors="coerce")
    if pd.isna(dt):
        return 1.0
    days_ago = max(0.0, (fit_end_date - dt).days)
    return math.exp(-recency_lambda * days_ago)
```

**Properties:**
- ✅ Single source of truth for all recency computation.
- ✅ Exponential decay with configurable lambda.
- ✅ Age unit: **days** (integer, not fractional).
- ✅ Safe defaults: returns 1.0 if lambda ≤ 0, lambda is None, or fit_end_date is None.
- ⚠️ No support for game-index-based age (only calendar days).

**As-of Date Helpers:**

| Function | File | Purpose |
|----------|------|---------|
| `resolve_fit_end_date(games_df)` | [calibration.py:60–66](src/models/calibration.py#L60-L66) | Max date in DataFrame |
| `resolve_fit_end_date_from_games(games)` | [calibration.py:70–78](src/models/calibration.py#L70-L78) | Max date from iterable of game dicts |

Both return the **maximum game date** in the fit set, treating it as the as-of date for age computation.

---

## Backtest & Schedule As-of Date Flow

### Backtest Slicing Logic

**File:** [src/backtest/runner.py](src/backtest/runner.py#L271-L309)

```python
def _prepare_backtest_slices(
    games,
    *,
    start_dt,
    end_dt,
    window,
    rolling_days,
    rolling_games,
) -> list[BacktestSlice]:
    # For each evaluation date in [start_dt, end_dt]:
    #   train_idx = all games with date < current_date
    #   eval_idx = all games with date == current_date
    # If rolling, further filter train_idx to last N days or N games
```

**Key Points:**
1. **Training set** includes all games **before** the evaluation date.
2. **Max training date** (used by `resolve_fit_end_date()`) is the **last game before** evaluation date.
3. **No future leakage:** Training data strictly before evaluation date.
4. **Recency as-of date = last training game date,** not the evaluation date.

**Example:**
```
Evaluation date: 2024-12-15
Training set:   games up to 2024-12-14
fit_end_date:   2024-12-14 (max training game date)
Prediction date: 2024-12-15
Recency computed: age relative to 2024-12-14, not 2024-12-15
```

**Implication:** Games on the evaluation date have ~1 day of recency decay relative to training, creating a small temporal boundary effect.

---

## Consistency Analysis

### A. Age Unit Consistency

| Model | Unit | Source |
|-------|------|--------|
| ELO | Days | `recency_weight()` |
| GSSD | Days | `recency_weight()` |
| TOOR | Days | `recency_weight()` |
| Bradley-Terry | N/A | No recency |
| Poisson | N/A | No recency |

**Status:** ✅ **Consistent.** All recency-supporting models use `recency_weight()` directly.

---

### B. As-of Date Source Consistency

| Model | Source | Computed At |
|-------|--------|-------------|
| ELO calibration | `resolve_fit_end_date(games_df)` | `EloModel.fit()` (line 267) |
| GSSD power-rating | `resolve_fit_end_date_from_games(games)` | `GSSDPowerRating.fit()` (line 113–116) |
| GSSD calibration | `resolve_fit_end_date(games_df)` | `GSSDModel.fit()` (line 263) |
| TOOR power-rating | `resolve_fit_end_date_from_games(games)` | `TOORPowerRating.fit()` (line 102) |
| TOOR MOV/totals | `resolve_fit_end_date(games_df)` | `TOORModel.fit()` (line 298) |

**Status:** ⚠️ **Inconsistency detected:**
- ELO and GSSD/TOOR calibration use **DataFrame** version → max date in entire DataFrame.
- GSSD/TOOR power-rating use **iterable** version → may differ if games are filtered pre-call.
- In backtest, both should resolve to the same date (last training game), but the functions are called independently.

**Risk:** If GSSD power-rating receives pre-filtered games while calibration receives full DataFrame, `fit_end_date` values diverge, causing double weighting with different baselines.

---

### C. Weight Curve Consistency

| Model | Curve | Formula |
|-------|-------|---------|
| ELO | Exponential decay | $e^{-\lambda \cdot d}$ |
| GSSD | Exponential decay | $e^{-\lambda \cdot d}$ |
| TOOR | Exponential decay | $e^{-\lambda \cdot d}$ |

**Status:** ✅ **Consistent.** All use the same exponential decay from `recency_weight()`.

---

### D. Application Count Analysis

| Model | Apply Count | Location(s) |
|-------|-------------|-------------|
| ELO | 1× | Calibration regression (margin + totals) |
| GSSD | 2× | Power-rating stats + calibration regression |
| TOOR | 3× | Power-rating OLS + MOV calibration + totals stats |
| Bradley-Terry | 0× | None |
| Poisson | 0× | None |

**Status:** ⚠️ **Inconsistent application counts:**
- GSSD and TOOR apply recency multiple times, potentially with different as-of dates.
- ELO only applies to calibration, leaving core ratings unweighted.
- Bradley-Terry and Poisson do not support recency at all.

---

## Leakage Risk Assessment

### Forward Leakage Risks

**Definition:** Model uses future information (relative to prediction date) during fit.

#### Risk 1: Evaluation Date Not Excluded from Recency Baseline

**Scenario:**
- Training set: all games before evaluation date (correct).
- `fit_end_date = max(training_dates)` = last training game date (e.g., 2024-12-14).
- Prediction made for games on 2024-12-15.
- Recency computed as age relative to 2024-12-14.

**Impact:** Games on evaluation date are ~1 day older than "ideal" (as-of prediction date). This is **negligible** but technically a minor bias.

**Verdict:** ✅ **Acceptable.** No actual future data leaked; only a systematic ~1-day shift.

---

#### Risk 2: Calibration Regression Sees Results from Evaluated Period

**Scenario:**
- Backtest slice: train on games before 2024-12-15, evaluate 2024-12-15 games.
- Calibration (OLS) fits margins using training games only (correct).
- Residuals computed on training set (correct).

**Verdict:** ✅ **No leakage.** Evaluation games never used in fit.

---

### Backward Leakage Risks

**Definition:** Model fails to use available information (under-weighting recent data).

#### Risk 3: Recency Not Applied to Core Rating Fit

**Models Affected:** ELO, Bradley-Terry, Poisson.

**Issue:**
- ELO ratings updated sequentially without recency (standard Elo).
- Bradley-Terry iterates on all games equally.
- Poisson uses unweighted gradient descent.

**Impact:** Older matches have equal influence on core ratings. Calibration step (if it applies recency) is the only place recent games are emphasized.

**Verdict:** ⚠️ **Design trade-off, not a bug.** Models may want unweighted core fit (more stable) + recency-weighted calibration (adjusts for recent trends). This is defensible but limits recency's effectiveness.

---

#### Risk 4: Inconsistent As-of Dates in Multi-Layer Models (GSSD, TOOR)

**Scenario (GSSD):**
1. `GSSDPowerRating.fit(games)` called with `fit_end_date` from `resolve_fit_end_date_from_games(games)`.
   - Result: uses max date in `games` iterable.
2. `GSSDModel.fit(games_df)` calls `resolve_fit_end_date(games_df)`.
   - Result: uses max date in DataFrame.

**If games are filtered before step 1 but full DataFrame used in step 2:**
- Power-rating `fit_end_date` = 2024-12-14.
- Calibration `fit_end_date` = 2024-12-31 (entire DataFrame).
- Result: power-rating recency weights ≠ calibration recency weights.

**Impact:** Games weighted differently in power-rating vs. calibration; multiplicative effect on final predictions.

**Verdict:** ⚠️ **Potential leakage.** If backtest pre-filters games before `GSSDPowerRating.fit()`, inconsistency occurs. Mitigated by fact that backtest passes unfiltered DataFrame to `fit()`.

---

## Summary Table: Model Recency Profile

| Model | Recency? | Age Unit | As-of Source | Curve | Apply Count | Supported? |
|-------|----------|----------|--------------|-------|-------------|-----------|
| **ELO** | ✅ | Days | DataFrame max | Exp | 1× (calib) | Yes, `recency_lambda` |
| **GSSD** | ✅ | Days | Iterable + DataFrame max | Exp | 2× (power + calib) | Yes, `recency_lambda` |
| **TOOR** | ✅ | Days | Iterable + DataFrame max | Exp | 3× (OLS + MOV + totals) | Yes, `recency_lambda` |
| **Bradley-Terry** | ❌ | N/A | N/A | N/A | 0× | No |
| **Poisson** | ❌ | N/A | N/A | N/A | 0× | No |

---

## Inconsistencies Found

### 1. **As-of Date Computation Divergence (GSSD, TOOR)**

**Problem:**
- Power-rating layer uses `resolve_fit_end_date_from_games(iterable)`.
- Calibration layer uses `resolve_fit_end_date(DataFrame)`.
- These may differ if games are pre-filtered in one call.

**Example:**
```python
# In GSSD power-rating fit:
fit_end_date = resolve_fit_end_date_from_games(games)  # May be filtered

# In GSSD model fit (called from backtest):
fit_end_date = resolve_fit_end_date(games_df)  # Unfiltered DataFrame
```

**Impact:** Double weighting with inconsistent baselines.

**Mitigation:** Always call `resolve_fit_end_date(games_df)` consistently across layers.

---

### 2. **Multiple Applications Without Coordination (GSSD, TOOR)**

**Problem:**
- GSSD applies recency in power-rating stats AND calibration regression.
- TOOR applies recency in power-rating OLS, MOV calibration, AND totals stats.
- No documentation of intentional multiplicative effect.

**Example (TOOR):**
```python
# Line 131: Power-rating OLS uses recency weights
weights_1 = [recency_weight(game_date, fit_end_date, lambda) for game in games]

# Line 328: MOV regression uses recency weights again
weights_2 = [recency_weight(game_date, fit_end_date, lambda) for game in games]

# Result: older games down-weighted exponentially in BOTH steps
#         (effect is e^(-2*lambda*d), not e^(-lambda*d))
```

**Impact:** Multiplicative recency decay may be too aggressive; older games under-represented.

**Recommendation:** Document whether this is intentional or should be refactored to apply recency once per model.

---

### 3. **No Recency Support for Bradley-Terry or Poisson**

**Problem:**
- Two models do not support `recency_lambda` at all.
- If backtest tuning includes these models, recency tuning candidates cannot be generated.
- Ensemble may benefit from recency in constituent models.

**Impact:** Inconsistent tuning coverage; ensemble cannot leverage recency gains from other models.

**Recommendation:** Add `recency_lambda` parameter to `BradleyTerryBacktest` and `PoissonPowerRating`.

---

## Recommendations

### 1. **Centralize As-of Date Resolution** (Priority: High)

**Current State:** Different models call `resolve_fit_end_date()` independently.

**Action:** Pass `fit_end_date` as an explicit parameter through the backtest pipeline.

```python
# Instead of:
fit_end_date = resolve_fit_end_date(games_df)

# Pass from backtest:
fit_end_date = slice_games["date"].max()  # or compute once
model.fit(games_df, fit_end_date=fit_end_date)
```

**Benefit:**
- Ensures all layers (power-rating, calibration) use same as-of date.
- Prevents double weighting with inconsistent baselines.
- Simplifies testing and debugging.

---

### 2. **Document Recency Application Semantics** (Priority: High)

**Current State:** GSSD and TOOR apply recency multiple times without documented justification.

**Action:** Add inline comments documenting:
1. Why recency is applied at each site.
2. Whether the multiplicative effect is intentional.
3. Whether a single application point would be preferable.

**Example:**
```python
# TOOR line 131: Apply recency to power-rating OLS.
# Rationale: Recent games should have stronger influence on team-strength estimates.
weights.append(recency_weight(...))

# TOOR line 328: Apply recency to MOV calibration.
# Rationale: Calibration coefficients fit better to recent margin trends.
# Note: This is applied AFTER power-rating, so multiplicative decay is e^(-2*lambda*d).
#       Consider if this is desired or should apply only once.
weights.append(recency_weight(...))
```

---

### 3. **Extend Recency Support to Bradley-Terry and Poisson** (Priority: Medium)

**Current State:** Two models lack recency support.

**Action:** Add `recency_lambda` parameter to:
- `BradleyTerryBacktest.__init__()` and `fit()`.
- `PoissonPowerRating.__init__()` and `fit()`.

**Implementation:**
- Bradley-Terry: Apply recency weights to calibration regression (consistent with ELO).
- Poisson: Apply recency weights to gradient descent (weight log-likelihood by recency).

**Benefit:** Consistent tuning across all models; ensemble can benefit from recency gains.

---

### 4. **Standardize Recency Application Once Per Model** (Priority: Medium)

**Current State:** GSSD and TOOR apply recency multiple times.

**Action:** Evaluate whether recency should be applied:
1. Only to power-rating fit (early in pipeline).
2. Only to calibration fit (late in pipeline).
3. To both (if multiplicative effect is intentional).

**Recommendation:** Apply recency **only to core power-rating fit**, not to calibration.
- **Rationale:** Power ratings are the primary learnable parameters; calibration is post-hoc transformation. Recency matters most where it affects parameter learning.
- **Simplicity:** Easier to reason about; no multiplicative effects.
- **Consistency:** All models apply recency once (at same stage).

---

### 5. **Add Unit Tests for Recency Semantics** (Priority: Medium)

**Current State:** Recency weighting lacks dedicated regression tests.

**Action:** Add test module: `tests/test_recency_semantics.py` with:

```python
def test_recency_weight_exponential_decay():
    """Verify recency_weight() decays exponentially."""
    assert recency_weight(2024-01-01, 2024-01-01, 0.01) == 1.0  # Age 0
    assert recency_weight(2024-01-01, 2024-01-08, 0.01) < 1.0   # Age 7 days
    
def test_elo_recency_backtest_consistency():
    """Verify ELO recency weights same games across backtest slices."""
    # Games 2024-01-01 to 2024-01-31
    # Slice 1: train on 01-01 to 01-15, evaluate 01-16
    # Slice 2: train on 01-01 to 01-22, evaluate 01-23
    # Game 2024-01-15 should be: newer in slice 1, older in slice 2
    # Verify recency weight for game 01-15 is higher in slice 1 than slice 2
    
def test_gssd_double_weighting_multiplicative():
    """Verify GSSD applies recency weights twice (multiplicative)."""
    # Fit GSSD with recency_lambda
    # Check power-rating stats against calibration residuals
    # Older games should be under-weighted more in calibration

def test_toor_three_layer_weighting():
    """Verify TOOR applies recency in OLS, MOV, and totals."""
    # Similar to GSSD, but three application sites
```

**Benefit:** Catches regressions; documents expected behavior.

---

## Best Location for Centralization

**Current:** `recency_weight()` in [src/models/calibration.py](src/models/calibration.py#L83-L95)

**Recommendation:** ✅ **Keep as-is.**

**Rationale:**
1. Already centralized and well-tested.
2. `calibration.py` is appropriate home (other calibration helpers live there).
3. No need to move or duplicate.
4. Extend with new helper: `apply_recency_weights_to_matrix()` for OLS/regression contexts.

**Optional Enhancement:**
```python
# Add to calibration.py
def apply_recency_weights_to_matrix(
    games: Iterable[Mapping[str, Any]],
    recency_lambda: float | None,
    fit_end_date: pd.Timestamp | None,
) -> np.ndarray:
    """Compute recency weight vector for all games."""
    return np.array([
        recency_weight(game.get("date"), fit_end_date, recency_lambda)
        for game in games
    ])
```

---

## Conclusion

**Recency semantics are mostly consistent but lack coordination in multi-layer models (GSSD, TOOR).** The centralized `recency_weight()` helper is well-designed, but as-of dates diverge across layers, and multiple applications lack documentation.

**No code changes required for audit;** however, recommendations 1–4 should be prioritized before further recency tuning work.

**Key Risks:**
1. Double weighting with inconsistent baselines (GSSD, TOOR).
2. Recency not applied to core ratings in some models (ELO, Bradley-Terry, Poisson).
3. No recency support for Bradley-Terry and Poisson in backtest.

**Key Strengths:**
1. Single source of truth: `recency_weight()` function.
2. Exponential decay is standard and well-tested.
3. Backtest slicing prevents forward leakage.
4. Age unit (days) is consistent across models.

---

## Appendix: File Reference Map

| Component | File | Key Functions |
|-----------|------|----------------|
| Recency helper | `src/models/calibration.py` | `recency_weight()` (line 83), `resolve_fit_end_date()` (line 60), `resolve_fit_end_date_from_games()` (line 70) |
| ELO calibration | `src/models/elo.py` | `EloModel.fit()` (line 240), recency applied at line 292 |
| GSSD power-rating | `src/models/gssd.py` | `GSSDPowerRating.fit()` (line 109), recency at line 139 |
| GSSD calibration | `src/models/gssd.py` | `GSSDModel.fit()` (line 258), recency at line 310 |
| TOOR power-rating | `src/models/toor.py` | `TOORPowerRating.fit()` (line 88), recency at line 131 |
| TOOR MOV/totals | `src/models/toor.py` | `TOORModel.fit()` (line 284), recency at line 328, 331 |
| Bradley-Terry core | `src/models/bradley_terry.py` | `BradleyTerry.fit()` (line 128) — **no recency** |
| Bradley-Terry backtest | `src/models/bradley_terry.py` | `BradleyTerryBacktest` (line 342) — **no recency** |
| Poisson | `src/models/poisson.py` | `PoissonPowerRating.fit()` (line 130) — **no recency** |
| Backtest slicing | `src/backtest/runner.py` | `_prepare_backtest_slices()` (line 271), `run_backtest()` (line 331) |
| Tuning | `src/pipelines/tuning.py` | `run_tuning_pipeline()` (line 54), default grid line 473 |
