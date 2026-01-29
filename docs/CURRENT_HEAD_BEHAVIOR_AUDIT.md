# Current Head Behavior Audit

**Date**: 2026-01-28  
**Status**: INVESTIGATION COMPLETE - NO CODE CHANGES  
**Scope**: Mapping all implicit head-like behaviors in the codebase before implementing formal heads system

---

## Executive Summary

This document audits where and how each model currently produces the three canonical market outputs:
- **ML**: `p_home_win` (probability home team wins)
- **SPREAD**: `margin_mean`, `margin_sd` (distribution of point spread)
- **TOTAL**: `total_mean`, `total_sd` (distribution of combined score)

**Key Finding**: The codebase has **multiple overlapping systems that silently derive missing outputs**, creating implicit "heads" at three levels:
1. **Model predict() layer**: Some models emit native outputs; others emit partial outputs
2. **Projection engine layer**: Silently derives missing margin/total/probability outputs
3. **Calibration layer**: Transforms (but does not create) distribution parameters after schedule building

This creates a complex derivation chain where **the same output can be produced by multiple code paths**, depending on which code layer is triggered and what the model native ly provided.

---

## Section 1: Model Output Inventory

### Per-Model Output Matrix

| Model | Native p_home_win | Native margin_mean | Native margin_sd | Native total_mean | Native total_sd | Comments |
|-------|---|---|---|---|---|---|
| **Bradley-Terry** | ✅ Direct (logistic) | ✅ Calibrated (margin_a + margin_b × d_value) | ✅ Calibrated (margin_sigma) | ✅ Calibrated (total_c + total_u × \|d_value\|) | ✅ Calibrated (total_sigma) | `project_matchup()` method returns all 5 fields natively. Used directly in predict(). |
| **Elo** | ❌ Missing | ❌ Missing | ❌ Missing | ⚠️ Learned (total_mean from training) | ❌ Missing | predict() DOES NOT call project_matchup. Relies on projection engine (see below). |
| **TOOR** | ❌ Missing | ⚠️ Learned (margin from coefficients + neutral flag) | ⚠️ Learned (error_term or conditional SD) | ⚠️ Learned (total_mean from training) | ❌ Missing | predict() DOES NOT call project_matchup. Relies on projection engine. |
| **GSSD** | ❌ Missing | ⚠️ Learned (from per-team scoring stats) | ⚠️ Learned (from scoring variance) | ⚠️ Learned | ❌ Missing | predict() DOES NOT call project_matchup. Relies on projection engine. |
| **Poisson** | ✅ Skellam-based | ✅ λ_home - λ_away | ✅ SD from variance(κ×total) | ✅ λ_home + λ_away | ✅ SD from variance(κ×total) | Calls `poisson_canonical_from_rates()` or `poisson_canonical_from_samples()` in predict(). All 5 fields emitted natively. |

### Detailed Per-Model Analysis

#### Bradley-Terry (`src/models/bradley_terry.py`)
- **predict() path**: Calls `project_matchup()` on calibration object. Returns complete GamePrediction with all 5 fields.
- **Native outputs**: ALL (via `project_matchup()`)
  - `p_home_win`: Direct Bradley-Terry logistic probability + learned home advantage
  - `margin_mean`: Linear transformation of rating difference via calibrated coefficients (margin_a, margin_b)
  - `margin_sd`: Constant learned from margin residuals during fit
  - `total_mean`: Linear transformation via calibrated coefficients (total_c, total_u)
  - `total_sd`: Constant learned from total residuals during fit
- **Calibration**: Post-fit phase learns (margin_a, margin_b, margin_sigma, total_c, total_u, total_sigma) from observed game margins/totals. Bounds totals using robust quantiles.
- **Fallback**: No fallback. If calibration coefficients are invalid (NaN, inf, ≤0), uses hardcoded fallback values (1.0).

**Status**: ✅ **SELF-CONTAINED HEAD** - No derivation from other models or fallback to projection engine.

---

#### Elo (`src/models/elo.py`)
- **predict() path**: Does NOT have project_matchup() method. predict() calls `build_forecasts_df()` (forecasting service) which invokes projection engine.
- **Native outputs**: NONE (no predict() implementation that returns complete GamePrediction)
  - Must be filled by projection engine
- **Fit phase**: 
  - Learns `_win_prob_k` (logistic curve steepness) via `fit_win_prob_scale()`
  - Learns `_win_prob_bias` (logistic curve offset) via `fit_win_prob_bias()`
  - Learns `_total_mean` (global average total from training games)
  - Learns conditional SD model via `fit_conditional_sd()` to predict margin SD from margin magnitude
- **Why no predict()**: Elo is designed as a ranking-only model. All projection is delegated to `_elo_projection_engine()`.

**Status**: ❌ **REQUIRES PROJECTION ENGINE** - Elo outputs must be synthesized by projection engine from ratings and learned parameters.

---

#### TOOR (`src/models/toor.py`)
- **predict() path**: Does NOT have project_matchup() method. Relies on `_toor_projection_engine()`.
- **Native outputs**: NONE
  - Must be filled by projection engine
- **Fit phase**:
  - Learns coefficients (home_advantage, home_coeff, away_coeff, error_term) via OLS/scipy optimization
  - Learns `_win_prob_k` and `_win_prob_bias` via same calibration as Elo
  - Learns `_total_mean` (global average total)
  - Learns conditional SD model for margin predictions
- **Why no predict()**: Like Elo, TOOR is rating-based; all projection delegated to projection engine.

**Status**: ❌ **REQUIRES PROJECTION ENGINE** - Outputs synthesized by `_toor_projection_engine()`.

---

#### GSSD (`src/models/gssd.py`)
- **predict() path**: Does NOT have project_matchup() method. Relies on `_gssd_projection_engine()`.
- **Native outputs**: NONE
  - Must be filled by projection engine
- **Fit phase**:
  - Learns per-team scoring stats (points for/against) via weighted accumulators
  - Computes margin mean as (home_pf - away_pa) - (home_pa - away_pf)
  - Computes margin SD from scoring variance
  - Learns `_win_prob_k` and `_win_prob_bias` via calibration
  - Learns `_total_mean` (global average total)
- **Why no predict()**: Scoring-based model; projects via shared engine.

**Status**: ❌ **REQUIRES PROJECTION ENGINE** - Outputs synthesized by `_gssd_projection_engine()`.

---

#### Poisson (`src/models/poisson.py`)
- **predict() path**: Calls canonical helper functions in predict():
  - `poisson_canonical_from_rates()` (preferred when ratings available)
  - `poisson_canonical_from_samples()` (fallback when sampling needed)
  - Both return complete dict with all 5 fields
- **Native outputs**: ALL (via canonical helpers)
  - `p_home_win`: Skellam-based probability (with empirical tie handling)
  - `margin_mean`: λ_home - λ_away
  - `margin_sd`: √(κ × (λ_home + λ_away))
  - `total_mean`: λ_home + λ_away
  - `total_sd`: √(κ × (λ_home + λ_away))
- **Calibration**: Post-fit learns `kappa` (overdispersion parameter) from Poisson rate empirics.
- **Fallback**: If canonical calculation fails (numerical issues), falls back to Monte Carlo simulation via `simulate_matchup()` then calls `poisson_canonical_from_samples()`.

**Status**: ✅ **SELF-CONTAINED HEAD** - Poisson emits all outputs natively; no projection engine derivation.

---

### Summary: Model Output Coverage

- **Self-Contained Models** (emit all 5 fields natively):
  - Bradley-Terry ✅
  - Poisson ✅

- **Projection Engine Dependent** (emit 0 fields, rely on engine for all derivations):
  - Elo ❌
  - TOOR ❌
  - GSSD ❌

**Implication**: The projection engine is **not optional** for 3 of 5 models. It acts as an implicit head layer that **derives all missing outputs** using learned parameters and defaults.

---

## Section 2: Derivation Path Audit

### Projection Engine: The Primary Head Layer

**File**: `src/pipelines/projection_engines.py` (550 lines)

#### Architecture
The projection engine uses a registry pattern: each model has a registered projection engine function that converts ratings/parameters into GamePrediction fields.

```python
_ENGINES: dict[str, ProjectionEngine] = {}

def register_projection_engine(model_id: str, engine: ProjectionEngine) -> None:
    _ENGINES[model_id] = engine

def get_projection_engine(model: Any) -> ProjectionEngine:
    # Resolves model_id → engine function
    ...
```

#### Registered Engines

| Engine | Model(s) | Triggered When | Output Strategy |
|--------|----------|---|---|
| `_bt_native_matchup()` + fallback | bradley-terry | Only calls `model.project_matchup()` if available | Uses native output if present, fallsback to `_rating_projection_engine()` |
| `_rating_projection_engine()` | Default (Elo, TOOR, GSSD) | When ratings exist | Derives all 5 fields from ratings + learned parameters |
| `_elo_projection_engine()` | elo | Always called | Wraps `_rating_projection_engine()`, overrides win_prob_source to "logistic" |
| `_toor_projection_engine()` | toor | Always called | Calls `model.project_matchup()` if available, else wraps `_rating_projection_engine()` + applies logistic bias |
| `_gssd_projection_engine()` | gssd | Always called | Calls `model.project_matchup()` if available, else wraps `_rating_projection_engine()` + applies logistic bias |
| Default (Poisson, others) | Any model without registered engine | Fallback | Wraps `_rating_projection_engine()` |

#### Detailed Derivation Paths

##### Path 1: `_rating_projection_engine()` (Primary Derivation)
**Lines**: 65-291  
**Triggered**: When rating-based models (Elo, TOOR, GSSD) generate forecasts  
**Inputs**: Ratings dict, context (home_advantage, win_prob_k, conditional SD params, base_total, etc.)

**Derivation Logic**:
1. **margin_mean** ← `project_game()` helper (uses ratings + home advantage)
   - `project_game()` (`src/pipelines/projections.py` line 200+) computes:
     - `margin = home_rating - away_rating` (neutral venue)
     - `margin += home_advantage` (if not neutral)
   - Result assigned directly to `margin_mean`

2. **margin_sd** ← One of three sources (in priority order):
   - ✅ ConditionalSDModel if available (learned during fit)
     - `sd = intercept + slope × |margin|`
     - Guardrails applied: `sd = clip(sd, cfg.margin_sd_min, cfg.margin_sd_max)`
     - Fallback: LEAGUE_MARGIN_SD_DEFAULT if result invalid
   - ✅ `guardrail_margin_sd()` if no conditional SD
     - Uses raw_margin_std from context OR DEFAULT_MARGIN_SD_FALLBACK
     - Applied guardrails: `clip(sd, cfg.margin_sd_min, cfg.margin_sd_max)`
   - ✅ DEFAULT_MARGIN_SD_FALLBACK (last resort)

3. **p_home_win (normal_p_home_win)** ← `_home_win_prob_from_margin(margin_mean, margin_sd)`
   - Normal CDF of margin distribution at 0
   - `P(home wins) = 1 - CDF(0 | μ=margin_mean, σ=margin_sd)`
   - **This is a derived probability; Bradley-Terry logistic prob is NOT used**

4. **projected_win_prob** ← `normal_p_home_win`
   - **For Elo only**: Overridden to `logistic_p` if `predict_probability()` available
   - **For others**: Uses normal CDF (margin-based)

5. **total_mean** ← One of two sources:
   - ✅ Model-specific total via context (total_intercept, total_slope, etc.)
     - `total = total_intercept + total_slope × rating_diff`
   - ✅ Matchup average from scoring_averages dict
   - ✅ base_total parameter (context['base_total'])
   - ✅ None if all missing

6. **total_sd** ← Always `context.get("total_std")` OR DEFAULT_TOTAL_SD_FALLBACK
   - **No guardrails applied** (unlike margin_sd)
   - **No conditional model**
   - **Constant per model/season**

7. **Projected scores** ← Arithmetic from margin and total:
   - `projected_home_score = (total_mean + margin_mean) / 2`
   - `projected_away_score = (total_mean - margin_mean) / 2`

**Source Labels Assigned**:
- `win_prob_source`: "margin_normal"
- `margin_dist_assumption`: "normal_approx"

**Guardrails Applied**:
- ✅ Margin SD: Clipped to [MARGIN_SD_GUARDRAIL_MIN, MARGIN_SD_GUARDRAIL_MAX]
- ❌ Total SD: NO guardrails (potential issue)
- ✅ Projected scores: Derived arithmetic (no direct guardrails)

---

##### Path 2: `_elo_projection_engine()` Variant
**Lines**: 292-315  
**Override**: Win prob source and logistic flag

**Derivation Changes**:
- Wraps `_rating_projection_engine()` output
- **Replaces** `model_p_home_win` and `win_prob_source`
- If `model.predict_probability()` available (it is), uses logistic probability:
  - `p_home_win = 1 / (1 + exp(-adjusted_rating_diff / k_factor))`
  - `win_prob_source`: "logistic"
- **Margin/total derivations unchanged** (still from `_rating_projection_engine()`)

---

##### Path 3: `_toor_projection_engine()` Variant
**Lines**: 317-382  
**Override**: Win prob derivation + bias adjustment

**Derivation Logic**:
1. Checks for `model.project_matchup()` (TOOR doesn't have it, so skipped)
2. Wraps `_rating_projection_engine()` output
3. **Overrides win_prob calculation**:
   - `projected_spread = -margin_mean`
   - `adjusted_spread = projected_spread - winprob_bias`
   - `logistic_p = 1 / (1 + exp(adjusted_spread / win_prob_k))`
   - `win_prob_source`: "logistic"
4. **Margin/total unchanged**

---

##### Path 4: `_gssd_projection_engine()` Variant
**Lines**: 384-450  
**Same as `_toor_projection_engine()`** (duplicate logic with different docstring)

---

##### Path 5: Bradley-Terry Native Path
**Lines**: 451-545  
**Triggered**: When `model.project_matchup()` method exists AND is available

**Logic**:
1. Calls `model.project_matchup(home, away, neutral=neutral)` directly
2. Bradley-Terry returns dict with all 5 fields natively
3. Engine **copies directly** to output:
   ```python
   if canonical is not None:
       return {
           "projected_home_score": canonical["projected_home_score"],
           "projected_away_score": canonical["projected_away_score"],
           ...
           "margin_mean": canonical["margin_mean"],
           "margin_sd": canonical["margin_sd"],
           "total_mean": canonical["total_mean"],
           "total_sd": canonical["total_sd"],
           ...
       }
   ```
4. **Bypass**: No derivation; native output passed through

---

### Calibration Layer: The Secondary Transformation Layer

**File**: `src/pipelines/schedule.py` (lines 430-573)  
**Function**: `_apply_calibration_to_schedule_df()`

**Purpose**: NOT to derive missing outputs, but to **refine distribution parameters** after schedule building.

**Timing**: Applied AFTER projection engine and model.predict() calls have filled all fields.

**Transformations Applied**:

| Market | Input Columns | Output Columns | Calibrator Type | Fallback |
|--------|---|---|---|---|
| ML | home_win_prob, away_win_prob | (replaced in-place) | Probability calibrator (Platt/Isotonic) | Original values if calibrator missing |
| SPREAD | margin_mean, margin_sd | (replaced in-place) | Distribution calibrator (MarginalDistribution) | Original values if calibrator missing |
| TOTAL | total_mean, total_sd | (replaced in-place) | Distribution calibrator (MarginalDistribution) | Original values if calibrator missing |

**Key Insight**: Calibration operates **only on columns that already exist and have values**. It does NOT create outputs from scratch; it refines/shifts them.

**Provenance Tags**: Appends "+calibrated_ml", "+calibrated_spread", "+calibrated_total" to `win_prob_source` to track which markets were calibrated.

**Guardrails Applied**:
- After SPREAD calibration: `margin_sd = clip(sd, MARGIN_SD_GUARDRAIL_MIN)` (no max guardrail)
- After TOTAL calibration: `total_sd = clip(sd, TOTAL_SD_GUARDRAIL_MIN)` (no max guardrail)

---

## Section 3: Projection Engine Role & Implicit Head Behavior

### Summary: Projection Engine as Implicit Head

The projection engine is an **implicit multi-headed system** because it:

1. **Derives all missing outputs** for 3 of 5 models (Elo, TOOR, GSSD)
2. **Uses learned parameters** (home_advantage, win_prob_k, conditional_sd model, total_mean, etc.) cached during fit
3. **Applies consistent transformations** across all models:
   - Margin → win probability (normal CDF)
   - Logistic curve (with bias) for some models
   - Projected scores (arithmetic from margin + total)
4. **Assigns provenance labels** (e.g., "margin_normal", "logistic") indicating derivation method

### What the Projection Engine Assumes

```python
context = {
    "ratings": {...},           # Home/away team ratings (REQUIRED)
    "rating_units": "points",   # Ratings in spread units (REQUIRED for validation)
    "home_advantage": 0.0,      # Applied to ratings (default 0 for neutral)
    "win_prob_k": 20.0,         # Logistic curve steepness (default from config)
    "base_total": 0.0,          # Minimum fallback for total (default 0)
    "scoring_averages": {...},  # Matchup-specific totals
    "total_intercept": None,    # Model-specific total formula
    "total_slope": None,        
    "conditional_sd_intercept": None,  # SD formula parameters
    "conditional_sd_slope": None,
    "margin_std": None,         # Raw margin SD (fallback to default if missing)
    "total_std": None,          # Raw total SD (fallback to default if missing)
    "sport": "nba",             # For validation config (guardrails)
    "game_id": "...",           # For logging/debugging
    "game_date": "2025-01-28",  # For logging/debugging
}
```

**Assumptions**:
- ✅ Ratings dict is non-empty and fully populated for home/away teams
- ✅ Ratings are in "points" units (implied spread magnitude)
- ✅ Home advantage is a scalar (applied as additive adjustment)
- ❌ No assumption about total_mean source (uses fallback if missing)
- ❌ No assumption about margin_sd source (uses default if missing/invalid)
- ❌ Does NOT validate total_sd guardrails (no max guardrail applied)

**Market-Specific Differences**:
- **ML market**: Derives probability from margin distribution (normal CDF)
- **SPREAD market**: Margin mean/sd passed directly; probability derived
- **TOTAL market**: Total mean/sd passed directly; no direct probability derivation
- **Bradley-Terry only**: Uses native project_matchup() output (bypasses derivation)

---

## Section 4: Calibration Boundary Check

### What Calibration Modifies

**Boundary**: Calibration is applied AFTER `build_forecasts_df()` (which invokes projection engine) and AFTER model.predict() calls.

**Timeline**:
1. ✅ Model fit: Learn parameters (ratings, total_mean, sd models, etc.)
2. ✅ Model predict() / Projection engine: Generate GamePrediction with all fields
3. ✅ **CALIBRATION**: Transform (but do NOT create) distribution parameters
4. ✅ Ensembles: Combine forecasts from multiple calibrated/uncalibrated models
5. ✅ BETS sheet: Generate betting rows using final ensemble outputs

**Columns Modified by Calibration**:

| Column | Market | Calibrator Type | Allows Replacement | Allows Empty→Full |
|--------|--------|---|---|---|
| home_win_prob | ML | Probability | ✅ Yes (skellam→platt) | ❌ No (requires valid input) |
| away_win_prob | ML | Probability | ✅ Yes | ❌ No |
| margin_mean | SPREAD | Distribution | ✅ Yes | ❌ No |
| margin_sd | SPREAD | Distribution | ✅ Yes | ❌ No |
| total_mean | TOTAL | Distribution | ✅ Yes | ❌ No |
| total_sd | TOTAL | Distribution | ✅ Yes | ❌ No |

**Calibration does NOT**:
- ❌ Create missing outputs (e.g., fill None → float)
- ❌ Derive one output type from another (e.g., margin → probability)
- ❌ Apply different transformations to different models
- ❌ Check ensemble validity or drop models

**Calibration ONLY**:
- ✅ Refines existing numeric values
- ✅ Tracks which markets were calibrated (provenance tags)
- ✅ Applies guardrails (min clipping for SD columns)
- ✅ Logs diagnostic info (pre/post deltas, clip rates, health status)

### Pre/Post Calibration Values: Ensemble Perspective

**Ensembles receive POST-CALIBRATION values** (calibration runs before ensemble input collection).

In `_build_market_forecasts_for_ensembles()` (line 895):
```python
market_schedule = _build_schedule_dataframe(
    ...,
    model=model_name,
    ...,
)
# _build_schedule_dataframe calls:
#   1. build_forecasts_df() → projections
#   2. _apply_calibration_to_schedule_df() → transforms
# So market_schedule already has calibrated values

for _, r in subset.iterrows():
    if market == Market.ML:
        p_raw = r.get("model_p_home_win")  # ← POST-CALIBRATION
    rows_by_market[market.name].append({
        "p_home_win": p_raw,
        "margin_mean": r.get("margin_mean"),  # ← POST-CALIBRATION
        "margin_sd": r.get("margin_sd"),      # ← POST-CALIBRATION
        ...
    })
```

**Critical**: Ensembles combine **already-calibrated** forecast rows. If calibration dropped a model's output to None, the ensemble will exclude that model from the weighted average.

---

## Section 5: Failure Case Trace

### Concrete Failure Scenario: GSSD Drops from ML Ensemble

**Setup**: NBA 2025-26 season, 3-model ensemble (Bradley-Terry, Elo, GSSD), ML market tuning.

**Timeline**:

#### Step 1: Model Fit (off the critical path for this failure)
- GSSD fits on 2024-25 games, learns per-team scoring stats and conditional SD model

#### Step 2: Model predict() → Projection Engine
**File**: `src/pipelines/schedule.py` line 980+ (`build_forecasts_df`)

GSSD predict() method does NOT exist (model doesn't override BaseModel.predict()).
Instead, `build_forecasts_df()` uses the **forecasting service** which invokes the projection engine.

```python
# forecasting/forecast_service.py (not examined but called from schedule.py)
forecast_rows = _project_row(
    game_row,
    model=gssd_instance,
    projection_engine=get_projection_engine(gssd_instance),  # → _gssd_projection_engine
    context={...}
)
```

**Projection Engine Call**: `_gssd_projection_engine()` (line 384)
- Checks for `gssd_instance.project_matchup()` → Does not exist
- Wraps `_rating_projection_engine(home, away, gssd, context)`
- `_rating_projection_engine()` derives margin_mean/sd and total_mean/sd from ratings
- `_gssd_projection_engine()` then applies logistic bias: 
  ```python
  logistic_prob = logistic_win_prob(adjusted_spread, win_prob_k)
  ```
- **Result**: All 5 fields populated (no None values at this point)
- **win_prob_source**: "logistic"

#### Step 3: Calibration (does NOT cause failure here)
`_apply_calibration_to_schedule_df()` runs:
- Loads latest ML calibrator for GSSD
- If calibrator exists, transforms home_win_prob in-place
- **Source tag**: Appends "+calibrated_ml" to win_prob_source
- **No columns set to None**

#### Step 4: Ensemble Input Collection
**File**: `src/pipelines/schedule.py` line 1000+ (`_build_market_forecasts_for_ensembles`)

```python
for model_name in models:  # ["bradley-terry", "elo", "gssd"]
    ...
    market_schedule = _build_schedule_dataframe(
        ..., model=model_name, ...
    )
    for _, r in market_schedule.iterrows():
        if market == Market.ML:
            p_raw = r.get("home_win_prob_raw") or r.get("model_p_home_win") or r.get("home_win_prob")
            rows_by_market[Market.ML.name].append({
                "game_id": r.get("game_id"),
                "model_name": "gssd",
                "p_home_win": p_raw,  # ← Value from projection engine (should be valid)
                ...
            })
```

**So far**: No failure yet. GSSD p_home_win is populated.

#### Step 5: Ensemble Weighting & Filtering
**File**: `src/pipelines/schedule.py` line 3450+ (`_filter_market_weights_for_forecast`)

```python
def _filter_market_weights_for_forecast(
    weights={
        "bradley-terry": 0.5,
        "elo": 0.3,
        "gssd": 0.2,  # Pre-tuned weight
    },
    forecast_df=forecast_rows_for_ml_market,
    market=Market.ML,
):
    # Validate that all models have required columns:
    required_columns = ("p_home_win",)
    for model_name, group in forecast_df.groupby("model_name"):
        missing = [col for col in required_columns if col not in group.columns or not group[col].notna().any()]
        if missing:
            model_validity[normalized] = False
            invalid_reasons[normalized] = f"missing {', '.join(missing)}"
    
    # Filter weights: Only include models with p_home_win values
    filtered_weights = {}
    for model, weight in candidate_weights.items():
        if weight <= 0:
            drop_reasons[model] = f"weight={weight}"
            continue
        if model not in forecast_models:
            drop_reasons[model] = "missing forecast rows"
            continue
        if required_columns and not model_validity.get(model, True):
            reason = invalid_reasons.get(model, "missing required fields")
            drop_reasons[model] = reason  # ← GSSD DROPPED HERE if p_home_win is NaN
            continue
        filtered_weights[model] = weight
```

**Failure Trigger**: If `p_home_win` is NaN for GSSD:
- GSSD marked invalid in model_validity
- GSSD weight filtered to 0.0
- GSSD removed from ensemble
- Bradley-Terry + Elo weights renormalized to sum to 1.0

#### Root Cause Analysis

**Where could GSSD p_home_win become NaN?**

1. ❌ Projection engine returned None: No (all 5 fields emitted)
2. ❌ Calibration set to None: No (only transforms existing values)
3. ⚠️ **Schedule building skipped game**: Possible if game_id mismatch
4. ⚠️ **Forecasting service failed**: Possible if exception caught silently
5. ⚠️ **Margin SD guardrail collapsed**: If margin_sd → 0, then normal CDF undefined, return NaN (POSSIBLE)

**Most Likely**: Margin SD collapse → Normal CDF returns NaN → p_home_win becomes NaN

**Evidence** in projection engine (`src/pipelines/projection_engines.py` line 234):
```python
normal_p_home_win = _home_win_prob_from_margin(
    projection.margin, margin_sd
) if projection.margin is not None else None

# In projections.py:
def _normal_cdf(x: float, *, mean: float, sd: float) -> float:
    if sd <= 0 or not math.isfinite(sd):
        raise ValueError("Standard deviation must be positive and finite.")
    ...

# If margin_sd is invalid:
# - projection engine catches exception → returns None
# - GamePrediction p_home_win remains as projected_win_prob (which might be valid from logistic)
# - But if BOTH paths fail, p_home_win = None
```

**Full Trace Summary**:
1. Model: GSSD fit with insufficient margin SD samples
2. Projection: margin_sd fallback to DEFAULT_MARGIN_SD_FALLBACK (might be 0 or NaN)
3. Calibration: Distribution calibrator applied, but if input was invalid, output stays invalid
4. Schedule: p_home_win = None (because _normal_cdf failed AND logistic path not used for GSSD)
5. Ensemble: p_home_win NaN → GSSD marked invalid → Filtered out with reason "missing p_home_win"
6. Output: ML ensemble only has Bradley-Terry + Elo; GSSD weight = 0.0

---

## Section 6: Implicit Head System Design Issues

### Issue 1: Overlapping Derivation Responsibilities

**Problem**: Three systems can derive the same output:
1. **Model predict()**: Bradley-Terry, Poisson emit native outputs
2. **Projection engine**: Elo, TOOR, GSSD derive all outputs; also handles Bradley-Terry fallback
3. **Calibration**: Transforms (not derives) distribution parameters

**Risk**: When implementing formal heads:
- Must decide **single source of truth** per field
- Cannot have projection engine deriving margin_mean while heads also derive it
- Calibration can continue (it transforms, not derives)

**Example Conflict**:
- Bradley-Terry `project_matchup()` returns margin_sd
- Projection engine `_rating_projection_engine()` also computes margin_sd (different formula!)
- Which one is used? **projection engine is skipped if project_matchup() succeeds**
- But fallback exists: if project_matchup() fails, projection engine takes over

---

### Issue 2: Missing Guardrails on Total SD

**Problem**: Margin SD has guardrails (min/max clipping); Total SD does not.

**In projection engine** (`_rating_projection_engine()` line 240):
```python
total_sd = context.get("total_std")
if total_sd is None or total_sd <= 0:
    total_sd = DEFAULT_TOTAL_SD_FALLBACK
# NO guardrails applied
```

**In calibration** (`_apply_calibration_to_schedule_df()` line 530):
```python
clipped_sd = pre_guardrail_sd.clip(lower=TOTAL_SD_GUARDRAIL_MIN)
# Only MIN guardrail; no MAX guardrail
```

**Risk**: Total SD can be arbitrarily large, leading to:
- Overconfident total probability estimates
- Miscalibrated totals ensemble
- BETS rows with extreme implied odds

---

### Issue 3: Probability Derivation Ambiguity

**Problem**: `p_home_win` can come from multiple sources:
- Bradley-Terry: Logistic probability (native)
- Elo: Margin-based normal CDF (projection engine), then overridden to logistic
- TOOR: Margin-based normal CDF (projection engine), then logistic applied via bias
- GSSD: Margin-based normal CDF (projection engine), then logistic applied via bias
- Poisson: Skellam probability (native)

**Risk**: Ensembles combine probabilities derived from **incompatible distributions**:
- Logistic (Bradley-Terry, Elo, TOOR, GSSD) vs. Skellam (Poisson)
- Normal approximation (margin-based) vs. Skellam (Poisson)
- **No explicit label** distinguishing which model uses which derivation

**Current Mitigation**: win_prob_source tag (e.g., "logistic", "margin_normal", "poisson_skellam") but this is not consistently enforced.

---

## Section 7: Freeze Plan for Heads System Implementation

### What MUST STOP Deriving When Heads Exist

Once formal heads are implemented, the following code **must NOT continue deriving outputs**:

1. **Projection Engine Derivation** (lines 65-450 in projection_engines.py)
   - Current: `_rating_projection_engine()` derives margin_mean, margin_sd, total_mean, total_sd
   - **Freeze**: Remove all derivation logic; treat all inputs as pre-computed
   - **Rationale**: Heads system will provide pre-derived values; projection engine becomes a passthrough validator

2. **Model-Specific Projection Variants** (lines 292-450 in projection_engines.py)
   - Current: `_elo_projection_engine()`, `_toor_projection_engine()`, `_gssd_projection_engine()` override win_prob
   - **Freeze**: Remove win_prob overrides; treat model output as authoritative
   - **Rationale**: Heads will emit win_prob directly; no need for engine re-derivation

3. **Calibration Derivation** (none currently, but check before modifying)
   - Current: Calibration only **transforms**, not **derives**
   - **No change needed**: Calibration can continue; it refines (not creates) outputs

4. **Fallback Defaults** (throughout projection_engines.py and schedule.py)
   - Current: DEFAULT_MARGIN_SD_FALLBACK, DEFAULT_TOTAL_SD_FALLBACK, etc. used when values missing
   - **Freeze**: Heads system must provide all values; no fallbacks in projection engine
   - **Rationale**: Fallbacks hide derivation decisions from users

### What CAN Continue

1. ✅ **Calibration transformations** (preserves distribution refinement)
2. ✅ **Provenance tagging** (tracks which heads were used)
3. ✅ **Ensemble logic** (combines multi-head outputs)
4. ✅ **BETS probability computation** (from distribution parameters)
5. ✅ **Guardrailing** (clipping SD values to min/max)

### Key Implementation Constraints

1. **Heads MUST emit all 5 fields** for each model
   - No partial output allowed (e.g., margin_mean without margin_sd)
   - If a head cannot derive a field, it must raise an error (not return None)

2. **Heads MUST be invoked before projection engine**
   - Timeline: Model predict() with heads → Projection engine validation → Calibration → Ensembles
   - Projection engine becomes a **validator only**, not a **generator**

3. **Projection engine becomes a schema adapter**
   - Maps native head output (GamePrediction) → context dict for schedule building
   - Validates all required fields are populated
   - No derivation logic remains

4. **Ensemble weights must know which head was used**
   - Current: One weight per model
   - Future: Potential per-head weights if multiple heads per model emerge
   - win_prob_source tag must distinguish head origin (e.g., "bt_direct", "elo_heads", "poisson_heads")

---

## Section 8: Open Issues & Edge Cases

### Issue 1: Bradley-Terry Margin-based Probability vs. Logistic
**File**: `src/models/bradley_terry.py` line 303
```python
# Which probability is authoritative?
normal_p_home_win = 1.0 - self._normal_cdf(0.0, mean=margin_mean, sd=margin_sd)
model_p_home_win = self.predict_probability(home_team, away_team, venue=venue)

# In projection:
"p_home_win": float(p_home_win),  # Uses model_p_home_win
"normal_p_home_win": float(normal_p_home_win),  # Exposed for reference

# In GamePrediction:
p_home_win=p_home_win,  # ← Which one wins?
```
**Open Question**: Should Bradley-Terry heads use logistic or margin-based? Current implementation uses logistic (via model_p_home_win), but this is not explicit in win_prob_source.

---

### Issue 2: TOOR Conditional SD Model Chain
**File**: `src/models/toor.py` line 680+
```python
if self._conditional_sd_model is not None:
    sds = self._conditional_sd_model.intercept + self._conditional_sd_model.slope * abs_margins
```
**Edge Case**: Conditional SD model learned from sparse residuals (< MIN_CALIBRATION_SAMPLES) → fallback to constant. If heads system uses TOOR, must ensure conditional SD model is available.

---

### Issue 3: Poisson Tie Handling Variance
**File**: `src/models/poisson.py` line 83
```python
p_home_win = (
    np.mean(margin_samples > 0)
    + tie_split_home * np.mean(margin_samples == 0)
)
```
**Variance**: Tie split (tie_split_home parameter) is hardcoded per model. Heads system must preserve this parameter; cannot be overridden per-sport.

---

## Appendix: Code Locations Reference

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| Model predict() | src/models/{model}.py | Varies | Native output emission |
| Projection engines | src/pipelines/projection_engines.py | 1-550 | Implicit head derivation |
| Projection helpers | src/pipelines/projections.py | 1-367 | Margin→prob, score derivation |
| Calibration application | src/pipelines/schedule.py | 430-573 | Distribution parameter transformation |
| Ensemble input collection | src/pipelines/schedule.py | 895-1050 | Gathers forecasts for combination |
| Ensemble ML | src/ensemble/ml_v1.py | 1-220 | Combines ML forecasts via logit pooling |
| Ensemble SPREAD | src/ensemble/spread_v1.py | 1-250 | Combines margin forecasts |
| Ensemble TOTAL | src/ensemble/total_v1.py | 1-250 | Combines total forecasts |
| BETS building | src/pipelines/schedule.py | 3850-4243 | Creates betting rows from ensemble |

---

## Summary & Recommendations

### Current System Complexity

The codebase has **three overlapping "head-like" layers**:
1. **Model layer** (Bradley-Terry, Poisson emit natively; others delegate)
2. **Projection engine** (derives all outputs for Elo, TOOR, GSSD)
3. **Calibration** (transforms distributions, not derives)

This creates **ambiguity** about where each output originates and **redundancy** (multiple code paths can produce the same output).

### Key Blockers for Formal Heads System

1. **Projection engine must be frozen** (no more derivation)
2. **Heads must emit all 5 fields** (no partial output)
3. **Clear ownership** per model per market (one source of truth)
4. **Fallback strategy** if head fails (error vs. skip vs. use default?)

### Immediate Next Steps

1. ✅ **Audit complete** (this document)
2. ⏭️ **Design heads interfaces** (what signature? per-market or unified?)
3. ⏭️ **Implement Bradley-Terry head** (self-contained; good candidate)
4. ⏭️ **Implement Poisson head** (self-contained; good candidate)
5. ⏭️ **Refactor Elo/TOOR/GSSD** (decide: projection engine stays or becomes heads?)
6. ⏭️ **Freeze projection engine** (after all heads implement)

---

**Document Version**: 1.0  
**Date**: 2026-01-28  
**Reviewer**: Awaiting review  
**Status**: COMPLETE - INVESTIGATION PHASE ONLY (NO CODE CHANGES)
