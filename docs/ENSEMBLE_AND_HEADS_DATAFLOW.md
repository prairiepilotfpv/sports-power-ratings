# Ensemble and Heads Dataflow Map

**Status**: Investigation & Documentation (No Code Changes)  
**Date**: 2025-01-28  
**Scope**: Understanding how ensembles and "heads" work end-to-end in the sports-power-ratings system.

---

## A) Definitions & Contracts

### Forecast Row
A **forecast row** is a per-game, per-model record containing:
- **Keys**: `game_id`, `date`, `home_team`, `away_team`, `model_name`
- **Required columns** (market-dependent):
  - **ML market**: `p_home_win`, `p_away_win` (probabilities)
  - **SPREAD market**: `margin_mean`, `margin_sd` (margin distribution params)
  - **TOTAL market**: `total_mean`, `total_sd` (total points distribution params)
- **Optional columns**: `projected_home_score`, `projected_away_score`, `win_prob_source`, metadata

**Forecast rows** originate from model `.predict()` calls and are aggregated into `market_forecast_rows` dict.

---

### Market
A **market** is one of three supported venues for forecasts:

| Market | Purpose | Outputs | Required Fields |
|--------|---------|---------|-----------------|
| **ML** | Moneyline (win probability) | `p_home_win`, `p_away_win` | `p_home_win` |
| **SPREAD** | Point margin distribution | `margin_mean`, `margin_sd` | Both required |
| **TOTAL** | Combined points distribution | `total_mean`, `total_sd` | Both required |

**Canonical enum**: `src/markets/base.py::Market` (ML, SPREAD, TOTAL).

---

### Source_id
A **source_id** (e.g., `win_prob_source`, `spread_source_id`, `total_source_id`) is a string identifier indicating:
1. **Base source**: Model name (e.g., `"elo"`, `"bradley-terry"`) or ensemble name (e.g., `"ensemble_ml_v1"`)
2. **Provenance tags** (optional, appended with `+`):
   - `"calibrated_ml"` — ML (moneyline) probability was calibrated
   - `"calibrated_spread"` — Spread (margin) distribution was calibrated
   - `"calibrated_total"` — Total points distribution was calibrated

**Examples**:
- `"elo"` (single model, no calibration)
- `"ensemble_ml_v1"` (ensemble, not yet calibrated)
- `"ensemble_ml_v1+calibrated_ml"` (ensemble + ML calibration)
- `"ensemble_ml_v1+calibrated_ml+calibrated_spread"` (ensemble + ML + SPREAD calibrations)

**Idempotency**: Tags are appended deterministically; calling append twice produces no duplicates.

---

### ensemble_applied
A **flag** (often implicit in code) indicating whether a multi-model ensemble was invoked for a market:
- `True`: Multiple models were combined via `MLWeightedAverageEnsemble`, `SpreadWeightedAverageEnsemble`, or `TotalWeightedAverageEnsemble`
- `False`: Single model was selected, or no ensemble output was produced

In the current system, "ensemble_applied" is checked implicitly by looking at `_ml_games_updated`, `_spread_games_updated`, `_total_games_updated` counters in [src/pipelines/schedule.py](src/pipelines/schedule.py#L3400-3450).

---

### Coverage
**Coverage** refers to completeness of forecast rows across models and games:
- **Per-model coverage**: Does model X have valid forecasts for all games in `market_forecast_rows[MARKET]`?
- **Per-game coverage**: Does game G have forecasts from all expected models?
- **Per-market coverage**: Are all required columns populated across all models for a given market?

**Coverage collapse** occurs when valid forecasts drop to zero or a subset of games/models due to:
- Missing forecast rows (model didn't generate output)
- NaN/invalid values in required columns
- Join/filter operations that exclude games
- Weight filtering removing all models for a market

---

## B) End-to-End Pipeline Trace

### Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│ SCHEDULE COMMAND ENTRY: run_schedule_pipeline()                    │
│ (src/pipelines/schedule.py::run_schedule_pipeline, ~line 2700)      │
└────────┬────────────────────────────────────────────────────────────┘
         │
         ├─ Load all games (all_games_df) from DB
         ├─ Filter to BETS time window (if market_csv provided)
         └─ Resolve ensemble config, weights, and market allowlists
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: Build per-model schedules                                   │
│ (src/pipelines/schedule.py::_build_market_forecasts_for_ensembles)  │
│ Lines ~895-1100                                                      │
└────────┬────────────────────────────────────────────────────────────┘
         │
         │ For each market (ML, SPREAD, TOTAL):
         │   For each allowed model for that market:
         │     - Fit model (if not incremental)
         │     - Call model.predict()
         │     - Apply per-model calibration
         │     - Collect rows into market_forecast_rows[MARKET]
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2: Resolve ensemble weights                                    │
│ (src/pipelines/schedule.py::_resolve_ensemble_weights, ~line 1300)  │
│ Output: weights dict, models list, weight_source label              │
└────────┬────────────────────────────────────────────────────────────┘
         │
         │ Query order (highest priority first):
         │   1. Tuning DB (if active tuning run exists)
         │   2. Selection DB (equal weights over selected models)
         │   3. Best backtest run in DB
         │   4. Active ensemble from DB
         │   5. File-based weights (outputs/ensembles/<market>/)
         │   6. Fallback: equal weights
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: Filter weights to available forecast models                 │
│ (src/pipelines/schedule.py::_filter_market_weights_for_forecast)    │
│ Lines ~1459-1650                                                     │
└────────┬────────────────────────────────────────────────────────────┘
         │
         │ For each model in candidate_weights:
         │   - Check if model has forecast rows
         │   - Check if model has required columns
         │   - Check if model has non-zero weight
         │   - Drop models that fail any check
         │ Re-normalize weights over remaining models
         │ If only 0-1 models left but >1 valid, fallback to uniform
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 4: Apply ensemble                                              │
│ (src/pipelines/schedule.py, 3 ensemble blocks: ML/SPREAD/TOTAL)     │
│ Lines ~3246-3800                                                     │
└────────┬────────────────────────────────────────────────────────────┘
         │
         │ For ML market:
         │   - Instantiate MLWeightedAverageEnsemble
         │   - For each game, call ensemble.combine(forecast_df)
         │   - Output: (p_home_win_raw, components_json)
         │   - Write to bets_schedule_df[game_idx][{home,away}_win_prob]
         │   - Write components_json to ml_ensemble_components_json
         │
         │ For SPREAD market:
         │   - Instantiate SpreadWeightedAverageEnsemble
         │   - For each game, call ensemble.combine(forecast_df)
         │   - Output: (margin_mean_raw, margin_sd_raw, components_json)
         │   - Write to bets_schedule_df[game_idx][margin_{mean,sd}]
         │
         │ For TOTAL market:
         │   - Instantiate TotalWeightedAverageEnsemble
         │   - For each game, call ensemble.combine(forecast_df)
         │   - Output: (total_mean_raw, total_sd_raw, components_json)
         │   - Write to bets_schedule_df[game_idx][total_{mean,sd}]
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 5: Apply calibration                                           │
│ (src/pipelines/schedule.py::_apply_calibration_to_schedule_df)      │
│ Lines ~333-600                                                       │
└────────┬────────────────────────────────────────────────────────────┘
         │
         │ For each market (ML, SPREAD, TOTAL):
         │   - Load latest calibrator (if exists)
         │   - Apply transform to raw predictions
         │   - Update {home,away}_win_prob / margin_{mean,sd} / total_{mean,sd}
         │   - Append "+calibrated_<market>" to win_prob_source
         │   - Log calibration summary
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 6: Build BETS dataframe                                        │
│ (src/pipelines/schedule.py::_build_bets_dataframe, ~line 1800)      │
│ Input: schedule_df (merged across all markets & models)             │
│ Output: bets_df (one row per team-side per game)                    │
└────────┬────────────────────────────────────────────────────────────┘
         │
         │ For each game (unique game_id):
         │   - Create row for home team ("selection"=home_team)
         │   - Create row for away team ("selection"=away_team)
         │   - Populate probabilities from merged schedule columns
         │   - Populate margin/total from merged schedule columns
         │   - Include ensemble_components_json for audit trail
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 7: Validate BETS contract                                      │
│ (src/pipelines/schedule.py::_validate_bets_ensemble_contract)       │
│ Lines ~1034-1080                                                     │
└────────┬────────────────────────────────────────────────────────────┘
         │
         │ Check that:
         │   - ML rows use valid sources (ensemble or single model)
         │   - SPREAD rows use ensemble_spread_v1 (if ensemble applies)
         │   - TOTAL rows use ensemble_total_v1 (if ensemble applies)
         │   - No "direct" or "direct+calibrated_*" fallbacks (not allowed)
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 8: Export workbook                                             │
│ (src/pipelines/schedule.py, write_workbook calls)                   │
│ Lines ~3850-3900                                                     │
└────────┬────────────────────────────────────────────────────────────┘
         │
         └─ SCHEDULE sheet: per-game forecasts + probabilities
            BETS sheet: deduped betting rows (2 per game)
            Calibration metadata sheet (if calibrated)
```

---

### Detailed Step Breakdown

#### **STEP 1: Per-Model Forecasts** (`_build_market_forecasts_for_ensembles`)

**Location**: [src/pipelines/schedule.py:895-1100](src/pipelines/schedule.py#L895)

**Inputs**:
- `schedule_df` (all upcoming games)
- `models` (list of model names)
- `allowed_models_by_market` (dict mapping market → list of model names, or None for all)
- `market_metrics` (dict mapping market → metric slot, e.g., "log_loss")
- `as_of_date`, `fit_end_date` (for data filtering)

**Process**:
```
For each market (ML, SPREAD, TOTAL):
  market_allowed = allowed_models_by_market.get(market.name) or models
  
  For each model in market_allowed:
    model_df = copy(schedule_df)
    schedule = _build_schedule_dataframe(
      model_df,
      db_path, sport, season, model,
      upcoming_only=True,
      model_params=resolve_effective_params(db, model, market),
      ...
    )
    
    # schedule now contains columns:
    # - date, game_id, home_team, away_team
    # - p_home_win, p_away_win  (if ML model)
    # - margin_mean, margin_sd  (if SPREAD model)
    # - total_mean, total_sd    (if TOTAL model)
    # - win_prob_source, model_p_home_win, ...
    
    For each row in schedule:
      market_forecast_rows[market.name].append({
        game_id, date, home_team, away_team, model_name,
        <market-specific columns>,
        ...
      })
```

**Key Functions Called**:
- `_build_schedule_dataframe()` ([line 291](src/pipelines/schedule.py#L291)): Calls `build_forecasts_df()`, then `_apply_calibration_to_schedule_df()`
- `build_forecasts_df()` (src/forecasting/forecast_service.py:1074): Fits model, calls `.predict()`, builds DataFrame
- Model `.predict()` methods (src/models/{elo,bradley_terry,gssd,poisson,toor}.py): Returns list of `GamePrediction` objects

**Output**: `market_forecast_rows` dict with structure:
```python
{
  "ML": [
    {"game_id": "...", "date": "...", "home_team": "...", "away_team": "...",
     "model_name": "elo", "p_home_win": 0.55, "p_away_win": 0.45, ...},
    {"game_id": "...", "model_name": "bradley-terry", "p_home_win": 0.52, ...},
    ...
  ],
  "SPREAD": [
    {"game_id": "...", "model_name": "elo", "margin_mean": 3.5, "margin_sd": 10.2, ...},
    ...
  ],
  "TOTAL": [
    {"game_id": "...", "model_name": "poisson", "total_mean": 210.5, "total_sd": 8.0, ...},
    ...
  ]
}
```

---

#### **STEP 2: Weight Resolution** (`_resolve_ensemble_weights`)

**Location**: [src/pipelines/schedule.py:1300-1450](src/pipelines/schedule.py#L1300)

**Inputs**:
- `market` (ML, SPREAD, or TOTAL)
- `ensemble_id` (e.g., "ensemble_ml_v1")
- `config_weights` (manual weights from ensemble config, or None)
- `selection_context`, `tuning_context` (from DB lookups)

**Resolution Order** (highest priority wins):
1. **Tuning DB**: If active tuning run found AND matches selection run ID → use tuned weights
2. **Selection Equal**: If active selection found (but tuning doesn't match) → uniform weights over selected models
3. **Best Backtest Run**: Query DB for best ensemble_market_tuning record by optimized metric
4. **DB Active**: Query `ensemble_market_weights` table for last activated run
5. **File-based**: Load from `outputs/ensembles/<sport>/<season>/<market>/<ensemble_id>.json` (via `load_market_weights()`)
6. **Fallback**: None (will be handled by `_filter_market_weights_for_forecast`)

**Output**: Tuple `(weights, models, weight_source, weight_run_id, selection_run_id)` where:
- `weights`: Dict like `{"elo": 0.5, "bradley-terry": 0.5}` or None
- `models`: List of model names the weights apply to
- `weight_source`: String label like "db_tuned", "file", "equal", etc.
- `weight_run_id`: Run ID from tuning or backtest, or None

---

#### **STEP 3: Weight Filtering** (`_filter_market_weights_for_forecast`)

**Location**: [src/pipelines/schedule.py:1459-1650](src/pipelines/schedule.py#L1459)

**Inputs**:
- `weights` (dict or None, from Step 2)
- `forecast_df` (DataFrame with all models' forecasts for this market)
- `market` (ML, SPREAD, or TOTAL)

**Process**:
1. **Build candidate weights**:
   - If `weights` provided: Use those keys, default 0.0 for missing models
   - If `weights` is None: Set all models to 1.0 (equal weights)

2. **Filter by column availability**:
   - Require: `p_home_win` for ML, `margin_mean` + `margin_sd` for SPREAD, `total_mean` + `total_sd` for TOTAL
   - Drop models missing required columns or having all NaN

3. **Filter by weight**:
   - Drop models with weight ≤ 0

4. **Fallback logic**:
   - If filtered count ≤ 1 AND valid models > 1: Use uniform weights over all valid models
   - If filtered count = 0: Use uniform weights over all valid models (or raise error)

5. **Normalize** weights to sum to 1.0

**Output**: Tuple `(normalized_weights, final_models)` where `final_models` is a set of model names to use

---

#### **STEP 4: Ensemble Application** (ML/SPREAD/TOTAL blocks)

**Location**: [src/pipelines/schedule.py:3246-3800](src/pipelines/schedule.py#L3246)

**ML Ensemble Example** ([lines 3246-3400](src/pipelines/schedule.py#L3246)):

```python
if bets_schedule_df is not None and ml_rows:
  forecast_df = pd.DataFrame(ml_rows)
  
  # Step 2: Resolve weights
  weights, ensemble_models, weight_source, ... = _resolve_ensemble_weights(...)
  
  # Step 3: Filter weights to available models
  filtered_weights, final_models = _filter_market_weights_for_forecast(
    weights=weights,
    forecast_df=forecast_df,
    market=Market.ML.name,
  )
  
  # Step 4: Apply ensemble
  use_ensemble = len(unique_ml_models) > 1
  
  if use_ensemble:
    ensemble = MLWeightedAverageEnsemble(
      sport, season,
      ensemble_id="ensemble_ml_v1",
      weights=filtered_weights,
    )
    
    # For each game, combine model forecasts
    for game_idx, game_row in bets_schedule_df.iterrows():
      game_id = game_row["game_id"]
      
      # Get all forecasts for this game
      game_forecast = forecast_df[forecast_df["game_id"] == game_id]
      
      if not game_forecast.empty:
        p_home, components_json = ensemble.combine(game_forecast)
        
        bets_schedule_df.loc[game_idx, "home_win_prob"] = p_home
        bets_schedule_df.loc[game_idx, "away_win_prob"] = 1.0 - p_home
        bets_schedule_df.loc[game_idx, "ml_ensemble_components_json"] = components_json
        bets_schedule_df.loc[game_idx, "win_prob_source"] = "ensemble_ml_v1"
```

**Key Ensemble Classes**:

| Class | File | Method | Inputs | Outputs |
|-------|------|--------|--------|---------|
| `MLWeightedAverageEnsemble` | [src/ensemble/ml_v1.py](src/ensemble/ml_v1.py) | `.combine()` | DataFrame(model_name, p_home_win) | (p_home_win_raw, components_json) |
| `SpreadWeightedAverageEnsemble` | [src/ensemble/spread_v1.py](src/ensemble/spread_v1.py) | `.combine()` | DataFrame(model_name, margin_mean, margin_sd) | (margin_mean_raw, margin_sd_raw, components_json) |
| `TotalWeightedAverageEnsemble` | [src/ensemble/total_v1.py](src/ensemble/total_v1.py) | `.combine()` | DataFrame(model_name, total_mean, total_sd) | (total_mean_raw, total_sd_raw, components_json) |

**ML Ensemble Algorithm** (logit pooling):
```python
def _logit_pool(probs, weights):
  z_sum = 0.0
  for p, w in zip(probs, weights):
    p_clipped = clip(p, eps, 1-eps)
    z_i = log(p_clipped / (1 - p_clipped))
    z_sum += w * z_i
  return 1.0 / (1.0 + exp(-z_sum))
```

**SPREAD Ensemble Algorithm** (weighted average + between-model variance):
```python
combined_mean = sum(w_i * margin_i)
within_var = sum(w_i * sigma_i^2)
between_var = sum(w_i * (margin_i - mean)^2)
combined_sd = sqrt(within_var + between_var)
```

---

#### **STEP 5: Calibration Application** (`_apply_calibration_to_schedule_df`)

**Location**: [src/pipelines/schedule.py:333-600](src/pipelines/schedule.py#L333)

**Process** (per market):

1. **Load Calibrator**:
   ```python
   calibrator = load_latest_calibrator(
     sport=sport, season=season, model=model, market="ML"  # or "spread", "total"
   )
   ```

2. **Apply Transform**:
   - **ML**: Platt scaling on `home_win_prob` → `home_win_prob_calibrated`
   - **SPREAD**: Distribution calibration on `(margin_mean, margin_sd)` → calibrated versions
   - **TOTAL**: Distribution calibration on `(total_mean, total_sd)` → calibrated versions

3. **Update In-Place**:
   ```python
   df.loc[valid_mask, "home_win_prob"] = df.loc[valid_mask, "home_win_prob_calibrated"]
   # (and similarly for other markets)
   ```

4. **Append Provenance Tag**:
   ```python
   if "calibrated_ml" not in win_prob_source:
     win_prob_source += "+calibrated_ml"
   ```

**Output**: Modified `schedule_df` with calibrated predictions and updated `win_prob_source` tags

---

#### **STEP 6: BETS DataFrame Construction** (`_build_bets_dataframe`)

**Location**: [src/pipelines/schedule.py:1800-2600](src/pipelines/schedule.py#L1800)

**Input**: `schedule_df` (merged across all models/markets from STEPS 1-5)

**Process**:
```python
bets_rows = []

for each unique game_id in schedule_df:
  game_data = schedule_df[schedule_df["game_id"] == game_id]
  
  # Row 1: Home team perspective
  bets_rows.append({
    "selection": game_data["home_team"],
    "opponent": game_data["away_team"],
    "date": game_data["date"],
    "home_win_prob": game_data["home_win_prob"],
    "away_win_prob": game_data["away_win_prob"],
    "margin_mean": game_data["margin_mean"],  # Positive = home favored
    "margin_sd": game_data["margin_sd"],
    "total": game_data["total"],
    "total_sd": game_data["total_sd"],
    "win_prob_source": game_data["win_prob_source"],
    "ml_ensemble_components_json": game_data["ml_ensemble_components_json"],
    "spread_ensemble_components_json": game_data["spread_ensemble_components_json"],
    "total_ensemble_components_json": game_data["total_ensemble_components_json"],
    ...
  })
  
  # Row 2: Away team perspective (opponent's home_win_prob)
  bets_rows.append({
    "selection": game_data["away_team"],
    "opponent": game_data["home_team"],
    "date": game_data["date"],
    "home_win_prob": game_data["away_win_prob"],  # Flipped!
    "away_win_prob": game_data["home_win_prob"],
    ...
  })

bets_df = pd.DataFrame(bets_rows)
```

**Key Deduplication**: If input `schedule_df` has duplicate `game_id`s (from multiple runs), keep only first occurrence.

---

#### **STEP 7: Contract Validation** (`_validate_bets_ensemble_contract`)

**Location**: [src/pipelines/schedule.py:1034-1080](src/pipelines/schedule.py#L1034)

**Checks**:
1. **ML rows**: `win_prob_source` must match `ensemble_ml_v1` OR single model (NOT "direct" or "direct+calibrated_*")
2. **SPREAD rows**: `win_prob_source` must contain `ensemble_spread_v1` (no fallback)
3. **TOTAL rows**: `win_prob_source` must contain `ensemble_total_v1` (no fallback)

**Violation Handling**: Raises `RuntimeError` with detailed list of violating row IDs and expected contracts.

---

#### **STEP 8: Export** (Workbook Writing)

**Locations**:
- [src/pipelines/schedule.py:3850-3900](src/pipelines/schedule.py#L3850) (main schedule block)
- [src/data/reporting.py](src/data/reporting.py) (Excel formatting)

**Output Files**:
- `SCHEDULE` sheet: All games + all models' forecasts (wide format)
- `BETS` sheet: One row per (game, selection), with ensemble+calibration data
- Metadata sheet: Config, weights source, calibration summaries

---

## C) Coverage Collapse Diagnostics

### Root Causes & Detection

#### 1. **Missing Forecast Rows (Per-Model)**

**Condition**: A model doesn't generate forecast rows for some or all games.

**Root Causes**:
- Model fit failed (invalid parameters, no valid training data)
- Model `.predict()` returned empty list
- Database lookup for model params failed
- Game date outside model's historical window

**Code Location**: [src/pipelines/schedule.py:895-1100](src/pipelines/schedule.py#L895) in `_build_market_forecasts_for_ensembles`

**Detection**:
```python
ml_rows = market_forecast_rows.get("ML", [])
ml_forecast_df = pd.DataFrame(ml_rows)
unique_models = set(ml_forecast_df["model_name"].unique())
expected_models = {"elo", "bradley-terry"}

missing = expected_models - unique_models
if missing:
  logger.warning(f"Missing forecast rows for models: {missing}")
```

---

#### 2. **Column Validation Failures**

**Condition**: Model forecast rows exist but lack required columns for a market.

**Root Causes**:
- Model doesn't support a market (e.g., Elo doesn't support TOTAL)
- Projection engine failed to derive secondary columns (margin from score, total from ML, etc.)
- Calibration step set columns to NaN

**Code Location**: [src/pipelines/schedule.py:1459-1650](src/pipelines/schedule.py#L1459) in `_filter_market_weights_for_forecast`

**Detection**:
```python
required_cols = ("margin_mean", "margin_sd")  # For SPREAD
for model, group in forecast_df.groupby("model_name"):
  missing = [c for c in required_cols if c not in group.columns or not group[c].notna().any()]
  if missing:
    logger.error(f"Model {model} missing {missing} for SPREAD market")
```

---

#### 3. **Join/Filter Key Mismatch**

**Condition**: Ensemble rows can't be joined to BETS games due to game_id/date/team mismatch.

**Root Causes**:
- BETS schedule built from narrower time window (e.g., market CSV) than ensemble forecasts
- Game IDs differ (fallback ID generation vs. imported IDs)
- Team names normalized inconsistently

**Code Location**: 
- BETS window filter: [src/pipelines/schedule.py:2790-2850](src/pipelines/schedule.py#L2790)
- Ensemble application: [src/pipelines/schedule.py:3330-3360](src/pipelines/schedule.py#L3330) (game_idx lookup)

**Detection**:
```python
ml_rows_df = pd.DataFrame(market_forecast_rows["ML"])
bets_ids = set(bets_schedule_df["game_id"].unique())
ml_ids = set(ml_rows_df["game_id"].unique())

orphaned = ml_ids - bets_ids
if orphaned:
  logger.warning(f"Ensemble forecasts for {len(orphaned)} games not in BETS window")
```

---

#### 4. **NaN Filtering & Guardrails**

**Condition**: Valid forecast rows are filtered out during calibration or guardrail application.

**Root Causes**:
- Calibrator returns NaN for some rows
- SD guardrail clips values to minimum threshold
- Win probability guardrail rejects extreme calibrated values

**Code Location**: [src/pipelines/schedule.py:500-600](src/pipelines/schedule.py#L500) (SPREAD/TOTAL calibration)

**Detection**:
```python
before = forecast_df["margin_sd"].notna().sum()
# ... apply calibration ...
after = forecast_df["margin_sd"].notna().sum()

if after < before:
  logger.warning(f"Calibration dropped {before - after} margin_sd values")
```

---

#### 5. **Weight Filtering Dropping Models**

**Condition**: All or most models dropped during `_filter_market_weights_for_forecast`, falling back to uniform weights or single model.

**Root Causes**:
- Tuned weights have zero weight for most models
- Models don't have required forecast columns (covered in #2)
- Models have NaN predictions
- Ensemble config specifies narrow model list for a market

**Code Location**: [src/pipelines/schedule.py:1459-1650](src/pipelines/schedule.py#L1459), especially lines ~1510-1575

**Detection**:
```python
dropped = set(candidate_weights.keys()) - set(filtered_weights.keys())
if dropped:
  logger.info(f"[_filter_market_weights_for_forecast] Dropped {len(dropped)} models for {market}: {dropped}")

if len(filtered_weights) <= 1 and len(valid_models) > 1:
  logger.info(f"Collapsed from {len(valid_models)} valid to {len(filtered_weights)} after filtering; using uniform fallback")
```

---

#### 6. **Schedule Window Mismatch (BETS vs Forecasts)**

**Condition**: BETS games are from a narrower window than ensemble forecast window, causing orphaned forecasts.

**Scenario**: 
- Run 1: `--market-csv jan15.csv` with 30 games → BETS has 30 games
- Ensemble built from 100 games in season DB → Forecasts exist for all 100
- 70 games outside BETS window have no BETS rows

**Code Location**: [src/pipelines/schedule.py:2790-2850](src/pipelines/schedule.py#L2790) (BETS window logic)

**Detection**:
```python
if bets_schedule_df is not None:
  bets_games = set(bets_schedule_df["game_id"].unique())
  ml_forecast_games = set(ml_rows_df["game_id"].unique())
  
  orphaned = ml_forecast_games - bets_games
  if orphaned:
    logger.warning(f"Ensemble forecasts exist for {len(orphaned)} games outside BETS window")
```

---

#### 7. **Model Doesn't Support Market**

**Condition**: Model metadata indicates `supports_margin=False` or `supports_total=False`, but model is assigned to that market.

**Example**: Elo model with `supports_total=False` assigned to TOTAL ensemble.

**Code Location**: [src/models/{elo,bradley_terry}.py](src/models/elo.py) (metadata definition)

**Detection**:
```python
model_meta = model.metadata()
market_required = ("margin_mean", "margin_sd") if market == "SPREAD" else ("total_mean", "total_sd")

if market == "SPREAD" and not model_meta.supports_margin:
  logger.error(f"Model {model.model_id} doesn't support margin but assigned to SPREAD ensemble")

if market == "TOTAL" and not model_meta.supports_total:
  logger.error(f"Model {model.model_id} doesn't support total but assigned to TOTAL ensemble")
```

---

### **"ML Becomes BT-Only" Scenario**

This is the most common coverage collapse case. Here's why it happens:

1. **Initial State**: Ensemble configured with `["elo", "bradley-terry"]` for ML market
2. **Per-model forecasts generated**: Both models produce `p_home_win` for all games
3. **Weight resolution**: Tuned weights `{"elo": 0.5, "bradley-terry": 0.5}` loaded from DB
4. **Weight filtering**: 
   - Elo forecasts have NaN `p_home_win` for 20 games (data issue?)
   - Elo is dropped from filtered weights
   - Only Bradley-Terry remains
5. **Ensemble application**:
   - `use_ensemble = len(final_models) > 1` → False (only 1 model)
   - Single-model logic kicks in: just pass through Bradley-Terry values
   - `win_prob_source = "bradley-terry"` (NOT ensemble)

**Affected Code**: [src/pipelines/schedule.py:3290-3310](src/pipelines/schedule.py#L3290)

```python
unique_ml_models = set(forecast_df.get("model_name", []).dropna().unique())
use_ensemble = len(unique_ml_models) > 1

if use_ensemble:
  # Apply ensemble
  ...
else:
  # Single model or no ensemble
  logger.info(f"Skipping ML ensemble: only {len(unique_ml_models)} model(s)")
  # Pass through single model's values directly
```

**Prevention**: Add strict assertion before filtering:
```python
assert len(unique_ml_models) > 1, (
  f"ML ensemble misconfigured: {len(unique_ml_models)} model(s) available. "
  f"Expected >= 2 for ensemble_ml_v1. Available: {unique_ml_models}. "
  f"Forecast counts: {forecast_df['model_name'].value_counts().to_dict()}"
)
```

---

## D) Ensemble Layer Inventory

### MLWeightedAverageEnsemble

**File**: [src/ensemble/ml_v1.py](src/ensemble/ml_v1.py)

**Purpose**: Combine multiple model moneyline probabilities via logit pooling.

**Inputs**:
- `sport`, `season`: For loading weight file
- `ensemble_id`: Default `"ensemble_ml_v1"`
- `weights`: Dict `{model_name: float}` or None (load from file)

**Method Signature**:
```python
def combine(forecast_df: pd.DataFrame) -> tuple[float | None, str]:
  """
  Args:
    forecast_df: DataFrame with columns [model_name, p_home_win]
  
  Returns:
    (p_home_win_raw, components_json)
  """
```

**Algorithm**: Logit (log-odds) pooling:
```
z_i = log(p_i / (1 - p_i)) for each model i
z_combined = sum(w_i * z_i)
p_combined = 1 / (1 + exp(-z_combined))
```

**Output Columns**:
- `p_home_win`: Combined probability (raw, before calibration)
- `p_away_win`: 1 - p_home_win

**Components JSON** (stored as string, parsed back when needed):
```json
[
  {
    "model": "elo",
    "weight": 0.5,
    "value": 0.55,
    "uncertainty": null
  },
  {
    "model": "bradley-terry",
    "weight": 0.5,
    "value": 0.52,
    "uncertainty": null
  }
]
```

**Special Handling**:
- If a model has `p_home_win = NaN`: Marked as invalid, excluded from pooling
- If all models are invalid: Returns `(None, "[]")`
- If model weights are 0: Warned but not applied
- Deterministic weight normalization over valid models

---

### SpreadWeightedAverageEnsemble

**File**: [src/ensemble/spread_v1.py](src/ensemble/spread_v1.py)

**Purpose**: Combine multiple model margin forecasts with between-model variance.

**Inputs**:
- `sport`, `season`: For loading weight file
- `ensemble_id`: Default `"ensemble_spread_v1"`
- `weights`: Dict or None
- `include_between_model_variance`: Boolean (default True)

**Method Signature**:
```python
def combine(game_rows: pd.DataFrame) -> tuple[float | None, float | None, str]:
  """
  Args:
    game_rows: DataFrame with columns [model_name, margin_mean, margin_sd]
  
  Returns:
    (margin_mean_raw, margin_sd_raw, components_json)
  """
```

**Algorithm**: Weighted average + variance composition:
```
combined_mean = sum(w_i * margin_mean_i)

within_var = sum(w_i * margin_sd_i^2)
between_var = sum(w_i * (margin_mean_i - combined_mean)^2)

combined_sd = sqrt(within_var + between_var) if include_between_model_variance
            = sqrt(within_var) otherwise
```

**Output Columns**:
- `margin_mean`: Combined margin prediction (home - away, positive = home favored)
- `margin_sd`: Combined margin standard deviation

**Components JSON**:
```json
[
  {
    "model": "elo",
    "weight": 0.5,
    "value": 3.5,
    "uncertainty": 10.2
  }
]
```

**Special Handling**:
- If `margin_sd ≤ 0 or not finite`: Marked invalid, excluded
- Between-model variance adds diversity penalty (wider distribution if models disagree)

---

### TotalWeightedAverageEnsemble

**File**: [src/ensemble/total_v1.py](src/ensemble/total_v1.py)

**Purpose**: Combine multiple model total (combined points) forecasts.

**Inputs**:
- `sport`, `season`: For loading weight file
- `ensemble_id`: Default `"ensemble_total_v1"`
- `weights`: Dict or None
- `include_between_model_variance`: Boolean (default True)

**Method Signature**:
```python
def combine(game_rows: pd.DataFrame) -> tuple[float | None, float | None, str]:
  """
  Args:
    game_rows: DataFrame with columns [model_name, total_mean, total_sd]
              (or total_mean aliases as total)
  
  Returns:
    (total_mean_raw, total_sd_raw, components_json)
  """
```

**Algorithm**: Identical to SPREAD, but operates on total points.

**Special Handling**:
- Falls back to **uniform weights** if configured weights have no positive values (line ~85)
  ```python
  configured_weights_positive = any(float(v) > 0.0 for v in self._weights.values())
  if not configured_weights_positive:
    raw_weights = [1.0 for _ in models]
  ```
- This is intentional: TOTAL ensemble should not fail due to missing weights; uniform is safer fallback

---

## E) Open Questions & Known Gaps

### Bradley–Terry "Heads" Design

**Question**: How should Bradley–Terry produce `margin_mean` and `margin_sd` when it only has team strength ratings?

**Current Situation**: 
- Bradley-Terry fits ratings (strength differences)
- Can compute margin via sigmoid or linear transform
- But lacks inherent margin distribution (SD)
- Current workaround: Use calibration to back out SD from training data, OR use fallback constant SD

**Possible Approaches** (no current implementation):

1. **Thurstone–Mosteller / Probit-Noise Head**:
   - Assume margin `~ Normal(μ, σ)` where μ = strength difference
   - Jointly fit σ (noise term) via MLE
   - Produces native `(margin_mean, margin_sd)` suitable for SPREAD ensemble
   - **Pros**: Theoretically sound, principled
   - **Cons**: Requires re-fitting; adds hyperparameter (σ)

2. **Probability-to-Margin Inversion**:
   - Use Bradley-Terry `p_win` + assumed σ to back-solve margin
   - `margin = Φ^{-1}(p_win) * σ + constant`
   - Uses fixed or learned σ
   - **Pros**: Reuses existing ratings
   - **Cons**: Assumes Normal distribution; may not match actual margin distribution

3. **Joint Likelihood** (Win + Margin):
   - Extend Bradley-Terry to fit both win outcomes AND margin values simultaneously
   - Produces ratings + σ in one step
   - **Pros**: Captures both signals
   - **Cons**: More complex optimization; may overfit to margin data

**Where This Would Live**: [src/models/bradley_terry_heads.py](src/models/bradley_terry_heads.py) (not yet created)

---

### Projection Engine Coverage

**Question**: Which models produce which outputs natively vs. derived?

**Current Status**:
- **Elo**: Produces `p_win`, `margin_mean` (via calibration), NO `total`
- **Bradley-Terry**: Produces `p_win`, `margin_mean` + `margin_sd` (via calibration), NO `total`
- **GSSD**: Produces `margin_mean`, `margin_sd`, and can produce `total_mean`, `total_sd`
- **Poisson**: Produces `total_mean`, can be converted to `p_win` and margin
- **TOOR**: Produces `margin_mean`, `margin_sd`, and optionally `total`

**Projection Engine** ([src/pipelines/projections.py](src/pipelines/projections.py)) fills gaps:
- ML-to-Margin: If model produces `p_win` but not margin, use Normal CDF inversion
- Margin-to-Total: If margin + total_sd known, compute total = margin + (home_score + away_score)

**Question**: Should "heads" be created per model to standardize this, or is projection engine sufficient?

---

### Ensemble Weights Persistence

**Question**: When/where are ensemble weights persisted after tuning?

**Current Locations**:
1. **DB**: `ensemble_market_tuning` table stores run results + weights
2. **DB**: `ensemble_market_actives` table stores active run pointer
3. **File**: `outputs/ensembles/<sport>/<season>/<market>/<ensemble_id>.json` (manual or test-driven)

**Question**: Should weights auto-export to file after successful tuning, or remain DB-only?

---

## F) Single "Next Action" List

Based on this investigation, here are concrete fixes (in priority order):

### 1. **Assert Multi-Model Ensemble Before Collapse** (HIGH)
- **What**: Add assertion that catches ML/SPREAD/TOTAL collapsing to single model before it happens
- **Where**: [src/pipelines/schedule.py:3290-3310](src/pipelines/schedule.py#L3290) (before ensemble decision)
- **Why**: Current behavior silently degrades to BT-only when Elo has NaNs; assertion makes this loud
- **Test**: `test_bets_ensemble_gating.py` (modify to verify assertion fires when appropriate)

### 2. **Document Model Support Matrix** (MEDIUM)
- **What**: Create table mapping each model → (supports_ml, supports_spread, supports_total, method)
- **Where**: New file [docs/MODEL_SUPPORT_MATRIX.md](docs/MODEL_SUPPORT_MATRIX.md)
- **Why**: Clarifies valid ensemble configs; prevents misconfiguring unsupported models to markets
- **Test**: Unit test that model metadata matches documented support

### 3. **Calibration Failure Transparency** (MEDIUM)
- **What**: When calibrator fails or doesn't exist, log with explicit "using uncalibrated" at INFO level (not DEBUG)
- **Where**: [src/pipelines/schedule.py:500-600](src/pipelines/schedule.py#L500) (each market calibration block)
- **Why**: Currently INFO level only on success; failures buried in DEBUG, hard to diagnose
- **Test**: Verify log contains "using uncalibrated" when calibrator missing

### 4. **Weight Filtering Fallback Audit** (MEDIUM)
- **What**: Log warning when `_filter_market_weights_for_forecast` falls back to uniform (i.e., tuned weights collapsed)
- **Where**: [src/pipelines/schedule.py:1528-1535](src/pipelines/schedule.py#L1528)
- **Why**: Operators need visibility when tuned weights don't apply; affects performance
- **Test**: `test_bets_ensemble_gating.py` already verifies this

### 5. **Game-Level Ensemble Component Audit** (MEDIUM)
- **What**: For each game with ensemble, log model count + weights (sample first 5 games)
- **Where**: [src/pipelines/schedule.py:3330-3360](src/pipelines/schedule.py#L3330) (after ensemble.combine loop)
- **Why**: Helps verify ensemble is actually using multiple models per game
- **Test**: Parse log output; verify each game's components JSON has ≥2 models with non-zero weight

### 6. **BETS Sheet Ensemble Column Inventory** (LOW)
- **What**: Document which columns in BETS sheet contain ensemble outputs (for downstream consumers)
- **Where**: Update [docs/CLI.md](docs/CLI.md) (BETS sheet section)
- **Why**: Current docs don't clarify that `ml_ensemble_components_json`, `spread_ensemble_components_json`, etc. are primary audit trail
- **Test**: No code change; documentation only

### 7. **Bradley-Terry Margin Head** (LOW)
- **What**: Design & spec how BT should produce margin_sd (do not implement; just document)
- **Where**: New file [docs/BRADLEY_TERRY_HEADS.md](docs/BRADLEY_TERRY_HEADS.md) or expand existing [docs/bradley_terry_heads.md](docs/bradley_terry_heads.md)
- **Why**: Currently BT produces margin only via calibration; direct approach would be cleaner
- **Test**: No code change; design doc only

### 8. **Ensemble config default per-market** (LOW)
- **What**: Ensure every sport/season pair has fallback ensemble configs in [src/ensemble/default_configs/](src/ensemble/default_configs/)
- **Where**: Check/create `ML.json`, `SPREAD.json`, `TOTAL.json` in that dir
- **Why**: Current fallback to hardcoded model list; file-based config more transparent
- **Test**: Unit test verifying all 3 markets have default config present

---

## Summary

The ensemble and heads architecture is a **three-layer system**:

1. **Per-Model Forecasts** ([src/pipelines/schedule.py:895](src/pipelines/schedule.py#L895)): Each model produces native outputs via `.predict()`, aggregated into `market_forecast_rows`

2. **Ensemble Combination** ([src/ensemble/{ml_v1,spread_v1,total_v1}.py](src/ensemble/)): Multi-model forecasts combined using market-specific algorithms (logit pooling for ML, weighted average for SPREAD/TOTAL)

3. **Calibration & Provenance** ([src/pipelines/schedule.py:333](src/pipelines/schedule.py#L333)): Calibrators adjust raw ensemble outputs; tags appended to track which markets were calibrated

**Coverage collapses** when forecast rows are missing (per-model failures), columns invalid (projection engine failures), weights zero (tuning artifact), or joins fail (window mismatch).

**Heads** (e.g., Bradley-Terry margin head) are currently missing; they should derive missing market outputs from a base model's native representation.

