# Ensemble System Architecture

## Overview

The ensemble system combines predictions from multiple power rating models to produce more accurate and robust forecasts across three market types: ML (moneyline), SPREAD, and TOTAL. This document describes the architecture, standardized schema, weight resolution, and variance calculation methods.

## Core Principles

1. **Market-Specific Implementations**: Each market type (ML, SPREAD, TOTAL) has its own ensemble class optimized for that market's characteristics
2. **Weighted Averaging**: Model predictions are combined using configurable weights that sum to 1.0
3. **Uncertainty Quantification**: SPREAD and TOTAL ensembles incorporate both within-model and between-model variance
4. **Deterministic Output**: Component ordering and weight application are deterministic for reproducibility

## Standardized JSON Schema

All three market types now use a uniform component schema:

```json
{
  "model": "string",           // Model name (normalized)
  "weight": 0.0-1.0,           // Normalized weight in ensemble
  "value": float|null,         // Market-specific prediction value
  "uncertainty": float|null    // Market-specific uncertainty (optional)
}
```

### Market-Specific Value Mappings

| Market | `value` Field | `uncertainty` Field |
|--------|---------------|---------------------|
| ML     | Probability (0.0-1.0) | `null` (not applicable) |
| SPREAD | Projected margin (home - away) | Margin standard deviation |
| TOTAL  | Projected total points | Total standard deviation |

### Example Components

**ML Market:**
```json
[
  {"model": "elo", "weight": 0.5, "value": 0.6, "uncertainty": null},
  {"model": "gssd", "weight": 0.5, "value": 0.55, "uncertainty": null}
]
```

**SPREAD Market:**
```json
[
  {"model": "bradley-terry", "weight": 0.4, "value": 3.5, "uncertainty": 12.0},
  {"model": "toor", "weight": 0.6, "value": 4.0, "uncertainty": 11.5}
]
```

**TOTAL Market:**
```json
[
  {"model": "poisson", "weight": 0.5, "value": 215.5, "uncertainty": 18.0},
  {"model": "toor", "weight": 0.5, "value": 217.0, "uncertainty": 16.5}
]
```

## Weight Resolution Priority

Ensemble weights are resolved in the following order (highest priority first):

1. **Override config** - Passed directly to ensemble constructor via `weights` parameter
2. **Custom per-market path** - `outputs/ensembles/<sport>/<season>/<market>/<ensemble_id>.json`
3. **Season default** - `outputs/ensembles/<sport>/<season>/<market>/default.json`
4. **Legacy path** - `outputs/ensembles/<sport>/<season>/ensemble_config.json`
5. **Global default** - `src/ensemble/default_configs/<market>.json`
6. **Equal weights fallback** - All models weighted equally (1.0 each, then normalized)

### Important Behaviors

- **Missing models default to 0.0 weight** and are excluded from the ensemble with a warning
- **Explicit zero weights are preserved** during normalization
- **Model names are normalized** to lowercase for consistent lookups across configs
- **Weights are automatically renormalized** to sum to 1.0 for valid predictions

## Ensemble Implementations

### ML Weighted Average Ensemble

**Class:** `MLWeightedAverageEnsemble`

**Input:** DataFrame with columns `model_name`, `p_home_win`

**Output:** `(p_home_win_combined, components_json)`

**Algorithm:**
1. Normalize model names
2. Load/apply weights (default 0.0 for missing models)
3. Filter to valid probabilities (not NaN)
4. Renormalize weights over valid models
5. Compute weighted average: `p_combined = Σ(weight_i × prob_i)`
6. Sort components by model name
7. Return combined probability and JSON

**Key Features:**
- No uncertainty quantification (ML predictions are point estimates)
- Invalid probabilities receive 0.0 weight
- Components JSON sorted alphabetically by model

### SPREAD Weighted Average Ensemble

**Class:** `SpreadWeightedAverageEnsemble`

**Input:** DataFrame with columns `model_name`, `margin_mean`, `margin_sd`

**Output:** `(margin_mean_combined, margin_sd_combined, components_json)`

**Algorithm:**
1. Normalize model names
2. Load/apply weights (default 0.0 for missing models)
3. Filter to valid margins (not NaN)
4. Renormalize weights over valid models
5. Compute weighted mean: `margin_combined = Σ(weight_i × margin_i)`
6. Compute within-model variance (weighted RMS of SDs)
7. Optionally add between-model variance
8. Return combined margin, combined SD, and JSON

**Within-Model Variance:**
```python
within_var = Σ(weight_i × sd_i²)  # for models with valid SD
margin_sd_combined = sqrt(within_var)
```

**Between-Model Variance (optional):**
```python
# Use SAME weights as mean calculation (no re-normalization)
between_var = Σ(weight_i × (margin_i - margin_combined)²)
margin_sd_combined = sqrt(within_var + between_var)
```

**Key Features:**
- Between-model variance **uses already-normalized weights** (Issue #1 fix)
- Between-variance only applied when ≥2 models have both mean and SD
- Debug logging shows when between-variance is applied/skipped
- Uncertainty increases when models disagree

### TOTAL Weighted Average Ensemble

**Class:** `TotalWeightedAverageEnsemble`

**Input:** DataFrame with columns `model_name`, `total_mean` (or `total`), `total_sd`

**Output:** `(total_mean_combined, total_sd_combined, components_json)`

**Algorithm:** Identical to SPREAD ensemble, but operating on total points instead of margins.

**Key Features:**
- Same variance calculation approach as SPREAD
- Supports both `total_mean` and `total` column names for backward compatibility

## Critical Bug Fixes

### Issue #1: Between-Model Variance Re-normalization Bug (FIXED)

**Problem:** Weights were re-normalized when computing between-model variance, causing the variance calculation to use different weights than the mean calculation.

**Impact:**
- Tuned ensemble weights were not preserved through variance calculation
- Between-variance contribution was incorrectly amplified
- Predictions became unstable when models had missing variance values

**Fix:** Use the already-normalized weights directly from components:
```python
# BEFORE (buggy):
weight_sum = sum(comp["w"] for comp in var_set)
normalized_weights = [comp["w"] / weight_sum for comp in var_set]  # Re-normalizes!

# AFTER (fixed):
weights_for_var = [comp["weight"] for comp in var_set]  # Already normalized
mean_for_var_set = combined_mean  # Use already-computed weighted mean
```

### Issue #2: ML Default Weight Bug (FIXED)

**Problem:** Missing models defaulted to weight 1.0, creating unexpected equal-weighting behavior.

**Impact:**
- Unknown models were silently included in ensemble
- Renaming models without updating config caused incorrect predictions
- No warning that models were being treated as equally-weighted

**Fix:** Default to 0.0 and log warning:
```python
# BEFORE:
raw_weights = [float(self._weights.get(m, 1.0)) for m in models]

# AFTER:
raw_weights = [float(self._weights.get(m, 0.0)) for m in models]
zero_weight_models = [m for m, w in zip(models, raw_weights) if w == 0.0]
if zero_weight_models:
    logger.warning("Models excluded (zero weight): %s", zero_weight_models)
```

### Issue #3: JSON Schema Inconsistency (FIXED)

**Problem:** Each market type used different JSON keys, breaking validation and making code brittle.

**Impact:**
- Validation code hardcoded key lookups for specific markets
- Adding new markets required updating multiple validation functions
- Equal-weight baseline reconstruction failed for SPREAD/TOTAL

**Fix:** Standardized schema with `create_component()` helper:
```python
# All markets now use:
comp = create_component(
    model=model_name,
    weight=normalized_weight,
    value=prediction_value,
    uncertainty=optional_sd
)
```

### Issue #5: SPREAD/TOTAL Components Lost (FIXED)

**Problem:** Only ML ensemble components were saved to database; SPREAD/TOTAL components lost after BETS sheet closed.

**Impact:**
- Impossible to validate tuned ensemble weights for SPREAD/TOTAL
- Couldn't reconstruct equal-weight baselines for comparison
- No audit trail for which models were ensembled in historical predictions

**Fix:** Added database migration and updated repository:
```sql
ALTER TABLE bets_predictions ADD COLUMN spread_ensemble_components_json TEXT;
ALTER TABLE bets_predictions ADD COLUMN total_ensemble_components_json TEXT;
```

### Issue #10: Model Name Normalization Inconsistency (FIXED)

**Problem:** Model names normalized in ensemble config but not consistently when loading weights.

**Impact:**
- "Bradley-Terry" in config might not match "bradley_terry" in weights file
- Ensemble lookups failed due to case/punctuation mismatches
- Silent fallback to 0.0 weight when names didn't match exactly

**Fix:** Normalize all model names through `normalize_model_name()`:
```python
from models.registry import normalize_model_name

# In ensemble files:
model = normalize_model_name(str(getattr(r, "model_name")))

# In io.py:
weights = {normalize_model_name(k): float(v) for k, v in raw_weights.items()}
```

## Database Schema

### bets_predictions Table

Ensemble component JSON is persisted in three columns:

| Column | Type | Description |
|--------|------|-------------|
| `ml_ensemble_components_json` | TEXT | ML market component JSON |
| `spread_ensemble_components_json` | TEXT | SPREAD market component JSON |
| `total_ensemble_components_json` | TEXT | TOTAL market component JSON |

This enables:
- **Validation** of tuned weights vs. equal-weight baselines
- **Auditing** of which models contributed to historical predictions
- **Debugging** of ensemble behavior over time

## Logging and Diagnostics

### Warning Levels

**WARNING:**
- Models excluded due to zero weight
- Failed to parse ensemble weights JSON
- Weight config renormalized (sum ≠ 1.0)

**DEBUG:**
- Between-model variance applied (shows within, between, total SD)
- Between-variance skipped (only 1 model with SD)
- Config resolution attempts and fallbacks

**ERROR:**
- Unexpected error loading ensemble weights
- Database migration failures

### Example Logs

```
WARNING  ensemble.ml_v1:ml_v1.py:97 Models excluded from ensemble (zero weight): ['bradley-terry', 'elo']. To include these models, add weights to config for sport=nba, season=2025-26

DEBUG  ensemble.spread_v1:spread_v1.py:184 SPREAD ensemble: Applied between-model variance (within=140.25, between=8.50, total_sd=12.20) for 3 models with uncertainty

DEBUG  ensemble.spread_v1:spread_v1.py:194 SPREAD ensemble: Between-model variance skipped (only 1 model with SD: gssd). Need ≥2 models with both value and uncertainty.
```

## Usage Examples

### Creating an Ensemble with Custom Weights

```python
from ensemble.ml_v1 import MLWeightedAverageEnsemble

# Provide explicit weights
ensemble = MLWeightedAverageEnsemble(
    sport="nba",
    season="2025-26",
    weights={"elo": 0.6, "gssd": 0.4}
)

# Combine predictions
import pandas as pd
forecasts = pd.DataFrame([
    {"model_name": "elo", "p_home_win": 0.65},
    {"model_name": "gssd", "p_home_win": 0.58}
])
prob, components = ensemble.combine(forecasts)
# prob = 0.62, components = JSON with standardized schema
```

### Loading Weights from Config

```python
from ensemble.spread_v1 import SpreadWeightedAverageEnsemble

# Weights loaded from outputs/ensembles/nba/2025-26/SPREAD/ensemble_spread_v1.json
ensemble = SpreadWeightedAverageEnsemble(
    sport="nba",
    season="2025-26"
)

forecasts = pd.DataFrame([
    {"model_name": "bradley-terry", "margin_mean": 3.5, "margin_sd": 12.0},
    {"model_name": "toor", "margin_mean": 4.0, "margin_sd": 11.5}
])
margin, sd, components = ensemble.combine(forecasts)
```

### Disabling Between-Model Variance

```python
from ensemble.total_v1 import TotalWeightedAverageEnsemble

# Only use within-model variance
ensemble = TotalWeightedAverageEnsemble(
    sport="nba",
    season="2025-26",
    include_between_model_variance=False
)
```

## Best Practices

### Weight Configuration

1. **Always provide explicit weights** for production ensembles
2. **Store weights in market-specific paths** for clarity
3. **Version control weight configs** to track changes over time
4. **Use tuning pipeline** to optimize weights based on historical performance

### Model Management

1. **Normalize model names consistently** (lowercase, no spaces)
2. **Set explicit zero weights** for models you want to exclude
3. **Monitor warning logs** for excluded models
4. **Update weight configs** when adding/removing models from ensemble

### Variance Handling

1. **Enable between-model variance** for SPREAD/TOTAL by default
2. **Ensure models provide uncertainty estimates** (margin_sd, total_sd)
3. **Monitor debug logs** to see when variance is applied/skipped
4. **Validate uncertainty ranges** are reasonable (use guardrails)

### Testing

1. **Provide explicit weights in tests** (don't rely on defaults)
2. **Test with missing values** (NaN probabilities, missing SDs)
3. **Verify component JSON** has correct schema
4. **Check weights sum to 1.0** for valid predictions

## Files Reference

| File | Purpose |
|------|---------|
| `src/ensemble/schema.py` | Standardized component schema and helpers |
| `src/ensemble/ml_v1.py` | ML weighted average ensemble |
| `src/ensemble/spread_v1.py` | SPREAD weighted average ensemble |
| `src/ensemble/total_v1.py` | TOTAL weighted average ensemble |
| `src/ensemble/config.py` | Configuration loading and resolution |
| `src/ensemble/io.py` | Weight file I/O with normalization |
| `src/data/migrations.py` | Database schema migrations |
| `src/data/bets_repository.py` | Component JSON persistence |
| `src/pipelines/ensemble_tuning.py` | Weight optimization |
| `src/pipelines/ensemble_weight_validation.py` | Validation against baselines |

## Migration Notes

### Updating Existing Ensemble Configs

If you have existing ensemble configurations, no migration is needed. The system automatically:

1. **Normalizes model names** when loading weights
2. **Converts legacy schemas** when reading component JSON
3. **Adds database columns** via migration on first use
4. **Logs warnings** for any misconfigurations

### Breaking Changes

The following behaviors changed:

- **Missing models now default to 0.0 weight** (was 1.0) - models must have explicit weights to be included
- **Component JSON schema changed** - validation code must use new keys (`weight`, `value`, `uncertainty`)
- **Between-variance calculation corrected** - predictions may change slightly due to bug fix

If you need the old behavior temporarily, you can:
1. Add explicit `1.0` weights for all models in your config
2. Update validation code to handle new schema keys
3. Re-tune ensemble weights after deploying fixes

## Performance Considerations

- **Weight normalization** happens once per game (O(n) where n = number of models)
- **Component sorting** is O(n log n) but n is typically small (<10 models)
- **Between-variance calculation** is O(n) and only applied when ≥2 models have SD
- **JSON serialization** is fast for small component arrays

Typical ensemble combination takes <1ms per game on modern hardware.

## Future Enhancements

Potential improvements for future versions:

1. **Confidence intervals** for ensemble predictions
2. **Dynamic weight adjustment** based on recent model performance
3. **Ensemble stacking** (meta-models that learn from base models)
4. **Uncertainty calibration** for ensemble SDs
5. **Model contribution analysis** (Shapley values, ablation studies)
