# Model Canonization Playbook

This repo supports multiple predictive models (BT, Elo, GSSD, TOOR, Poisson, etc.) that must remain modular, sport-agnostic, and consistent across:
- schedule/matchup projections
- backtest metrics
- market tuning (ML/SPREAD/TOTAL)
- persistence/activation in season DB

A model is "canonized" when it satisfies the contract checks in this document and can be tuned/activated per market using the existing CLI.

---

## Definitions

### Canonical prediction DTO
`GamePrediction` is the canonical output record for backtests and metrics scoring.

### Heads
Models may produce multiple heads:
- Probability head (ML): `p_home_win` (authoritative probability used for scoring)
- Margin head (SPREAD): `pred_margin` (or `margin_mean` mirrored into `pred_margin`)
- Total head (TOTAL): `pred_total` (or `total_mean` mirrored into `pred_total`)

### Projection engine outputs (schedule/matchups)
Projection engines provide the schedule-facing equivalents:
- `model_p_home_win` (model’s canonical probability for schedule display)
- `projected_win_prob` (may mirror model prob)
- `normal_p_home_win` (diagnostic only; margin-normal approximation)
- `margin_mean`, `margin_sd`
- `total_mean`, `total_sd`
- `win_prob_source` label (semantic tag)

---

## Canonical Contracts

### Contract A — Backtest scoring contract (must pass)
Backtest metrics depend on:
- `GamePrediction.p_home_win` for `log_loss`/`brier_score`
- `GamePrediction.pred_margin` for `mae_margin`
- `GamePrediction.pred_total` for `mae_total`

Rules:
1) If the model claims a margin head, `pred_margin` must be populated whenever a margin mean exists.
2) If the model claims a total head, `pred_total` must be populated whenever a total mean exists.
3) Avoid silent metric dropout: missing `pred_*` fields must be caught by regression tests.
4) `p_home_win` must be a single authoritative probability stream for that model and match what tuning optimizes.

### Contract B — Schedule/matchup probability semantics (must pass)
Schedule/matchups must display the same probability stream that backtest/tuning optimize.
Rules:
1) `model_p_home_win` is the authoritative schedule probability for that model.
2) `win_prob_source` must describe that probability stream:
   - `direct` (model provides probability directly)
   - `logistic` (logistic mapping used as canonical)
   - `margin_normal` (prob derived from margin + SD)
   - `sample` (simulation-based)
3) `normal_p_home_win` is diagnostic only and must not replace canonical prob fields.
4) Specialized engines (registered by model_id) may be used to enforce model-specific semantics; avoid if/else in the default engine.

### Contract C — Market tuning (must pass)
One command per model must tune all markets:
- ML, SPREAD, TOTAL
and persist separately per market without overwriting.
Rules:
1) `tune-model` must run all markets sequentially (or selected markets via flags).
2) Each market run persists to `model_market_tuning_runs` (distinct rows).
3) Activation persists to `model_market_active_params` per market.
4) Downstream pipelines resolve market params automatically from DB (no manual flags needed).

### Contract D — Ensemble integration (must pass)
When multiple models are active, ensembles aggregate predictions for each market.
Rules:
1) Ensemble classes (MLWeightedAverageEnsemble, SpreadWeightedAverageEnsemble, TotalWeightedAverageEnsemble) must all be imported in `src/pipelines/schedule.py`.
2) Ensembles activate automatically when 2+ models are present (unless overridden by config).
3) Source labels must show `"ensemble_<version>"` (e.g., `"ensemble_ml_v1"`), never `"direct+ensemble"` or concatenated strings.
4) Each market uses its own ensemble class; weights are persisted separately per market in `ensemble_market_weights` table.
5) Weight sources are tracked: `tuned`, `default`, or `config-file` (see `get_active_ensemble_market_weights_source()`).

---

## Canonization Workflow

### Step 0 — Recon (no code changes)
For the target model:
1) Identify tunable surface: `__init__` args + `fit()` kwargs
2) Identify backtest adapter: where `GamePrediction` is built
3) Identify schedule projection path: which projection engine is used and what it outputs
4) Identify any semantic mismatches:
   - multiple probabilities with different meanings
   - schedule shows different p than tuning optimizes
   - `pred_*` missing leading to silent metric dropout

Deliver: a minimal patch plan (few files, few functions)

### Step 1 — Patch (minimal changes)
Apply only what’s needed:
- make `p_home_win` authoritative and consistent with schedule’s `model_p_home_win`
- ensure `pred_margin` / `pred_total` are populated when means exist
- if schedule semantics differ, add a dedicated projection engine and register it
- avoid model math changes unless explicitly required
#### Market-Specific Tuning Implementation
To support per-market tuning:
1. **Identify tunable parameters**: List all `__init__` args and `fit()` kwargs that affect predictions.
2. **Accept market context**: Model must accept a `market` parameter (e.g., `"ML"`, `"SPREAD"`, `"TOTAL"`) to load market-specific params.
3. **Query active params**: Use `get_active_model_market_params(db_path, sport, season, model, market)` to fetch persisted params.
4. **Override defaults**: Merge market params over base defaults without mutating shared state.

Example:
```python
class MyModel(BaseModel):
    def __init__(self, k_factor: float = 20.0, **kwargs):
        self.k_factor = k_factor
        # Accept market-specific overrides via kwargs
        
    def fit(self, games_df: pd.DataFrame, market: str = "ML", **kwargs) -> None:
        # Use self.k_factor which may have been overridden for this market
        ...
```

#### Calibration System Integration
If your model produces spread/total estimates, integrate variance modeling:
1. **Margin SD**: Use `ConditionalSDModel` to predict spread uncertainty conditional on margin magnitude.
   - Fit parameters: `conditional_sd_intercept`, `conditional_sd_slope`
   - Apply guardrails via `guardrail_margin_sd(raw_sd, guardrail_min, guardrail_max)`
2. **Recency weighting**: Use suite-wide semantics from `calibration.recency_weight()`:
   - Age unit: **days** (not games)
   - Curve: `exp(-lambda * days)`
   - As-of date: `fit_end_date` resolved from training data max date
3. **Projection context**: When building projections, pass `conditional_sd_intercept` and `conditional_sd_slope` in projection context so the default engine applies guardrails automatically.

#### Projection Engine Registration
If your model has unique projection semantics, register a dedicated engine:
```python
# In src/pipelines/projection_engines.py
def my_model_projection_engine(
    home_team: str,
    away_team: str,
    model: Any,
    context: ProjectionContext,
) -> ProjectionOutput:
    # Custom projection logic
    ratings = context.get("ratings", {})
    home_rating = ratings.get(home_team)
    away_rating = ratings.get(away_team)
    
    # Build projection with model-specific semantics
    margin_mean = home_rating - away_rating
    margin_sd = model.get_margin_sd()  # Model-specific
    
    # Return canonical structure
    return {
        "projected_home_score": None,
        "projected_away_score": None,
        "projected_total": None,
        "projected_win_prob": None,
        "margin_mean": margin_mean,
        "margin_sd": margin_sd,
        "total_mean": None,
        "total_sd": None,
        "logistic_home_win_prob": logistic_win_prob(margin_mean, k=20),
        "model_p_home_win": logistic_win_prob(margin_mean, k=20),  # Authoritative
        "win_prob_source": "logistic",
    }

# Register for your model_id
register_projection_engine("my-model", my_model_projection_engine)
```

**When to use a dedicated engine:**
- Model provides probability via simulation/sampling (not logistic)
- Model has native spread/total estimates that bypass rating arithmetic
- Model requires custom guardrails or transformations

**When NOT to use:**
- Model uses standard rating-difference → logistic → prob flow
- Only difference is parameter values (handle via tuning, not code branching)
### Step 2 — Regression tests
Add 1–3 tests:
- probability bounds and single-stream semantics
- MAE dropout prevention (`pred_margin`/`pred_total` mirroring)
- projection engine semantics (win_prob_source, logistic dominance, etc.)

### Step 3 — Full suite + smoke
- `pytest -q` must pass
- run one schedule and one matchup command for the model (manual smoke)
- run `tune-model --activate` and confirm 3 market rows exist for active params

---
## Pitfalls & Anti-patterns

### ❌ Anti-pattern: Concatenated source labels
**Bad:**
```python
win_prob_source = "direct+ensemble"  # Ambiguous!
```
**Good:**
```python
# Single source only
win_prob_source = "ensemble_ml_v1"  # When ensemble is active
win_prob_source = "logistic"        # When single model
```
**Why:** Source labels must be unambiguous. If multiple sources contribute, use the **dominant** or **final** source only.

### ❌ Anti-pattern: Diagnostic probability leaking into canonical fields
**Bad:**
```python
projection["model_p_home_win"] = normal_p_home_win  # Diagnostic!
projection["projected_win_prob"] = normal_p_home_win
```
**Good:**
```python
projection["normal_p_home_win"] = normal_p_home_win  # Diagnostic field
projection["model_p_home_win"] = logistic_p  # Canonical field
projection["projected_win_prob"] = logistic_p
```
**Why:** `normal_p_home_win` is for diagnostics/comparison only. The canonical probability must match what backtest/tuning optimize.

### ❌ Anti-pattern: Custom game ID formats
**Bad:**
```python
game_id = f"{home_team}-vs-{away_team}-{date}"  # Not canonical!
```
**Good:**
```python
from utils.game_id import make_game_id
game_id = make_game_id(sport, season, date, home_team, away_team)
# Returns: "nba:2024-25:2025-01-15:abc123def456"
```
**Why:** Hash-based IDs ensure consistency across all import paths and prevent duplicates.

### ❌ Anti-pattern: Model-specific branches in default projection engine
**Bad:**
```python
def _rating_projection_engine(...):
    if model_id == "my-special-model":
        # Special logic here
        return special_projection(...)
    else:
        # Default logic
        return default_projection(...)
```
**Good:**
```python
# In projection_engines.py
def my_special_model_engine(...) -> ProjectionOutput:
    # Dedicated engine
    return {...}

register_projection_engine("my-special-model", my_special_model_engine)
```
**Why:** Keep the default engine generic. Model-specific logic belongs in dedicated, registered engines.

### ❌ Anti-pattern: Silent metric dropout
**Bad:**
```python
class MyModel(BaseModel):
    def predict(self, home_team: str, away_team: str) -> GamePrediction:
        return GamePrediction(
            p_home_win=0.55,
            pred_margin=None,  # Missing! MAE will fail silently
            margin_mean=3.5,   # Data exists but not mirrored
            ...
        )
```
**Good:**
```python
class MyModel(BaseModel):
    def predict(self, home_team: str, away_team: str) -> GamePrediction:
        margin = self.get_margin(home_team, away_team)
        return GamePrediction(
            p_home_win=0.55,
            pred_margin=margin,  # Mirror for MAE scoring
            margin_mean=margin,
            ...
        )
```
**Why:** If a model claims a margin/total head, `pred_margin`/`pred_total` must be populated to enable metric scoring.

### ❌ Anti-pattern: Overwriting market params
**Bad:**
```python
# Save ML params
set_active_model_market_params(db, sport, season, model, "ML", ml_params, ...)
# Save SPREAD params - overwrites ML!
set_active_model_market_params(db, sport, season, model, "ML", spread_params, ...)  # Wrong market!
```
**Good:**
```python
# Each market gets its own row
set_active_model_market_params(db, sport, season, model, "ML", ml_params, ...)
set_active_model_market_params(db, sport, season, model, "SPREAD", spread_params, ...)
set_active_model_market_params(db, sport, season, model, "TOTAL", total_params, ...)
```
**Why:** Market params are isolated by the `market` column. Never reuse the wrong market key.

### ❌ Anti-pattern: Redefining recency semantics per model
**Bad:**
```python
class MyModel(BaseModel):
    def fit(self, games_df: pd.DataFrame) -> None:
        # Custom recency: games, not days
        for i, game in enumerate(games_df.iterrows()):
            weight = 0.95 ** (len(games_df) - i)  # Game-based decay
            ...
```
**Good:**
```python
from models.calibration import recency_weight, resolve_fit_end_date

class MyModel(BaseModel):
    def fit(self, games_df: pd.DataFrame, recency_lambda: float = 0.0) -> None:
        fit_end_date = resolve_fit_end_date(games_df)
        weights = games_df.apply(
            lambda row: recency_weight(row["date"], fit_end_date, recency_lambda),
            axis=1,
        )
        # Use suite-wide day-based decay
        ...
```
**Why:** Suite-wide recency semantics (day-based exponential decay) ensure consistent behavior across models and make tuning results comparable.

### ❌ Anti-pattern: Missing ensemble imports in schedule.py
**Bad:**
```python
# In schedule.py - only import ML ensemble
from ensemble.ml_v1 import MLWeightedAverageEnsemble
# Forgot SPREAD and TOTAL!
```
**Good:**
```python
# In schedule.py - import all three market ensembles
from ensemble.ml_v1 import MLWeightedAverageEnsemble
from ensemble.spread_v1 import SpreadWeightedAverageEnsemble
from ensemble.total_v1 import TotalWeightedAverageEnsemble
```
**Why:** All three ensemble classes must be imported for multi-model schedules to work across all markets. Missing imports cause runtime errors when building BETS sheet.

---
## Canonized Model Tracker

| Model | Contracts A/B/C/D | Regression Tests | Tune-Model Works | Calibration | Status |
|-------|-------------------|------------------|------------------|-------------|--------|
| Elo | ✅ | ✅ | ✅ | ✅ | Canonized |
| Bradley-Terry | ✅ | ✅ | ✅ | ✅ | Canonized |
| Poisson | ✅ | ✅ | ✅ | ✅ | Canonized |
| GSSD | ✅ | ✅ | ✅ | ✅ | Canonized |
| TOOR | ✅ | ✅ | ✅ | ✅ | Canonized |
| ZSD | ✅ | ✅ | ✅ | ✅ | Canonized |

---

## "Definition of Done"
A model is canonized when:
- **Contracts A/B/C/D pass**: backtest scoring, schedule semantics, market tuning, ensemble integration
- **Regression tests exist**: prevent probability drift, metric dropout, source labeling errors
- **`tune-model --activate` works**: for all 3 markets (ML/SPREAD/TOTAL) without overwriting
- **Schedule/matchups consistency**: show the same probability stream that tuning optimizes
- **Calibration integrated**: if applicable, margin SD uses conditional model + guardrails, recency uses suite-wide semantics
- **Projection engine registered**: if needed for model-specific semantics; otherwise uses default rating-based engine
- **Ensemble compatibility**: model works in multi-model schedules with proper source labeling
