# Hyperparameter Optimization System

## Overview

The sports-power-ratings codebase includes a comprehensive hyperparameter tuning system that optimizes individual model parameters via grid search backtesting. The system is designed to:

1. **Define parameter grids** per model
2. **Grid search** across parameter combinations
3. **Backtest each candidate** to compute performance metrics
4. **Select the best performer** based on a chosen metric
5. **Persist tuned parameters** to the database
6. **Auto-load tuned parameters** during inference

---

## Architecture

### Core Components

```
src/pipelines/tuning.py           # Main tuning orchestration
src/pipelines/tune_batch.py       # Batch tuning runner for multiple models
src/pipelines/model_params.py     # Parameter resolution & persistence
src/models/                       # Model implementations (elo, bradley-terry, gssd, toor, poisson)
src/cli/pipeline.py               # CLI wiring for "tune" command
```

### Key Workflows

#### 1. **Individual Model Tuning** (`src/cli/pipeline.py::_run_tuning`)
```bash
python -m src.cli.pipeline tune \
  --model elo \
  --csv data/raw/nba_history.csv \
  --start 2020-01-01 --end 2024-12-31 \
  --metric log_loss \
  --sport nba --season 2025-26 \
  --apply-best
```

**Process:**
1. Resolve parameter grid (default or override)
2. Compute baseline score (if `--require-improvement`)
3. For each parameter combination:
   - Create a model instance with those params
   - Run backtest via `run_backtest()`
   - Extract performance metric
4. Select best params based on metric
5. If `--apply-best` and improved, persist to DB

#### 2. **Batch Tuning** (`src/pipelines/tune_batch.py::run_tune_batch`)
```bash
python -m src.cli.pipeline tune-batch \
  --sport nba --season 2025-26 \
  --start 2024-11-01 --end 2024-12-01 \
  --csv data/raw/nba.csv
```

**Process:**
- Gathers models from ensemble config (or uses allowlist)
- Runs `run_tuning_pipeline()` for each model/metric pair
- Saves results to `outputs/tuning/`

#### 3. **Parameter Resolution** (`src/pipelines/model_params.py`)
When a model is instantiated during ranking/schedule/projection:
1. Check for CLI overrides (`--model-params`)
2. Check for file-based overrides (`--model-params-file`)
3. Query DB for market-specific tuned params
4. Fall back to defaults if nothing found

---

## Parameter Grids

### Default Grids by Model

#### **Elo** (`src/pipelines/tuning.py::_default_param_grid`)
```python
{
    "k_factor": [10.0, 20.0, 40.0],
    "home_advantage": [0.0, 50.0, 100.0],
    "initial_rating": [1500.0],
    "min_rating": [1.0],
    "total_shrinkage": [0.0, 0.25, 0.5, 0.75, 1.0],
    "total_team_prior_games": [5, 10, 20],
    "total_sd_floor": [0.25, 0.5, 0.75],
    # Also supports per-market overrides:
    # ML/SPREAD: recency_lambda, learn_home_advantage, conditional_sd, winprob_bias, learn_winprob_bias
}
```
- **Primary tuning metrics:** `log_loss` (ML), `brier_score` (ML), `mae_margin` (SPREAD), `mae_total` (TOTAL)
- **Tunable for different markets** via `src/pipelines/market_tuning.py`

#### **Bradley-Terry** 
```python
{
    "temp": [2.0, 3.0, 4.0],              # Softmax temperature
    "l2_lambda": [1e-4, 1e-3, 1e-2],      # L2 regularization
    "learn_hfa": [False, True],           # Learn home field advantage
}
```
- Simple, well-understood hyperparameters
- Tunable via grid search

#### **TOOR** (v1.1+)
```python
{
    "max_iter": [500],
    "tol": [1e-6],
    "optimizer": ["scipy"],               # Tunable: "scipy" vs "ols"
    "initial_home_adv": [2.5, 3.362, 4.5],
    "initial_home_coeff": [17.373, 20.0],
    "initial_away_coeff": [-14.855, -12.0],
    "recency_lambda": [None, 0.01],       # Recency weighting
    "conditional_sd": [False, True],      # Conditional variance
}
```
- Scipy optimizer is tunable (not fixed)
- Initial guesses are tunable (new in v1.1)
- Supports recency weighting and conditional SD

#### **GSSD**
```python
{
    "recency_lambda": [None, 0.005, 0.01, 0.02],
    "learn_home_advantage": [False, True],
    "conditional_sd": [False, True],
    "learn_winprob_bias": [False, True],
}
```
- Simpler grid than Elo (no per-team priors)
- Focuses on weighting and variance modeling

#### **Poisson**
```python
{
    "learning_rate": [0.02, 0.05],
    "reg_strength": [0.05, 0.2],
    "max_iter": [1500],
}
```
- Attack/defense gradient descent optimization
- Regularization controls overfitting

---

## Model Implementations

### Base Class: `BaseModel`
Located in `src/models/base.py`

All tunable models inherit and provide:
- `__init__()` - accepts hyperparameters as kwargs
- `fit(games_df, fit_end_date=None)` - trains on historical data
- `predict(upcoming_games_df)` - returns canonical predictions
- `metadata()` - returns `ModelMetadata` with version, role, ensemble weight

### Tunable Constructor Parameters

#### **Elo** (`src/models/elo.py::EloModel`)
```python
EloModel(
    k_factor=20.0,                       # Learning rate
    home_advantage=65.0,                 # Points/goals to add for home team
    initial_rating=1500.0,               # Starting team rating
    min_rating=1.0,                      # Minimum rating floor
    recency_lambda=None,                 # Exponential decay for weighting
    learn_home_advantage=False,          # Fit HFA vs use preset
    conditional_sd=False,                # Use conditional variance
    winprob_bias=0.0,                    # Offset for win prob
    learn_winprob_bias=False,            # Fit bias vs use preset
    total_shrinkage=0.5,                 # Regularization for total variance
    total_team_prior_games=10,           # Prior strength
    total_sd_floor=0.5,                  # Minimum SD
)
```

#### **Bradley-Terry** (`src/models/bradley_terry.py::BradleyTerryBacktest`)
```python
BradleyTerryBacktest(
    max_iter=500,                        # Iterative solver iterations
    tol=1e-8,                            # Convergence tolerance
    temp=3.0,                            # Softmax temperature (inverse scale)
    l2_lambda=1e-3,                      # L2 regularization strength
    hfa_logit=0.0,                       # Initial HFA estimate (logit scale)
    learn_hfa=True,                      # Learn HFA vs fix at hfa_logit
)
```

#### **TOOR** (`src/models/toor.py::TOORModel`)
```python
TOORModel(
    max_iter=500,                        # Scipy optimizer max iterations
    tol=1e-8,                            # Ftol for scipy.optimize.minimize
    recency_lambda=None,                 # Exponential decay weighting
    learn_home_advantage=True,           # Fit HFA vs use initial_home_adv
    conditional_sd=False,                # Use conditional variance
    winprob_bias=0.0,                    # Offset for ML predictions
    learn_winprob_bias=False,            # Fit bias vs use preset
    optimizer="scipy",                   # "scipy" | "ols" (tunable)
    initial_home_adv=3.362,              # Initial guess (tunable)
    initial_home_coeff=17.373,           # Initial guess (tunable)
    initial_away_coeff=-14.855,          # Initial guess (tunable)
)
```

#### **GSSD** (`src/models/gssd.py::GSSDModel`)
```python
GSSDModel(
    recency_lambda=None,                 # Exponential decay weighting
    learn_home_advantage=False,          # Fit HFA vs use preset
    conditional_sd=False,                # Use conditional variance
    winprob_bias=0.0,                    # Offset for ML predictions
    learn_winprob_bias=False,            # Fit bias vs use preset
    total_shrinkage=0.5,                 # Total variance regularization
    total_team_prior_games=10,           # Prior strength
    total_sd_floor=0.5,                  # Minimum SD
)
```

#### **Poisson** (`src/models/poisson.py::PoissonModel`)
```python
PoissonModel(
    max_iter=1500,                       # Gradient descent iterations
    learning_rate=0.05,                  # Step size
    tol=1e-6,                            # Convergence tolerance
    reg_strength=0.5,                    # L2 regularization
    recency_lambda=None,                 # Exponential decay weighting
    learn_home_advantage=False,          # Fit HFA vs use preset
    conditional_sd=False,                # Use conditional variance
    winprob_bias=0.0,                    # Offset for ML predictions
    learn_winprob_bias=False,            # Fit bias vs use preset
)
```

---

## Tuning Pipeline: Step-by-Step

### **Phase 1: Setup**
```python
# src/pipelines/tuning.py::run_tuning_pipeline()

1. Validate inputs (metric in allowed set, apply_best requires DB info)
2. Load games dataframe from CSV
3. Resolve parameter grid:
   - Check for grid_override (--grid-file)
   - Fall back to _default_param_grid(model)
4. Create output directory (outputs/tuning/<model>/<metric>/<timestamp>/)
```

### **Phase 2: Baseline (if require_improvement=True)**
```python
# Run one backtest with model_cls() (all defaults)
baseline_score = run_backtest(
    lambda: model_cls(),  # Default constructor
    games_df,
    metric=metric,  # Extract e.g., "log_loss" from results
)
```

### **Phase 3: Grid Search**
```python
# For each parameter combination in the grid:

for params in candidates:  # e.g., {"k_factor": 20.0, "home_advantage": 50.0}
    # Create and backtest
    outputs = run_backtest(
        lambda params=params: model_cls(**params),
        games_df,
        start_date, end_date,
        window="expanding",  # or "rolling_days" / "rolling_games"
        metric=metric,
    )
    
    # Extract metric value
    metric_value = outputs.metrics_overall.iloc[0].get(metric)
    
    # Record row:
    # {
    #   "params": {"k_factor": 20.0, ...},
    #   "log_loss": 0.5432,
    #   "brier_score": 0.2234,
    #   "mae_margin": 2.11,
    #   "mae_total": 4.56,
    #   "metric": "log_loss",
    #   "metric_value": 0.5432,
    #   ...
    # }
```

**Parallelization:**
- If `jobs > 1`, uses `parallel_map()` with multiprocessing
- Context is serialized (strings for paths)
- Results sorted by candidate index to ensure determinism

### **Phase 4: Best Selection & Improvement Check**
```python
results = pd.DataFrame(results_rows)

# Select best by minimum metric value (lower is better)
best_idx = results["metric_value"].idxmin()  # or idxmax() if higher-is-better
best_row = results.loc[best_idx]
candidate_score = best_row["metric_value"]

# Check improvement
if require_improvement and baseline_score is not None:
    improved = candidate_score < baseline_score
else:
    improved = True  # Accept any result

best_params = best_row["params"] if improved else {}
```

### **Phase 5: Persistence (if apply_best=True)**
```python
if apply_best and improved and db_path and sport and season:
    # Run best candidate once more (for logging/artifacts)
    run_backtest(
        lambda: model_cls(**best_params),
        games_df,
        output_dir=best_dir,
        db_path=db_path,  # This persists results to DB
        sport=sport,
        season=season,
    )
    
    # Called internally by backtest runner:
    # set_active_model_market_params(
    #     db_path, sport, season,
    #     model=model_name,
    #     market=market_name,
    #     params=best_params,
    #     source_run_id=run_id,
    #     metric_optimized=metric,
    #     best_score=best_score,
    # )
```

### **Phase 6: Output**
```python
# Write CSVs/JSON
tuning_results_{run_id}.csv     # All candidates + metrics
best_params_{run_id}.json       # Best params summary

return TuningOutputs(
    model=model_name,
    metric=metric,
    best_params=best_params,
    best_score=best_score,
    baseline_score=baseline_score,
    improved=improved,
    applied=applied,
    results=results,
    output_dir=base_dir,
)
```

---

## Parameter Resolution & Persistence

### **Saving Tuned Parameters**

**Single-model tuning:**
```bash
python -m src.cli.pipeline tune --model elo --csv data.csv \
  --start 2024-01-01 --end 2024-12-31 --metric log_loss \
  --sport nba --season 2025-26 --apply-best
```

Calls `set_active_model_market_params()` in `src/data/repository.py`:
- **Stored in:** `model_market_active_params` table
- **Keys:** (sport, season, model, market)
- **Values:** JSON params, source run_id, metric optimized, best score

**Legacy mirror (backward compat):**
- Also calls `upsert_model_tuned_params()` to populate `model_tuned_params` table
- Metric displayed as `metric_display_from_optimized()` (removes "backtest_" prefix)

### **Loading Tuned Parameters**

**Automatic (during rank/schedule/predict):**
```python
# src/pipelines/model_params.py::resolve_model_params()

resolved = resolve_model_params(
    "elo",
    db_path=Path("data/db/nba/2025-26.db"),
    sport="nba",
    season="2025-26",
    market=Market.ML,
)
# Returns: {"k_factor": 20.0, "home_advantage": 50.0, ...}
```

**Fallback order:**
1. CLI override (`--model-params '{"k_factor": 25.0}'`)
2. CLI file override (`--model-params-file params.json`)
3. DB active params (market-specific, if available)
4. Legacy DB params (backward compat)
5. Model defaults (constructor defaults)

### **Market-Specific Tuning**

Models support per-market tuning:
```bash
# Tune ELO specifically for SPREAD market
python -m src.cli.pipeline tune \
  --model elo --metric mae_margin --market SPREAD \
  --csv data.csv --start 2024-01-01 --end 2024-12-31 \
  --sport nba --season 2025-26 --apply-best
```

**Market mapping:**
- `log_loss`, `brier_score` → ML market
- `mae_margin` → SPREAD market
- `mae_total` → TOTAL market

---

## Backtest Integration

### **How Tuning Invokes Backtests**

```python
# src/backtest/runner.py::run_backtest()

outputs = run_backtest(
    model_factory=lambda: model_cls(**params),
    games_df=games_df,
    start_date="2024-01-01",
    end_date="2024-12-31",
    window="expanding",           # expanding, rolling_days, rolling_games
    metric=metric,
    output_dir=Path("outputs/tuning/elo/log_loss/..."),
    model_name="elo",
    db_path=Path("data/db/nba/2025-26.db"),
    sport="nba",
    season="2025-26",
)
```

Returns `BacktestOutputs`:
- `metrics_overall` - DataFrame with summary metrics (log_loss, brier_score, mae_margin, mae_total)
- `predictions` - Full prediction list
- `prediction_stats` - Aggregates by date/opponent/etc.

---

## Batch Tuning

### **tune-batch Command**

```bash
python -m src.cli.pipeline tune-batch \
  --sport nba --season 2025-26 \
  --start 2024-11-01 --end 2024-12-01 \
  --csv data/raw/nba.csv \
  --models bradley-terry,elo,gssd,poisson,toor \
  --metrics log_loss,mae_margin
```

**What it does:**
1. Resolves model list (from config union, explicit, or allowlist)
2. For each model × metric combination:
   - Calls `run_tuning_pipeline()`
   - Saves results to `outputs/tuning/<model>/<metric>/`
3. Produces summary report

**Configuration:**
- `--include-all-models` - tune every backtest model (default: exclude experimental)
- `--include-experimental` - include experimental variants (e.g., bradley-terry HFA variants)
- No `--models` - uses union from ensemble configs (sport/season-scoped)

---

## Grid Override Files

### **Format**

```json
{
  "elo": {
    "k_factor": [15.0, 20.0, 25.0],
    "home_advantage": [30.0, 50.0, 70.0],
    "initial_rating": [1500.0],
    "min_rating": [1.0]
  },
  "bradley-terry": {
    "temp": [2.5, 3.0],
    "l2_lambda": [1e-3, 1e-2]
  }
}
```

### **Usage**

```bash
python -m src.cli.pipeline tune --model elo \
  --grid-file custom_grid.json \
  --csv data.csv --start 2024-01-01 --end 2024-12-31 \
  --sport nba --season 2025-26 --apply-best
```

---

## Metrics

### **Supported Metrics** (`_METRICS` in tuning.py)
- `log_loss` - Log loss (ML market, lower is better)
- `brier_score` - Brier score (ML market, lower is better)
- `mae_margin` - Mean absolute error on margin (SPREAD, lower is better)
- `mae_total` - Mean absolute error on total (TOTAL, lower is better)

### **Metric Selection Heuristics**

**Default active metric per model:**
- Elo: `log_loss` (ML)
- Bradley-Terry: `log_loss` (ML)
- TOOR: `mae_margin` (SPREAD)
- GSSD: `log_loss` (ML)
- Poisson: `mae_margin` (SPREAD)

Configured in `src/pipelines/tuning_policy.py::default_active_metric_for_model()`.

---

## Important Conventions

### **1. Parameter Naming**
- Hyperparameters passed to models must match constructor parameter names exactly
- No name translation (e.g., "lambda" ≠ "l2_lambda")
- Grid keys are validated via `inspect.signature()` in `_resolve_tuning_candidates()`

### **2. Grid Size**
- Grids are combinatorial: `product(*values)`
- Elo default: 3 × 3 × 1 × 1 × 5 × 3 × 3 = **405 candidates**
- Bradley-Terry default: 3 × 3 × 2 = **18 candidates**
- Toor default: 1 × 1 × 1 × 3 × 2 × 2 × 2 × 2 = **48 candidates**
- Parallelization recommended for large grids

### **3. Output Structure**
```
outputs/tuning/
  elo/
    log_loss/
      2024-01-01_2024-12-31_expanding/
        2024-01-01_2024-12-31_expanding__baseline/
          backtest_results.csv
          metrics.csv
        2024-01-01_2024-12-31_expanding__best/
          backtest_results.csv
          metrics.csv
        2024-01-01_2024-12-31_expanding__k_factor=20.0_home_advantage=50.0.../
          backtest_results.csv
          metrics.csv
        tuning_results_2024-01-01_2024-12-31_expanding.csv
        best_params_2024-01-01_2024-12-31_expanding.json
```

### **4. Determinism**
- Parameter candidates are generated via `itertools.product()`
- Parallel results are sorted by candidate index before aggregation
- RNG seeds not yet centralized (TODO: add `--seed` flag)

### **5. Baseline Behavior**
- `require_improvement=True` (default): Best candidate must beat baseline
- `require_improvement=False`: Best candidate wins regardless
- Baseline is computed once (not per-metric)

---

## Workflow Examples

### **Example 1: Tune Elo for Log Loss**
```bash
python -m src.cli.pipeline tune \
  --model elo \
  --csv data/raw/nba_2024-25.csv \
  --start 2024-11-01 --end 2024-12-31 \
  --metric log_loss \
  --sport nba \
  --season 2024-25 \
  --apply-best \
  --require-improvement
```

**Output:**
- `outputs/tuning/elo/log_loss/.../tuning_results_*.csv`
- `outputs/tuning/elo/log_loss/.../best_params_*.json`
- Params persisted to DB (nba/2024-25.db)

### **Example 2: Tune All Models with Custom Grid**
```bash
python -m src.cli.pipeline tune-batch \
  --sport nba --season 2024-25 \
  --csv data/raw/nba_2024-25.csv \
  --start 2024-11-01 --end 2024-12-31 \
  --models elo,bradley-terry,gssd,poisson,toor \
  --metrics log_loss,mae_margin \
  --jobs 4
```

**Output:**
- `outputs/tuning/elo/log_loss/.../`
- `outputs/tuning/bradley-terry/log_loss/.../`
- `outputs/tuning/gssd/log_loss/.../`
- ...and so on

### **Example 3: Override Grid for Elo**
```bash
# Create grid file
cat > custom_elo.json << 'EOF'
{
  "elo": {
    "k_factor": [15.0, 20.0, 25.0, 30.0],
    "home_advantage": [40.0, 60.0, 80.0],
    "initial_rating": [1500.0],
    "min_rating": [1.0]
  }
}
EOF

python -m src.cli.pipeline tune \
  --model elo \
  --grid-file custom_elo.json \
  --csv data/raw/nba.csv \
  --start 2024-11-01 --end 2024-12-31 \
  --metric log_loss \
  --sport nba --season 2024-25 \
  --apply-best
```

### **Example 4: Tune TOOR with Multiple Initial Guesses**
```bash
cat > toor_init.json << 'EOF'
{
  "toor": {
    "max_iter": [500, 1000],
    "optimizer": ["scipy"],
    "initial_home_adv": [2.0, 3.362, 5.0],
    "initial_home_coeff": [15.0, 17.373, 20.0],
    "initial_away_coeff": [-18.0, -14.855, -12.0],
    "recency_lambda": [None, 0.01],
    "conditional_sd": [False, True]
  }
}
EOF

python -m src.cli.pipeline tune \
  --model toor \
  --grid-file toor_init.json \
  --csv data/raw/nhl.csv \
  --start 2024-11-01 --end 2024-12-31 \
  --metric mae_margin \
  --sport nhl --season 2024-25 \
  --apply-best
```

---

## Limitations & Future Work

1. **RNG Control**: No centralized seed management; results may vary across runs
2. **Grid Explosion**: Large grids (>1000 candidates) can be slow; needs smarter search (Bayesian, evolutionary)
3. **Market-Agnostic Tuning**: Current system tunes to one metric at a time; multi-objective tuning not yet supported
4. **No Cross-Model Dependencies**: Models tuned independently; no ensemble-aware tuning yet
5. **Parameter Interdependencies**: Assumes parameters are independent; some may interact
6. **Warmup Period**: First N games in expanding window not tuned; could tune warmup separately

---

## Summary

The hyperparameter tuning system is:
- **Modular:** Each model has its own grid definition and parameter set
- **Flexible:** Grids override via JSON, metrics configurable, parallelizable
- **Persistent:** Tuned params stored in DB and auto-loaded during inference
- **Integrated:** Backtests every candidate; selects best via grid search
- **Auditable:** Full results saved to CSV/JSON; run IDs track lineage

---

## Quick Reference: CLI Commands

```bash
# Tune single model
python -m src.cli.pipeline tune --model elo --csv data.csv \
  --start 2024-01-01 --end 2024-12-31 --metric log_loss \
  --sport nba --season 2024-25 --apply-best

# Tune all models (batch)
python -m src.cli.pipeline tune-batch --sport nba --season 2024-25 \
  --csv data.csv --start 2024-01-01 --end 2024-12-31

# Tune with custom grid
python -m src.cli.pipeline tune --model elo --grid-file custom.json \
  --csv data.csv --start 2024-01-01 --end 2024-12-31 \
  --sport nba --season 2024-25 --apply-best

# Tune with parallelization
python -m src.cli.pipeline tune --model elo --csv data.csv \
  --start 2024-01-01 --end 2024-12-31 --metric log_loss \
  --sport nba --season 2024-25 --apply-best --jobs 4
```
