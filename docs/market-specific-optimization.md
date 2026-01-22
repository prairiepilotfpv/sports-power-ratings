# Market-Specific Optimization Contract

This document describes the architecture and contract for market-specific optimization in the sports-power-ratings system.

## Overview

The system supports three prediction markets:
- **ML (Moneyline)**: Win probability predictions
- **SPREAD**: Point spread (margin) predictions  
- **TOTAL**: Over/under (total points) predictions

**Hard Contract**: Each market MUST use parameters and ensembles that are specifically optimized for that market's metrics. Cross-market contamination is not allowed.

## Architecture

### Key Principle: Market Isolation

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│    ML Market    │  │  SPREAD Market  │  │  TOTAL Market   │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ Metric:log_loss │  │ Metric:mae_marg │  │ Metric:mae_total│
│ Models:[elo,bt] │  │ Models:[elo,gs] │  │ Models:[poi,gs] │
│ Params: ML-tuned│  │ Params: SP-tuned│  │ Params: TO-tuned│
│ Ensemble: ML-wt │  │ Ensemble: SP-wt │  │ Ensemble: TO-wt │
└─────────────────┘  └─────────────────┘  └─────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
   home_win_prob       margin_mean/sd       total_mean/sd
```

### Core Components

1. **`MarketParamsResolution`** - Result of resolving parameters for a (model, market) pair
2. **`MarketEnsembleSpec`** - Ensemble configuration for a specific market
3. **`resolve_market_params()`** - Canonical resolver for market-specific model parameters
4. **`get_market_ensemble_spec()`** - Canonical resolver for market-specific ensemble config

### Database Schema

Parameters are stored keyed by `(sport, season, model, market)`:

```sql
-- Active params per model per market
CREATE TABLE model_market_active_params (
    sport TEXT NOT NULL,
    season TEXT NOT NULL,
    model TEXT NOT NULL,
    market TEXT NOT NULL,  -- ML, SPREAD, or TOTAL
    params_json TEXT NOT NULL,
    source_run_id TEXT,
    params_source TEXT,
    metric_optimized TEXT,
    best_score REAL,
    UNIQUE(sport, season, model, market)
);

-- Tuning runs tracked per market
CREATE TABLE model_market_tuning_runs (
    sport TEXT NOT NULL,
    season TEXT NOT NULL,
    model TEXT NOT NULL,
    market TEXT NOT NULL,
    metric_optimized TEXT NOT NULL,
    run_id TEXT NOT NULL,
    best_score REAL,
    best_params_json TEXT,
    ...
);
```

## Parameter Resolution

### Resolution Order

When `resolve_market_params(sport, season, model, market)` is called:

1. **Active Params** - Check `model_market_active_params` for explicit activation
2. **Best Tuning Run** - If no active params, use best score from `model_market_tuning_runs`
3. **Defaults** - If no tuning runs, use model defaults

### Source Labels

The `params_source_label` field indicates provenance:

| Label | Meaning |
|-------|---------|
| `tuned_active` | Explicitly activated from a tuning run |
| `db_market_active` | Active in DB (may be from bootstrap) |
| `db_market_best_run` | Auto-selected from best tuning run |
| `default_active` | Explicit default activation |
| `missing_active` | No params found, using model defaults |
| `legacy_active` | From old non-market-scoped storage |
| `cli` | Provided via CLI override |
| `file` | Loaded from params file |

### Market-Specific Metrics

Each market optimizes for a specific metric:

| Market | Primary Metric | Optimized Label |
|--------|---------------|-----------------|
| ML | `log_loss` | `backtest_log_loss` |
| SPREAD | `mae_margin` | `backtest_mae_margin` |
| TOTAL | `mae_total` | `backtest_mae_total` |

## Ensemble Configuration

### Default Models per Market

```python
DEFAULT_MARKET_MODELS = {
    "ML": ["elo", "bradley-terry"],
    "SPREAD": ["elo", "gssd", "toor"],
    "TOTAL": ["poisson", "gssd"],
}
```

### Resolution Order

1. **Config Override** - Explicit config dict passed to function
2. **Active Ensemble Weights** - `ensemble_market_active_weights` table
3. **Best Ensemble Run** - Best score from `ensemble_market_tuning_runs`
4. **Config File** - `outputs/ensembles/<sport>/<season>/<market>/<ensemble_id>.json`
5. **Defaults** - Equal weights on default models

## Schedule Generation

When generating the schedule Excel report:

1. For each game, predictions are collected per market
2. Each market uses `resolve_market_params()` to get market-specific params
3. Each market's ensemble spec determines which models contribute
4. Output columns are populated from the correct market:
   - `home_win_prob`, `away_win_prob` ← ML pipeline
   - `margin_mean`, `margin_sd` ← SPREAD pipeline
   - `total_mean`, `total_sd` ← TOTAL pipeline

## Debugging

### CLI Diagnostics

```bash
# Show active params status
python -m src.cli.pipeline tuning-status --sport nba --season 2025-26

# Show detailed market diagnostics for a game
python -m src.cli.pipeline market-debug --sport nba --season 2025-26 --game-id ABC123
```

### Programmatic Diagnostics

```python
from pipelines.market_config import (
    collect_game_diagnostics,
    format_diagnostics_report,
)

diagnostics = collect_game_diagnostics(
    db_path=db_path,
    sport="nba",
    season="2025-26",
    game_id="ABC123",
    models=["elo", "poisson"],
)
print(format_diagnostics_report(diagnostics))
```

### Output Example

```
============================================================
MARKET: TOTAL | GAME: ABC123
============================================================

Ensemble: ensemble_total_v1
  Models: ['poisson', 'gssd']
  Weights: {'poisson': 0.6, 'gssd': 0.4}
  Metric: mae_total
  Weights Source: db_active
  Run ID: run-total-202401

Model Parameters:
  [✓] poisson:
      Source: tuned_active
      Metric: backtest_mae_total
      Run ID: run-total-poisson-1
      Non-empty: True
      Params: {"lambda_home": 1.2, "lambda_away": 1.1}
  [✓] gssd:
      Source: tuned_active
      Metric: backtest_mae_total
      Run ID: run-total-gssd-1
      Non-empty: True
      Params: {"sigma": 0.8}
```

## Validation

### Contract Enforcement

The `validate_market_isolation()` function checks:

1. Each model has active params for the market it's used in
2. The `metric_optimized` matches the market's expected metric
3. No cross-market param leakage

### Test Coverage

Tests in `tests/test_market_specific_optimization.py` verify:

- **Params Isolation**: Different params per market don't leak
- **Ensemble Isolation**: TOTAL ensemble only uses TOTAL-configured models
- **No-Leak Fallback**: Missing TOTAL params don't fall back to SPREAD/ML
- **Schedule Columns**: Output columns come from correct market pipelines

## Tuning Workflow

### Per-Market Tuning

```bash
# Tune for ML market
python -m src.cli.pipeline tune --model elo --market ML --metric log_loss ...

# Tune for SPREAD market  
python -m src.cli.pipeline tune --model elo --market SPREAD --metric mae_margin ...

# Tune for TOTAL market
python -m src.cli.pipeline tune --model poisson --market TOTAL --metric mae_total ...
```

### Activation

```bash
# Activate best ML params
python -m src.cli.pipeline activate-tuning --model elo --market ML

# Bootstrap all missing actives from best runs
python -m src.cli.pipeline bootstrap-market-actives --sport nba --season 2025-26
```

## Migration Notes

If you have old non-market-scoped tuned params:

1. They will be detected with `params_source_label="legacy_active"`
2. Run new market-specific tuning to replace them
3. Use `bootstrap-market-actives` to promote best runs to active

## Related Files

- `src/pipelines/market_config.py` - Core market config layer
- `src/pipelines/model_params.py` - Parameter resolution internals
- `src/pipelines/schedule.py` - Schedule generation with market isolation
- `src/ensemble/config.py` - Ensemble configuration loading
- `src/data/repository.py` - Database operations for params/tuning
- `tests/test_market_specific_optimization.py` - Contract tests
