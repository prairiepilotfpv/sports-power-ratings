# Ensemble configuration guide

Goal: keep ensembles runnable with zero setup while allowing per-market overrides and short CLI commands.

## Resolution order (no surprises)
Per market, configs are resolved in this order (first hit wins):
1. Custom: `outputs/ensembles/<sport>/<season>/<market>/<ensemble_id>.json`
2. Sport/season default: `outputs/ensembles/<sport>/<season>/<market>/default.json`
3. Global repo default: `src/ensemble/default_configs/<market>.json`
4. Fallback: generated from the default allowlist (equal weights).

Legacy multi-market configs are still supported at:
- `outputs/ensembles/<sport>/<season>/ensemble_config.json` (with a `markets` object).
These are merged into the per-market resolution order above.

Schema for any file:
```json
{
  "sport": "nba",              // optional
  "season": "2025-26",         // optional
  "market": "ML",
  "ensemble_id": "ensemble_ml_v1",
  "metric_slot": "log_loss",   // ML|SPREAD|TOTAL defaults below
  "models": ["elo", "bradley-terry"],
  "weights": {"elo": 0.5, "bradley-terry": 0.5}
}
```

Defaults per market (also baked into repo JSON under `src/ensemble/default_configs/`):
- ML: `ensemble_ml_v1`, models `["elo", "bradley-terry"]`, metric `log_loss`
- SPREAD: `ensemble_spread_v1`, models `["elo", "gssd", "toor"]`, metric `mae_margin`
- TOTAL: `ensemble_total_v1`, models `["poisson", "gssd"]`, metric `mae_total`
Weights: equal if omitted or invalid. Missing models fall back to the default allowlist; unavailable models are dropped and weights renormalized.

## CLI helper: create season defaults
Write per-market defaults for a season (skips existing files unless `--overwrite`):
```bash
python -m src.cli.pipeline init-ensemble-config --sport nba --season 2025-26
```
Writes:
- `outputs/ensembles/nba/2025-26/ML/default.json`
- `outputs/ensembles/nba/2025-26/SPREAD/default.json`
- `outputs/ensembles/nba/2025-26/TOTAL/default.json`

## How schedule uses configs
- Always resolves configs via the order above; you will get an ensemble even with zero files in `outputs/`.
- Models/weights/metric_slot come from the resolved config (config weights override DB/file weights).
- Ensemble weights are resolved in this order: config weights -> active DB weights -> on-disk weights -> equal fallback.
- META records per-market config source/path, resolved models/weights, and metric slot.
- Missing/filtered models are skipped and weights renormalized.

## How tune-batch chooses models
- No `--models` provided: union of models across resolved configs (sport/season-scoped). If no configs, uses the default allowlist union.
- `--include-all-models`: tune every backtest model (old behavior).
  - `--include-experimental`: include models marked experimental (e.g., `bradley-terry`). Default excludes experimental variants.
- `--models` provided: exact list is used.

Example:
```bash
python -m src.cli.pipeline tune-batch \
  --sport nba --season 2025-26 \
  --start 2024-11-01 --end 2024-12-01 \
  --csv data/raw/nba.csv
```
Prints the final model list before tuning; experimental models are excluded unless explicitly requested.

## Customizing membership/weights
1) Copy a global default file to the market path you want (or use `init-ensemble-config`).
2) Edit `models` and `weights` (weights auto-renormalize; extras trimmed).
3) Optional: change `metric_slot` if you intentionally want a different tuned metric for that market.
4) Rerun `schedule` / `tune-batch`. META will show which file was used and the resolved models/weights.

## Running ensembles end-to-end

```bash
# Create defaults per market
python -m src.cli.pipeline init-ensemble-config --sport nba --season 2025-26

# Tune ensemble weights for each market (optional but recommended)
python -m src.cli.pipeline tune-ensemble --sport nba --season 2025-26 --market ML --ensemble ensemble_ml_v1 \
  --start-date 2020-01-01 --end-date 2024-12-31 --csv data/raw/nba_history.csv
python -m src.cli.pipeline tune-ensemble --sport nba --season 2025-26 --market SPREAD --ensemble ensemble_spread_v1 \
  --start-date 2020-01-01 --end-date 2024-12-31 --csv data/raw/nba_history.csv
python -m src.cli.pipeline tune-ensemble --sport nba --season 2025-26 --market TOTAL --ensemble ensemble_total_v1 \
  --start-date 2020-01-01 --end-date 2024-12-31 --csv data/raw/nba_history.csv

# (Optional) Calibrate ML ensemble probabilities
python -m src.cli.pipeline calibrate --sport nba --season 2025-26 --market ML --source ensemble_ml_v1 \
  --start-date 2020-01-01 --end-date 2024-12-31 --csv data/raw/nba_history.csv

# Export schedule forecasts (workbook includes ensemble outputs)
python -m src.cli.pipeline schedule --sport nba --season 2025-26 --model all --strict
```

## Forecast outputs from ensembles

The schedule workbook (`schedule_with_projections.xlsx`) includes a `BETS` sheet with ensemble-aware columns:
- ML: `home_win_prob`, `away_win_prob`, `win_prob_source`, `ml_ensemble_components_json`.
- SPREAD: `margin_mean`, `margin_sd`, `spread_source`, `spread_ensemble_components_json`.
- TOTAL: `total`, `total_sd`, `total_source`, `total_ensemble_components_json`.
- `market_forecast_source` captures the forecast source used for the row.

## Troubleshooting
- **Config missing:** global default is used; META shows `source=fallback/global_default`.
- **Model unavailable:** it is dropped and weights are renormalized (warning captured in META).
- **Empty weights:** equal weights are applied.
- **No configured models remain:** falls back to available models for that market.
- **Experimental models showing up:** rerun without `--include-experimental` or remove them from the config.

## Phase 6: Ensemble Governance (Advanced)

See [PHASE6_ENSEMBLE_GOVERNANCE.md](../PHASE6_ENSEMBLE_GOVERNANCE.md) for details on:
- **Market model allowlists**: Single source of truth for which models per market
- **Weight governance**: MIN_WEIGHT_EPS clamping, MIN_NEFF threshold enforcement
- **EnsembleAudit**: Comprehensive audit object tracking dropped models, coverage, Neff
- **Strict mode**: Fail-fast on weight collapse or insufficient model count
- **No silent fallback**: All fallbacks logged loudly for visibility

Quick reference:
```python
# In src/config.py
MARKET_MODEL_ALLOWLISTS = {
    "ML": ["bradley-terry", "elo", "gssd", "poisson", "toor"],
    "SPREAD": ["bradley-terry", "elo", "gssd", "poisson", "toor"],
    "TOTAL": ["bradley-terry", "elo", "gssd", "poisson", "toor"],
}

MIN_WEIGHT_EPS = 0.01  # Clamp weights below this
MIN_NEFF = 1.5  # Minimum effective model count
ENSEMBLE_STRICT_MODE = False  # Fail on Neff < threshold (when True)
```

Audit logging example:
```
[ensemble audit][ML] source=db_best_run Neff=2.8 (threshold=met) 
models=['elo', 'bradley-terry'] weights={'elo': 0.6, 'bradley-terry': 0.4}
```

