# Ensemble configuration guide

Goal: keep ensembles runnable with zero setup while allowing per-market overrides and short CLI commands.

## Resolution order (no surprises)
Per market, configs are resolved in this order (first hit wins):
1. Custom: `outputs/ensembles/<sport>/<season>/<market>/<ensemble_id>.json`
2. Sport/season default: `outputs/ensembles/<sport>/<season>/<market>/default.json`
3. Global repo default: `src/ensemble/default_configs/<market>.json`
4. Fallback: generated from the default allowlist (equal weights).

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
- ML: models `["elo", "bradley-terry"]`, metric `log_loss`
- SPREAD: models `["elo", "gssd", "toor"]`, metric `mae_margin`
- TOTAL: models `["poisson", "gssd"]`, metric `mae_total`
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
- META now records per-market config source/path, resolved models/weights, and metric slot.
- Missing/filtered models are skipped and weights renormalized.

## How tune-batch chooses models
- No `--models` provided: union of models across resolved configs (sport/season-scoped). If no configs, uses the default allowlist union.
- `--include-all-models`: tune every backtest model (old behavior).
- `--include-experimental`: include models marked experimental (e.g., `bradley_terry_hfa`). Default excludes experimental variants.
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

## Troubleshooting
- **Config missing:** global default is used; META shows `source=fallback/global_default`.
- **Model unavailable:** it is dropped and weights are renormalized (warning captured in META).
- **Empty weights:** equal weights are applied.
- **No configured models remain:** falls back to available models for that market.
- **Experimental models showing up:** rerun without `--include-experimental` or remove them from the config.
