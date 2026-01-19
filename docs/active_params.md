# Active model params provenance

- Tuning runs now auto-activate their best params: when a market tuning run finishes with a numeric `best_score`, the `(sport, season, model, market)` entry in `model_market_active_params` is upserted with `params_source=tuned`, the tuning `run_id`, and the optimized metric. Legacy `model_tuned_params` is mirrored for compatibility.
- Forecasting paths resolve parameters through `resolve_effective_params`, which enforces per-market actives and never falls back to other markets. If tuned runs exist but are not active, a warning is logged.
- Schedule, dashboard, and ensemble outputs now include provenance columns per row: `params_source_label`, `params_source_run_id`, `params_metric_optimized`, `params_best_score`, `params_fingerprint`, `params_nonempty`, plus the existing `params_source`/`tuning_run_id` fields.
- The new CLI helper surfaces what is active: `python -m src.cli.pipeline show-active-params --sport nba --season 2025-26 [--models elo,poisson]`.
- Default/legacy actives remain supported but are labeled as such (`default_active`, `legacy_active`, `missing_active`). When tuned runs exist but no active row is present, `resolve_effective_params` now surfaces `db_market_best_run` to indicate an auto-selected best-run fallback.
