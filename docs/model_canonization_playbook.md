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

---

## Canonization Workflow

### Registration locations (single sources of truth)
When adding a new model, make sure it is registered in these two places:
1) `src/models/registry.py` (model class registration)
2) `src/ensemble/config.py` (DEFAULT_MARKET_MODELS allowlists used by tuning + ensembles)

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

## “Definition of Done”
A model is canonized when:
- Contracts A/B/C pass
- tests exist that prevent probability drift and metric dropout
- `tune-model --activate` works for all markets without overwriting
- schedule/matchups show the same probability stream that tuning optimizes
