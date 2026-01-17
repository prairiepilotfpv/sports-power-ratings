# Ensemble Optimizer

Usage examples:

- Directory of per-model predictions (each file `<model>.csv` with `game_id,prob`):

```bash
python tools/ensemble_optimize.py --preds-dir outputs/preds/2024-25/ --labels data/raw/NBA2024-25.csv --out outputs/tuning/ensembles/ml_2024_weights.json
```

- Single CSV with columns `game_id,modelA,modelB,...`:

```bash
python tools/ensemble_optimize.py --preds-csv outputs/preds/2024-25/combined_preds.csv --labels data/raw/NBA2024-25.csv --out outputs/tuning/ensembles/ml_2024_weights.json
```

Notes:
- The script uses projected gradient descent to optimize weights constrained to the simplex (non-negative, sum to 1) minimizing log loss.
- The labels CSV must include `game_id,home_score,away_score` columns to compute binary home-win outcomes.
- Output JSON format:

```json
{
  "weights": {"modelA": 0.4, "modelB": 0.6},
  "meta": {"loss": 0.65}
}
```

If you want, I can add a CLI wrapper to automatically call this from `src/cli` or integrate it into the ensemble pipeline.
