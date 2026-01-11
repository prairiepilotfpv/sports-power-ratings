# Ensembles

This project supports lightweight ensembles for market-level forecasts. The initial implementation provides an ML (moneyline) ensemble that computes a weighted average of model `home_win` probabilities.

Usage
- Place a JSON file at `outputs/ensembles/<sport>/<season>/ensemble_ml_v1.json` containing a mapping of model names to numeric weights, e.g.:

```json
{
  "elo": 2.0,
  "bradley-terry": 1.0,
  "poisson": 0.5
}
```

- If the weights file is missing, the ensemble falls back to equal weights across available models for each game.
- The ensemble ID is `ensemble_ml_v1`. The schedule report pipeline will apply the ensemble to the `BETS` sheet when multiple models are produced for the chosen `bets_model` date and add a provenance column `ml_ensemble_components_json` containing per-model components and weights.

Calibration
- If a calibrator artifact exists under `outputs/calibrators/<sport>/<season>/<ensemble_id>/` (e.g. a `.joblib` file), the pipeline will attempt to apply it to the ensemble raw probabilities before writing the `BETS` sheet.

Design notes
- The ensemble API is intentionally minimal and dataframe-based; it does not introduce new prediction contracts. See `src/ensemble/ml_v1.py` for implementation details.
