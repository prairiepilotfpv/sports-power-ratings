# Ensemble Calibration System

Independent calibration subsystem for ensemble predictions. Completely separate from the betting pipeline.

## Purpose

Calibrate ensemble predictions against historical outcomes to:
1. Reduce prediction bias
2. Improve probability estimates for EV calculations
3. Fit calibrators that can be applied to future predictions

## Architecture

```
Historical Games (completed, with scores)
    ↓
Generate Ensemble Predictions (for each model)
    ↓
Ensemble Combine (weighted average)
    ↓
Build Calibration Dataset (predictions vs outcomes)
    ↓
Fit Calibrators (isotonic regression or Platt scaling)
    ↓
Save Calibrators (to models/calibrators/market/ensemble_id/)
    ↓
Apply to Future Predictions
```

## Usage

### Command Line

```bash
python -m src.calibration.calibration_cli \
  --db data/db/nba/2025-26.db \
  --sport nba \
  --season 2025-26 \
  --models bradley-terry elo gssd \
  --ensemble-id ensemble_v1 \
  --method isotonic \
  --start-date 2025-10-01 \
  --end-date 2025-12-31
```

### Programmatic

```python
from src.calibration import calibrate_ensemble

calibrators = calibrate_ensemble(
    db_path='data/db/nba/2025-26.db',
    sport='nba',
    season='2025-26',
    models=['bradley-terry', 'elo', 'gssd'],
    ensemble_id='ensemble_v1',
    start_date='2025-10-01',
    end_date='2025-12-31',
    method='isotonic',
)

for market, path in calibrators.items():
    print(f"{market}: {path}")
```

## Key Functions

### `calibrate_ensemble()`
Complete workflow: generate predictions, fit calibrators.

**Args:**
- `db_path`: Database path
- `sport`: Sport identifier
- `season`: Season identifier  
- `models`: List of model names to ensemble
- `ensemble_id`: ID for saved calibrators
- `start_date`: Optional date filter
- `end_date`: Optional date filter
- `method`: 'auto', 'isotonic', or 'platt'

**Returns:**
Dictionary mapping market names to calibrator file paths

### `generate_predictions_for_games()`
Generate ensemble predictions for a set of games.

### `compute_outcomes()`
Load actual game outcomes from database.

### `build_calibration_dataset()`
Match predictions to outcomes for calibrator fitting.

### `fit_and_save_calibrator()`
Fit calibrator and persist to disk.

## How It Works

1. **Load Games**: Query database for games in date range
2. **Generate Predictions**: Run each model for each game, each market
3. **Ensemble**: Combine model predictions with equal or tuned weights
4. **Load Outcomes**: Get actual scores from database
5. **Build Dataset**: Match predictions to outcomes
6. **Fit**: Use isotonic regression or Platt scaling to calibrate
7. **Save**: Persist calibrators to filesystem for later use

## Markets

- **ML** (Moneyline): Home win probability
- **SPREAD**: Margin/spread cover probability  
- **TOTAL**: Over/under probability

## Calibration Methods

- **auto**: Automatically select (isotonic if ≥100 records, else Platt)
- **isotonic**: Isotonic regression (non-parametric, recommended)
- **platt**: Platt scaling (logistic sigmoid)

## Logging

Comprehensive logging to `tmp_calibration.log`:
- Prediction generation progress
- Dataset construction details
- Calibrator fitting metrics
- File paths for saved calibrators

## Integration with Pipeline

Currently standalone. When ready to integrate:
1. Can be called from main pipeline CLI
2. Will read tuned weights from DB if available
3. Can feed calibrators back to pipeline for prediction use
4. No dependency on betting pipeline

## Files

- `calibration_engine.py`: Core logic
- `calibration_cli.py`: Command-line interface
- `__init__.py`: Package exports
- `README.md`: This file
