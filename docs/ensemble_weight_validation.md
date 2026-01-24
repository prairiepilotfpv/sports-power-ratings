# Ensemble Weight Validation

## Overview

Validates whether tuned ML ensemble weights actually outperform equal weights on completed games. Non-breaking validation that runs **after** schedule generation.

## Usage

Enable with the `--validate-ensemble-weights` flag:

```bash
python -m src.cli.pipeline schedule \
  --sport nba --season 2025-26 \
  --model all \
  --validate-ensemble-weights
```

## Output

Produces two outputs:

### 1. CLI Report (printed to stdout)
```
Ensemble weight validation report -> outputs/ensemble_validation/nba_2025-26_20260124T150000Z_ml_weights_comparison.json
  Games analyzed: 42
  Tuned log-loss: 0.5234
  Equal log-loss: 0.5421
  Improvement: 3.45%
```

### 2. JSON Report File
Saved to: `outputs/ensemble_validation/<sport>_<season>_<timestamp>_ml_weights_comparison.json`

Example content:
```json
{
  "sport": "nba",
  "season": "2025-26",
  "generated_at": "2026-01-24T15:00:00Z",
  "n_games": 42,
  "tuned_log_loss": 0.5234,
  "tuned_brier_score": 0.1823,
  "equal_log_loss": 0.5421,
  "equal_brier_score": 0.1945,
  "improvement_log_loss_pct": 3.45,
  "improvement_brier_pct": 6.27,
  "status": "complete",
  "reason": null
}
```

## Interpretation

**Good signs:**
- `improvement_log_loss_pct > 2%`: Weights are genuinely helping
- `improvement_log_loss_pct > 5%`: Strong evidence of effective weighting
- Both games analyzed (`n_games`) and metrics populated

**Warning signs:**
- `improvement_log_loss_pct < 0%`: Equal weights are better (overfitting?)
- `improvement_log_loss_pct` very large (>10%): Possible data leakage or selection bias
- `status: "incomplete"`: Missing component data; can't compute baseline

## How It Works

1. **After schedule Excel is generated**, loads the BETS sheet
2. **Merges with completed games** from the database
3. **Extracts outcomes** (home_win = 1 if home_score > away_score)
4. **Computes tuned loss** from `home_win_prob` column
5. **Reconstructs equal-weight baseline** from `ml_ensemble_components_json`:
   - If components=[{prob: 0.6}, {prob: 0.8}]
   - Equal weight = (0.6 + 0.8) / 2 = 0.7
6. **Compares both** using log-loss and Brier score
7. **Saves report** with metrics and improvement percentage

## Safety

✅ **Non-breaking:**
- Runs **after** schedule generation completes
- Doesn't modify schedule output
- Failures caught and logged (printed as warning)
- Errors don't prevent schedule export

✅ **Data integrity:**
- Only reads completed games (status='completed')
- Skips invalid probabilities (NaN)
- Handles missing component JSON gracefully

## Metrics Explained

### Log-Loss (Binary Cross-Entropy)
- **Lower is better**
- Measures probability calibration accuracy
- Heavily penalizes confident wrong predictions
- Formula: `-mean(y*log(p) + (1-y)*log(1-p))`

### Brier Score
- **Lower is better**
- Mean squared error of probabilities
- Range: 0 (perfect) to 1 (worst)
- Formula: `mean((p - y)^2)`

### Improvement %
- **Percentage reduction in loss**
- Formula: `(equal_loss - tuned_loss) / equal_loss * 100`
- Positive = tuned is better
- Negative = equal is better (suggests overfitting)

## Next Steps

**If improvement > 2%:**
- Tuning is effective; keep using optimized weights

**If improvement ≈ 0%:**
- Weights aren't helping; could use equal weights to reduce complexity
- Consider if models are too correlated

**If improvement < 0%:**
- Warning: weights may be overfitting to training window
- Consider wider training window or recalibration

## Disabling Validation

Validation is **off by default**. Simply don't use the flag:

```bash
# Skips validation
python -m src.cli.pipeline schedule --sport nba --season 2025-26 --model all
```

## Implementation Details

- **Module:** `src/pipelines/ensemble_weight_validation.py`
- **Integration:** `src/cli/pipeline.py` _run_schedule()
- **Tests:** `tests/test_ensemble_weight_validation.py`
- **Metrics:** Uses calibration helpers from `src/calibration/eval.py`
