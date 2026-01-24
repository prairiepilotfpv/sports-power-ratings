# Quick Start: Ensemble Weight Validation

## TL;DR

Measure if your tuned ensemble weights actually work better than equal weights.

```bash
# Enable validation when generating schedule
python -m src.cli.pipeline schedule --sport nba --season 2025-26 --model all --validate-ensemble-weights
```

Output:
```
Ensemble weight validation report -> outputs/ensemble_validation/nba_2025-26_20260124T150000Z_ml_weights_comparison.json
  Games analyzed: 42
  Tuned log-loss: 0.5234
  Equal log-loss: 0.5421
  Improvement: 3.45%  ← Tuned weights are 3.45% better
```

## What Gets Compared

| Aspect | Tuned Weights | Equal Weights |
|--------|---------------|---------------|
| **Probabilities** | From `home_win_prob` in BETS sheet | Reconstructed from ensemble components (average) |
| **Test Data** | Completed games only | Same completed games |
| **Metric** | Log-loss + Brier score | Log-loss + Brier score |
| **Result** | Baseline | Comparison |

## Output File

Saved to: `outputs/ensemble_validation/<sport>_<season>_<timestamp>_ml_weights_comparison.json`

```json
{
  "n_games": 42,
  "tuned_log_loss": 0.5234,
  "equal_log_loss": 0.5421,
  "improvement_log_loss_pct": 3.45,
  "status": "complete"
}
```

## Interpretation

```
improvement_log_loss_pct > 5%   → ✅ Weights working well, keep them
improvement_log_loss_pct 2-5%   → ✅ Weights helping, keep them  
improvement_log_loss_pct 0-2%   → ⚠️  Marginal benefit; consider equal
improvement_log_loss_pct < 0%   → ⚠️  Equal weights better (overfitting?)
```

## Non-Breaking

✅ Schedule generates normally regardless of validation
✅ Validation runs after, doesn't affect output  
✅ If validation fails, printed as warning only
✅ Flag is optional, off by default

## That's It

Just add `--validate-ensemble-weights` to your schedule command. Check the report to see if tuning is helping.
