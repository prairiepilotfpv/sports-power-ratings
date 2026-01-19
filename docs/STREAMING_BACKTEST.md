# Streaming backtests (Elo)

This repo now supports a streaming backtest fast-path for models that can update incrementally. It is enabled automatically when a model advertises `supports_incremental_update=True` (currently Elo) and the backtest window is expanding.

## What changes
- The backtest walks games chronologically once.
- Predictions are made for each evaluation date using the current model state.
- After predictions, the model is updated in-place with the game results.

## Calibration note
Elo’s calibration and totals parameters are computed in `fit()`. Streaming mode performs a single initial fit (and does not re-fit by default), so calibration/total parameters stay fixed as ratings update. This means streaming outputs can differ slightly from the expanding-window refit-per-date path.

To approximate the original behavior, you can optionally refit every N days or games using environment variables:
- `ELO_STREAM_REFIT_DAYS=7`
- `ELO_STREAM_REFIT_GAMES=250`

## Profiling
Set `TUNE_PROFILE=1` to print timing breakdowns per backtest run. Streaming runs include a `streaming=1` flag plus update/refit timings.
