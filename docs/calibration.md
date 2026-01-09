Calibration (post-fit probability calibration)
===========================================

Overview
--------
This project supports an optional post-fit calibration step that maps raw model win-probabilities
(`p_home_win`) to calibrated probabilities using either Platt scaling (logistic regression) or
isotonic regression. Calibration runs during backtests (walk-forward) and is fit only on prior
out-of-sample folds to avoid leakage.

CLI
---
Enable calibration in the backtest CLI using the `--calibrate` flag. Persisted calibrators can be
written with `--calib-dir`.

- `--calibrate`: enable post-fit probability calibration (default: off)
- `--calib-dir`: optional directory to persist fitted calibrator objects (joblib files)

Defaults & behavior
-------------------
- The runner selects the calibrator per-fold depending on training sample size:
  - `isotonic` when >= 500 training examples
  - `platt` when < 500 training examples
- Calibrators are fit on prior out-of-sample predictions collected by the walk-forward loop
  (i.e., previous test folds), guaranteeing no leakage from future data.
- When enabled, calibrated probabilities are added to the predictions frame as
  `p_home_win_calibrated`, and two metadata columns are added: `calibration_id` and
  `calibration_method`.

Persistence & outputs
---------------------
- Fitted calibrators are persisted to `<calib-dir>/<model>/<calibration_id>.joblib` when
  `--calib-dir` is provided.
- Per-fold calibration evaluation rows (brier/logloss comparisons) are written to
  `outputs/calibrators/<sport>/<season>/<model>/calibration_eval_<run_id>.(csv|json)`.

Notes
-----
- Calibration is best-effort: I/O or fitting failures are caught and do not fail the backtest.
- The default sample-size threshold (500) is conservative; you can adjust selection logic by
  modifying `src/backtest/runner.py` if needed.

If you'd like an opt-in registry for calibrator selection per model/sport, I can add a small
registry and a CLI flag to override the default selection.
