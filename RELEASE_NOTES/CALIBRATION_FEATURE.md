Calibration feature: summary and push instructions
===================================================

Summary
-------
This change introduces optional post-fit probability calibration for backtests.
Key points:
- Two calibrators implemented: Platt scaling (logistic) and Isotonic regression.
- Per-fold calibration is fit only on prior out-of-sample folds during walk-forward backtests.
- CLI flags: `--calibrate`, `--calib-dir`, and `--calibrator` (auto|platt|isotonic).
- Calibrators persisted when `--calib-dir` is provided.
- Per-fold calibration evals are written to `outputs/calibrators/<sport>/<season>/<model>/calibration_eval_<run_id>.(csv|json)`.
- Default registry mappings added: isotonic for pro leagues (NBA/NHL), platt for college (CBB/elo).

Branch & PR
-----------
Suggested branch name: `feature/calibration`
Suggested PR title: "Add post-fit probability calibration (Platt & Isotonic)"
Suggested PR body (short):
- Implements optional post-fit calibration during backtests.
- Adds CLI flags, registry, persistence, eval outputs, docs, and tests.
- Includes a lightweight test shim for environments without scikit-learn.

Local push steps
----------------
(perform these from your local clone)

```bash
# create branch
git checkout -b feature/calibration
# stage changes
git add src/backtest/runner.py src/cli/*.py src/pipelines/backtest.py src/calibration docs/RELEASE_NOTES/CALIBRATION_FEATURE.md tests/test_calibrators.py tests/test_calibration_integration.py TESTING.md
git commit -m "Add optional post-fit probability calibration (platt/isotonic), CLI flags, registry, persistence, docs, and tests"
# push branch
git push origin feature/calibration
# Open PR on GitHub with suggested title/body
```

CI note
-------
- For full calibration behavior in CI, add a job that installs `scikit-learn` (e.g., `pip install scikit-learn`) before running calibration tests. A test shim exists so the suite still runs when scikit-learn is absent, but the shim is lightweight and intended only for CI environments that prefer not to add heavy deps.

Night-close checklist
--------------------
- [ ] Ensure branch `feature/calibration` pushed.
- [ ] Open PR and add reviewers.
- [ ] Optionally add CI step to install `scikit-learn`.
- [ ] Draft release note excerpt (PR description includes summary above).

If you want, I can create a GitHub Actions workflow snippet (YAML) in `.github/workflows/` to install scikit-learn for CI runs — say the word and I’ll add it.
