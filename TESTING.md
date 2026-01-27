# Testing

Ensure dependencies are installed and your virtual environment is active.
On Windows (PowerShell): `./.pyenv/Scripts/Activate.ps1`.

## Install test tooling

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

## Run all tests

```bash
python -m pytest -q
```

## Calibration tests (optional dependency)

The calibration unit and integration tests require `scikit-learn`. If you want to run
these tests locally or in CI with full behavior, install scikit-learn:

```powershell
pip install scikit-learn
```

The test suite also includes a lightweight shim so tests will still run when
`scikit-learn` is not installed; the shim provides minimal, deterministic
behaviour for the calibration classes and is intended for CI environments
where installing additional dependencies are undesirable.

### Calibration Provenance Tags Tests

Tests for market-specific calibration provenance tags are in `tests/test_calibration_bets_integration.py`:

```bash
# Run all calibration provenance tests
python -m pytest tests/test_calibration_bets_integration.py::test_calibration_provenance_tags_* -v

# Run specific test
python -m pytest tests/test_calibration_bets_integration.py::test_calibration_provenance_tags_multiple_markets -v
```

These tests verify that:
- ML calibration appends `+calibrated_ml` to win_prob_source
- SPREAD calibration appends `+calibrated_spread` to win_prob_source
- TOTAL calibration appends `+calibrated_total` to win_prob_source
- Multiple market calibrations append all relevant tags
- Tags are idempotent (no duplicates)
- Missing win_prob_source column is handled gracefully

## Run fast unit tests only

```bash
pytest -q -m "not slow"
```

## Coverage

```bash
coverage run -m pytest -q
coverage report --fail-under=70
```

## Fast vs. slow tests

- **Fast tests** run quickly and avoid external dependencies.
- **Slow tests** are marked with `@pytest.mark.slow` (for example, tests that exercise larger datasets or heavier I/O).
  Use `-m "not slow"` to skip them.
