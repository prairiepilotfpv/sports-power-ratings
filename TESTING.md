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
where installing additional dependencies is undesirable.


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
