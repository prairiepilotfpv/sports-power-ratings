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
