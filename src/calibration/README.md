# Calibration Package

Standalone calibration utilities for model predictions. This package is
independent from the betting pipeline and works directly from completed games
plus model backtest predictions. Ensemble membership/weights are read from the
ensemble config files under `outputs/ensembles/`.

## Quick Start (CLI)

```bash
python -m calibration.standalone_cli \
  --db data/db/nba/2025-26.db \
  --sport nba \
  --season 2025-26 \
  --models bradley-terry elo toor \
  --markets ML SPREAD TOTAL \
  --source-id historical \
  --method auto
```

## Programmatic

```python
from calibration.historical_calibration import calibrate_sport_season
from markets.base import Market

results = calibrate_sport_season(
    db_path="data/db/nba/2025-26.db",
    sport="nba",
    season="2025-26",
    models=["bradley-terry", "elo"],
    markets=[Market.ML, Market.SPREAD, Market.TOTAL],
    source_id="historical",
    method="auto",
)
```

## Output Layout

```
outputs/calibrators/{sport}/{season}/{source_id}/{market}/
```

## Validation Integration

`python -m src.cli.pipeline validation-report` now checks calibration artifacts
and optionally runs calibration to ensure each market has a current calibrator.
If backtest start/end dates are provided in validation, calibration defaults to
the same window unless explicitly overridden.

## Key Modules

- `historical_calibration.py`: main workflow and dataset builders
- `distribution.py`: distribution calibration for SPREAD/TOTAL
- `standalone_cli.py`: CLI entry point
- `io.py`: load/save helpers
