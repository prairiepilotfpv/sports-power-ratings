## Standalone Calibration System

The calibration system is fully independent of the betting pipeline. It:
- Uses only the `games` table (completed games)
- Supports ML, SPREAD, and TOTAL markets
- Calibrates distributions for SPREAD/TOTAL
- Works across sports and seasons

### Architecture
```
games table (completed)
  -> model backtest predictions
  -> calibration datasets per market
  -> calibrators saved to outputs/calibrators/<sport>/<season>/<source_id>/<market>/
```

### CLI Usage
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

### Programmatic Usage
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

### Output Layout
```
outputs/calibrators/
  └─ {sport}/{season}/{source_id}/{market}/
```

### Notes
- Calibration uses walk-forward backtest predictions (no leakage).
- ML uses Platt/Isotonic; SPREAD/TOTAL use distribution calibration.
