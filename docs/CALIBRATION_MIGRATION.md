## Migration Guide: Betting-Integrated to Standalone Calibration

### Current State (January 26, 2026)
The old betting-integrated calibration system has been removed. Use the standalone
calibration workflow:
- CLI: `python -m calibration.standalone_cli`
- API: `calibration.historical_calibration.calibrate_sport_season`

### What Changed
- Data source: `games` table only (no `bets_predictions` dependency)
- Markets: ML, SPREAD, TOTAL
- Distribution-aware calibration for SPREAD/TOTAL
- Sport-agnostic (NBA, NFL, MLB, NHL, etc.)

### Standalone CLI
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

Key differences:
- `--db` is required
- `--models` replaces betting-source IDs
- `--markets` is a list (not market=source pairs)
- `--source-id` groups calibration artifacts

### Standalone API
```python
from calibration.historical_calibration import calibrate_sport_season
from markets.base import Market

results = calibrate_sport_season(
    db_path="data/db/nba/2025-26.db",
    sport="nba",
    season="2025-26",
    models=["bradley-terry", "elo", "toor"],
    markets=[Market.ML, Market.SPREAD, Market.TOTAL],
    source_id="historical",
    method="auto",
    start_date="2025-01-01",
    end_date="2025-12-31",
)

for market_name, (calibrator, saved_path) in results.items():
    print(f"{market_name}: {saved_path}")
```

### Calibrator Output Layout
```
outputs/calibrators/
  └─ {sport}/
      └─ {season}/
          └─ {source_id}/
              ├─ ML/
              ├─ spread/
              └─ total/
```

### Troubleshooting

Issue: "ModuleNotFoundError: No module named 'calibration'"
```bash
# Ensure src is on path
set PYTHONPATH=%PYTHONPATH%;%CD%\\src
python -m calibration.standalone_cli [args]
```

Issue: "No completed games found"
```bash
sqlite3 data/db/nba/2025-26.db \
  "SELECT COUNT(*) FROM games WHERE home_score IS NOT NULL"
```

Issue: "Unknown market: spread"
```bash
# Use uppercase or Market enum
# Correct: --markets ML SPREAD TOTAL
```

### Checklist
- [ ] Update scripts to use `calibration.standalone_cli`
- [ ] Verify games table has completed scores
- [ ] Run `tests/test_standalone_calibration.py`
- [ ] Confirm calibrators exist in `outputs/calibrators/...`

### References
- `docs/CALIBRATION_STANDALONE.md`
- `CALIBRATION_IMPLEMENTATION.md`
- `src/calibration/standalone_cli.py`
