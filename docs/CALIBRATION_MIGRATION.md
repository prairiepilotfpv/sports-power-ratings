## Migration Guide: From Betting-Integrated to Standalone Calibration

### Current State
The calibration system has been completely refactored and is now **100% independent** from the betting pipeline.

### What Changed

#### Removed Dependencies
```python
# OLD (Betting-integrated):
from pipelines.history_calibration import calibrate_market_from_history
from pipelines.calibration_utils import select_calibrator
from data.bets_repository import load_bets_predictions_for_validation

# NEW (Standalone):
from calibration.historical_calibration import calibrate_sport_season
from calibration.distribution import MarginalDistributionCalibrator
```

#### Data Flow Changes
```
OLD: betting pipeline → bets_predictions table → calibrate-history
NEW: raw games → models → calibration system → calibrators (independent)
```

### Migration Path

#### For Command-Line Users

**OLD (Betting Pipeline)**:
```bash
python -m src.cli.pipeline calibrate-history \
    --sport nba \
    --season 2025-26 \
    --market-source ML=ensemble_ml_v1 \
    --market-source spread=ensemble_spread_v1 \
    --market-source total=ensemble_total_v1 \
    --method auto
```

**NEW (Standalone)**:
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
- Specify `--db` explicitly (not inferred from sport/season)
- Specify `--models` to ensemble (not source_id)
- Specify `--markets` as list (not market-source pairs)
- Use `--source-id` for grouping calibrations (not market sources)

#### For Python Code

**OLD (Betting Pipeline)**:
```python
from pipelines.history_calibration import calibrate_market_from_history
from markets.base import Market

path = calibrate_market_from_history(
    db_path="data/db/nba/2025-26.db",
    sport="nba",
    season="2025-26",
    market=Market.ML,
    source="ensemble_ml_v1",  # Betting source ID
    start_date="2025-01-01",
    end_date="2025-12-31",
)
```

**NEW (Standalone)**:
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

Key differences:
- Input: models to ensemble, not betting source IDs
- Output: (calibrator_object, path) tuples for all markets
- Calibrators created for multiple markets in one call
- More flexible and transparent data flow

### Breaking Changes

⚠️ **WARNING**: Old and new systems are incompatible

| Aspect | OLD | NEW |
|--------|-----|-----|
| Data source | `bets_predictions` table | `games` table |
| Schema dependency | Betting schema (selection, line) | None (raw game data only) |
| Market limitation | ML only | ML, SPREAD, TOTAL |
| Distribution support | No | Yes (for SPREAD/TOTAL) |
| Sport support | NBA-centric | All sports |
| CLI location | `pipeline calibrate-history` | `calibration.standalone_cli` |
| Output format | Single calibrator per call | Multiple calibrators per call |

### Why These Changes?

1. **Betting Pipeline Pollution**: Calibration had hard dependencies on betting schema
2. **Limited Market Support**: Only worked with binary probabilities
3. **Distribution Blindness**: Couldn't calibrate distributions (SPREAD/TOTAL)
4. **Sport Assumptions**: Hardcoded for NBA betting context
5. **Schema Brittle**: Any changes to bets_predictions broke calibration

### Benefits of New System

✅ **Completely Independent**: No betting pipeline dependencies
✅ **More Flexible**: Works with any model/ensemble combination
✅ **Distribution-Aware**: Proper calibration for SPREAD/TOTAL
✅ **Sport-Agnostic**: Works with NFL, MLB, NHL, any sport
✅ **Cleaner Architecture**: Single responsibility principle
✅ **Easier Testing**: No betting schema mocking needed
✅ **Better Diagnostics**: Per-market fit results

### Backward Compatibility Notes

The old `calibrate-history` command still exists in `pipeline.py` for backward compatibility:
```bash
python -m src.cli.pipeline calibrate-history [args]
```

However, it:
- ⚠️ Still depends on bets_predictions table
- ⚠️ Still only works with ML market
- ⚠️ Will be deprecated and removed
- ⚠️ Should not be used for new work

**Recommendation**: Use `calibration.standalone_cli` for all new calibration work.

### Testing Your Migration

```bash
# 1. Try the standalone CLI
python -m calibration.standalone_cli \
    --db data/db/nba/2025-26.db \
    --sport nba \
    --season 2025-26 \
    --models bradley-terry elo \
    --markets ML SPREAD \
    --source-id test

# 2. Verify outputs were created
ls -la outputs/calibrators/nba/2025-26/test/

# 3. Run test suite
python tests/test_standalone_calibration.py
```

### Troubleshooting

**Issue**: "ModuleNotFoundError: No module named 'calibration'"
```bash
# Solution: Ensure src is on path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python -m calibration.standalone_cli [args]
```

**Issue**: "Cannot import name 'calibrate_market_from_history'"
```bash
# Solution: Use new API
from calibration.historical_calibration import calibrate_sport_season
```

**Issue**: "No completed games found"
```bash
# Solution: Verify games table has scores
sqlite3 data/db/nba/2025-26.db \
    "SELECT COUNT(*) FROM games WHERE home_score IS NOT NULL"
```

**Issue**: "Unknown market: spread"
```bash
# Solution: Use uppercase or Market enum
# Correct: --markets ML SPREAD TOTAL
# Or in code: from markets.base import Market; Market.SPREAD
```

### Checklist for Migration

- [ ] Update any scripts calling `calibrate_market_from_history`
- [ ] Update any documentation referencing betting calibration
- [ ] Test new CLI on your data
- [ ] Verify calibrators are generated for all markets
- [ ] Compare old vs new calibrator quality (if data available)
- [ ] Update any automation/cron jobs
- [ ] Notify team of new workflow

### Future Cleanup

Once migration is complete, planned removals:

```python
# These can be removed after migration:
# - src/pipelines/history_calibration.py
# - src/pipelines/calibration_utils.py (betting-specific parts)
# - calibrate-history command from pipeline.py
# - Any betting pipeline references in calibration docs
```

### Questions?

Refer to:
1. `docs/CALIBRATION_STANDALONE.md` - System architecture
2. `CALIBRATION_IMPLEMENTATION.md` - Implementation details
3. `calibration/standalone_cli.py` - CLI reference
4. `tests/test_standalone_calibration.py` - Usage examples
