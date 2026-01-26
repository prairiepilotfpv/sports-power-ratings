## Calibration System Refactoring: Decoupling from Betting Pipeline

### Summary of Changes

The calibration system has been **completely decoupled from the betting pipeline**. It now operates as a standalone subsystem that:

1. **Works with raw historical game data only** - No references to `bets_predictions` table
2. **Supports all sports equally** - No hardcoded assumptions about NBA or any specific sport
3. **Supports all markets properly**:
   - ML: Uses probability calibrators (Platt scaling, isotonic regression)
   - SPREAD: Uses distribution-based calibrators for margin predictions
   - TOTAL: Uses distribution-based calibrators for total predictions
4. **Uses model outputs directly** - Generates predictions from models, not betting records
5. **Independent CLI** - Run calibration independently via `calibration.standalone_cli`

### Architecture Changes

#### Old System (Removed from Pipeline)
```
betting pipeline
  └─ calibrate-history
      └─ reads from bets_predictions table
      └─ assumes betting schema (selection, line, model_prob)
      └─ only handles ML market with probabilities
      └─ heavily NBA/betting-focused
```

#### New System (Standalone)
```
calibration subsystem (completely independent)
  ├─ calibration/historical_calibration.py
  │   ├─ load_completed_games() - from games table only
  │   ├─ generate_model_predictions() - calls models directly
  │   ├─ build_ml_calibration_dataset()
  │   ├─ build_spread_calibration_dataset() - NEW: distribution-based
  │   ├─ build_total_calibration_dataset() - NEW: distribution-based
  │   └─ calibrate_sport_season() - main entry point
  │
  ├─ calibration/distribution.py (NEW)
  │   ├─ MarginalDistributionCalibrator - for SPREAD/TOTAL
  │   └─ ProbabilityFromDistributionCalibrator - if needed
  │
  └─ calibration/standalone_cli.py (NEW)
      └─ Complete independent CLI, not integrated with pipeline.py
```

### Key Improvements

1. **Sport-Agnostic**
   - Works with any sport (NFL, MLB, NHL, etc.)
   - No hardcoded column names or logic
   - Uses Market enum for market types

2. **Market-Aware**
   - ML: Binary probability calibration
   - SPREAD: Distribution calibration (margin_mean, margin_sd)
   - TOTAL: Distribution calibration (total_mean, total_sd)

3. **No Betting Dependencies**
   - Doesn't read bets_predictions table
   - Doesn't depend on betting schema
   - Doesn't need market lines, selections, or betting context
   - Works purely on model outputs + game outcomes

4. **Distribution Support**
   - SPREAD/TOTAL use `MarginalDistributionCalibrator`
   - Fits α * predicted_mean + β transformation
   - Calibrates both mean and SD of predictions

### Usage

#### Via Standalone CLI
```bash
# Calibrate all markets for a sport/season
python -m calibration.standalone_cli \
    --db data/db/nba/2025-26.db \
    --sport nba \
    --season 2025-26 \
    --models bradley-terry elo toor \
    --markets ML SPREAD TOTAL \
    --source-id historical \
    --method auto

# Output:
# Fitted calibrators saved to:
#   outputs/calibrators/nba/2025-26/historical/ML/
#   outputs/calibrators/nba/2025-26/historical/spread/
#   outputs/calibrators/nba/2025-26/historical/total/
```

#### Programmatically
```python
from calibration.historical_calibration import calibrate_sport_season
from markets.base import Market

results = calibrate_sport_season(
    db_path='data/db/nba/2025-26.db',
    sport='nba',
    season='2025-26',
    models=['bradley-terry', 'elo'],
    markets=[Market.ML, Market.SPREAD, Market.TOTAL],
    source_id='historical',
    method='auto',
)

for market_name, (calibrator, saved_path) in results.items():
    print(f"{market_name}: {saved_path}")
```

### Data Flow

1. **Load historical games** (`games` table, completed only)
2. **Generate predictions** from models for each market
3. **Compute actual outcomes** from game scores
4. **Build calibration datasets**:
   - ML: (probability, binary_outcome)
   - SPREAD: (margin_mean, margin_sd, actual_margin)
   - TOTAL: (total_mean, total_sd, actual_total)
5. **Fit calibrators** using appropriate algorithm
6. **Save calibrators** to persistent storage (joblib format)

### File Organization

```
outputs/calibrators/
  └─ {sport}/
      └─ {season}/
          └─ {source_id}/
              ├─ ML/
              │   └─ {source_id}_20260126T120000Z.joblib
              ├─ spread/
              │   └─ {source_id}_20260126T120000Z.joblib
              └─ total/
                  └─ {source_id}_20260126T120000Z.joblib
```

### No Betting Pipeline Integration

The `calibrate-history` command in `pipeline.py` is **kept for backward compatibility only**. 
It now delegates to the standalone system but should not be used for new work.

Users should instead use:
```bash
python -m calibration.standalone_cli [args]
```

### Removal from Pipeline (Planned)

Future cleanup:
- Remove `calibrate-history` from `pipeline.py` main CLI
- Remove `pipelines/history_calibration.py` (deprecated)
- Remove `calibration_utils.py` dependency from pipeline
- Update documentation to direct users to standalone CLI

### Testing

The standalone system should be tested across:
- [ ] NBA (done implicitly)
- [ ] NFL
- [ ] MLB
- [ ] NHL
- Any sport in the database

No betting-specific test data required; works with any `games` table data.

### Limitations and Future Work

1. **Calibrator Application**: Generated calibrators are not yet automatically applied to predictions
   - Separate module needed to load + apply calibrators to new predictions
   - Can be used in model output or post-processing, not in betting pipeline

2. **Retrospective Calibration**: Cannot calibrate on same data used for fitting
   - Would require cross-validation or held-out sets
   - Current implementation uses all available data (appropriate for historical baseline)

3. **Multiple Sports/Seasons**: Currently process one sport/season at a time
   - User can run script in batch mode if needed

4. **Ensemble Coordination**: Calibrators are per-model or per-ensemble
   - Future: might support per-market ensemble calibration separate from model tuning
