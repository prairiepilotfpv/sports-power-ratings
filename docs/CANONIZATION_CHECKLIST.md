# System Canonization Checklist

This document ensures the sports power ratings system remains stable and robust as you:
- Add/remove models
- Add new sports
- Change ensemble weighting
- Update tuning parameters

## ✅ Canonization Status

### Core Invariants (Protected by Tests)

| Invariant | Test Coverage | Status |
|-----------|--------------|--------|
| Game ID consistency across all import paths | `test_game_id_consistency_across_sources` | ✅ |
| Ensemble application for all 3 markets | `test_ensemble_application_all_markets` | ✅ |
| Market parameter isolation | `test_market_parameter_isolation` | ✅ |
| Source labeling correctness | `test_source_labeling_correctness` | ✅ |
| Schedule column contract | `test_schedule_column_contract` | ✅ |
| Backtest prediction contract | `test_backtest_prediction_contract` | ✅ |
| Full pipeline flow | `test_full_pipeline_flow_nba` | ✅ |

### Import & Export Contracts

| Contract | Implementation | Status |
|----------|---------------|--------|
| Hash-based game IDs | `src/utils/game_id.py::make_game_id` | ✅ |
| Sports-Reference uses canonical IDs | `src/ingest/sports_reference.py` | ✅ |
| Schedule columns match contract | `SCHEDULE_EXPORT_COLUMNS` | ✅ |
| Ensemble classes all imported | `src/pipelines/schedule.py` | ✅ |

### Model Canonization (Per Model)

| Model | Contracts A/B/C | Regression Tests | Tune-Model Works | Status |
|-------|----------------|------------------|------------------|--------|
| Elo | ✅ | ✅ | ✅ | Canonized |
| Bradley-Terry | ✅ | ✅ | ✅ | Canonized |
| Poisson | ✅ | ✅ | ✅ | Canonized |
| GSSD | ✅ | ✅ | ✅ | Canonized |
| TOOR | ✅ | ✅ | ✅ | Canonized |

---

## 🔧 How to Maintain Canonization

### When Adding a New Model

1. **Implement core interfaces:**
   ```python
   class MyModel(BaseModel):
       def fit(self, games_df: pd.DataFrame) -> None: ...
       def predict(self, home_team: str, away_team: str) -> GamePrediction: ...
       def metadata(self) -> ModelMetadata: ...
   ```

2. **Register the model:**
   ```python
   # In src/models/registry.py
   register_model("my-model", MyModel)
   ```

3. **Add projection engine (if needed):**
   ```python
   # In src/forecasting/projection_engines.py
   def my_model_projection_engine(...) -> dict: ...
   register_projection_engine("my-model", my_model_projection_engine)
   ```

4. **Run canonization tests:**
   ```bash
   pytest tests/test_pipeline_canonization.py -v
   pytest tests/models/test_my_model.py -v
   ```

5. **Smoke test the full pipeline:**
   ```bash
   python -m src.cli.pipeline rank --sport nba --season 2024-25 --model my-model
   python -m src.cli.pipeline schedule --sport nba --season 2024-25 --model my-model
   python -m src.cli.pipeline tune-model --model my-model --csv test.csv --activate
   ```

### When Adding a New Sport

1. **Import schedule/results:**
   ```bash
   python -m src.cli.pipeline import --sport mlb --season 2025 --input data/raw/mlb.csv
   ```

2. **Verify game IDs use canonical format:**
   ```bash
   python -c "from data.repository import load_games; games = load_games('data/db/mlb/2025.db', 'mlb', '2025'); print(games[0].game_id)"
   # Should print: mlb:2025:YYYY-MM-DD:hash12
   ```

3. **Run full pipeline:**
   ```bash
   python -m src.cli.pipeline rank --sport mlb --season 2025
   python -m src.cli.pipeline schedule --sport mlb --season 2025
   ```

4. **Verify ensemble system works:**
   - Check BETS sheet has `win_prob_source`, `spread_source`, `total_source` populated
   - Verify no "direct+ensemble" concatenation
   - Confirm 3 rows per game (ML, SPREAD, TOTAL)

### When Changing Ensemble Weights

1. **Update ensemble config files:**
   ```json
   // outputs/ensembles/nba/2025-26/ML/ensemble_ml_v1.json
   {
     "ensemble_id": "ensemble_ml_v1",
     "models": ["elo", "bradley-terry", "poisson"],
     "weights": {"elo": 0.3, "bradley-terry": 0.4, "poisson": 0.3}
   }
   ```

2. **Verify schedule uses new weights:**
   ```bash
   python -m src.cli.pipeline schedule --sport nba --season 2025-26
   ```
   
   Check META sheet for `ensemble_config_sha256` - it should change when weights change.

3. **Verify BETS sheet shows ensemble sources:**
   ```python
   import pandas as pd
   df = pd.read_excel('schedule.xlsx', sheet_name='BETS')
   print(df[df['market_type']=='ML']['win_prob_source'].unique())
   # Should show: ['ensemble_ml_v1']
   ```

### When Tuning Parameters

1. **Tune all markets separately:**
   ```bash
   python -m src.cli.pipeline tune-model --model elo \
     --csv nba_results.csv --start 2024-11-01 --end 2025-01-01 \
     --markets ML SPREAD TOTAL --activate
   ```

2. **Verify isolation in database:**
   ```sql
   SELECT model, market, params_json 
   FROM model_market_active_params 
   WHERE sport='nba' AND season='2025-26' AND model='elo';
   ```
   
   Should see 3 rows with different `params_json` per market.

3. **Verify schedule uses correct params:**
   ```bash
   python -m src.cli.pipeline schedule --sport nba --season 2025-26 --model elo
   ```
   
   Check model sheet - `params_market` column should show ML/SPREAD/TOTAL correctly assigned.

---

## 🚨 Red Flags (Things That Break Canonization)

### Immediate Action Required

- ❌ **Missing ensemble imports** → All three ensemble classes must be imported in `schedule.py`
- ❌ **Game ID format mismatch** → All imports must use `make_game_id()`, not custom formats
- ❌ **Source concatenation** → No "direct+ensemble" or "sample+calibrated" concatenations
- ❌ **Metric dropout** → `pred_margin` and `pred_total` must be populated in backtest predictions
- ❌ **Market param leakage** → ML params used for SPREAD/TOTAL projections

### Warnings (Fix When Possible)

- ⚠️ **Tests failing** → Run `pytest tests/test_pipeline_canonization.py` after any change
- ⚠️ **Column schema drift** → Schedule columns don't match `SCHEDULE_EXPORT_COLUMNS`
- ⚠️ **Missing model metadata** → Model doesn't return `ModelMetadata` from `metadata()`

---

## 📋 Daily Verification Checklist

Run these commands to verify system health:

```bash
# 1. Run canonization tests
pytest tests/test_pipeline_canonization.py -v

# 2. Check for duplicate games (should be 0)
python -c "import sqlite3; conn = sqlite3.connect('data/db/nba/2025-26.db'); cursor = conn.cursor(); cursor.execute('SELECT COUNT(*) FROM (SELECT date, home_team, away_team, COUNT(*) as cnt FROM games GROUP BY date, home_team, away_team HAVING cnt > 1)'); print('Duplicates:', cursor.fetchone()[0])"

# 3. Verify ensemble imports work
python -c "from ensemble.ml_v1 import MLWeightedAverageEnsemble; from ensemble.spread_v1 import SpreadWeightedAverageEnsemble; from ensemble.total_v1 import TotalWeightedAverageEnsemble; print('✅ All ensemble classes importable')"

# 4. Check schedule output format
python -m src.cli.pipeline schedule --sport nba --season 2025-26 --output tmp/verify.xlsx
python -c "import pandas as pd; from pipelines.schedule import SCHEDULE_EXPORT_COLUMNS; df = pd.read_excel('tmp/verify.xlsx', sheet_name='bradley-terry', skiprows=2); print('✅ Columns match' if list(df.columns) == SCHEDULE_EXPORT_COLUMNS else '❌ Column mismatch')"
```

---

## 📚 Reference Documentation

- **Model Canonization**: `docs/model_canonization_playbook.md`
- **Pipeline CLI**: `docs/CLI.md`
- **Daily Workflow**: `docs/daily-workflow.md`
- **Contracts Module**: `src/contracts.py`
- **Game ID Utilities**: `src/utils/game_id.py`
- **Test Suite**: `tests/test_pipeline_canonization.py`

---

## 🎯 Definition of "Canonized"

The system is considered **canonized** when:

1. ✅ All integration tests pass (`test_pipeline_canonization.py`)
2. ✅ All models pass contracts A/B/C (see `model_canonization_playbook.md`)
3. ✅ Game IDs are consistent across all import paths
4. ✅ Ensembles apply correctly to all 3 markets
5. ✅ Market parameters are isolated (no leakage)
6. ✅ Source labels are accurate (no concatenation bugs)
7. ✅ Schedule output matches `SCHEDULE_EXPORT_COLUMNS` contract
8. ✅ Full pipeline (import → rank → schedule → backtest) runs without errors
9. ✅ Zero duplicate games in database
10. ✅ Tuning works for all markets (`tune-model --activate`)

**Last Verified**: January 23, 2026
**System Version**: v1.0 (Post NHL Duplicate Fix & Ensemble Import Fix)
