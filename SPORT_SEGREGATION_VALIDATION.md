# Sport Segregation Validation

## Overview

The betting import system validates that **each CSV file contains bets for a single sport only**. This prevents accidental mixing of NBA and NHL bets (or other sports) in the same database.

## Why This Matters

- **Database Structure**: Each sport has its own database: `data/db/nba/2025-26.db`, `data/db/nhl/2025-26.db`, etc.
- **Game Resolution**: Game IDs are resolved using the sport code from the CSV, so mixed sports would cause game lookup failures or mismatches
- **Data Integrity**: Keeping sports segregated ensures clean, auditable records

## How It Works

### Validation Logic

When you run `betting import-csv`, the system:

1. **Reads the CSV** and extracts all unique `league` values
2. **Normalizes** them to lowercase (e.g., `NBA` → `nba`)
3. **Compares** against the `--sport` argument provided on the command line
4. **Rejects** the import if any mismatch is found

### Example: Correct Usage

```powershell
# ✅ This succeeds because the CSV contains only NBA bets
python -m src.cli.pipeline betting import-csv \
  --csv nba_history.csv \
  --sport nba \
  --season 2025-26
```

CSV file (`nba_history.csv`):
```
league,start_time,game,type,odds,odds_spread_total,result,units_wagered
nba,2026-01-08T00:00:00.000Z,WAS @ PHI,spread_away,-110,12.5,loss,1.1
nba,2026-01-08T00:00:00.000Z,TOR @ CHA,ml_away,-135,-135,win,1.35
```

### Example: Rejected Usage

```powershell
# ❌ This fails because the CSV contains both NBA and NHL bets
python -m src.cli.pipeline betting import-csv \
  --csv mixed_sports.csv \
  --sport nba \
  --season 2025-26
```

**Error Output:**
```
ValueError: CSV contains bets for mismatched sports: ['nhl']. 
Target database is for sport 'nba'. 
Each import should use a single sport; separate by sport and re-run import.
```

CSV file (`mixed_sports.csv`):
```
league,start_time,game,type,odds,odds_spread_total,result,units_wagered
nba,2026-01-08T00:00:00.000Z,WAS @ PHI,spread_away,-110,12.5,loss,1.1
nhl,2026-01-08T02:30:00.000Z,OTT @ UTA,ml_home,-125,-125,win,1.25
```

## Solution: Split by Sport

If your betting app exports a mixed-sport history, split it before importing:

```powershell
# Extract only NBA rows to nba_history.csv
# Extract only NHL rows to nhl_history.csv

# Then import separately
python -m src.cli.pipeline betting import-csv --csv nba_history.csv --sport nba --season 2025-26
python -m src.cli.pipeline betting import-csv --csv nhl_history.csv --sport nhl --season 2025-26
```

## Key Guarantees

✅ **Sport Segregation**: NHL bets go only to `data/db/nhl/2025-26.db`  
✅ **Sport Segregation**: NBA bets go only to `data/db/nba/2025-26.db`  
✅ **No Mixing**: The system rejects any attempt to mix sports in a single import  
✅ **No New DBs**: No new databases are created; uses existing database for sport/season  

## Implementation Details

**Code Location**: `src/data/betting_repository.py`, lines 320-330

```python
# Validate that the CSV sport matches the target database sport
csv_leagues = df['league'].str.lower().str.strip().unique()
csv_leagues = [lg for lg in csv_leagues if lg and not pd.isna(lg)]

if csv_leagues:
    mismatched = [lg for lg in csv_leagues if lg != sport.lower()]
    if mismatched:
        raise ValueError(
            f"CSV contains bets for mismatched sports: {mismatched}. "
            f"Target database is for sport '{sport}'. "
            f"Each import should use a single sport; separate by sport and re-run import."
        )
```

## Test Coverage

A dedicated test validates this behavior:

```bash
pytest tests/test_import_mixed_sports.py::test_import_bets_csv_rejects_mixed_sports -v
```

This test:
- Seeds both NBA and NHL games
- Creates a CSV with mixed sports
- Attempts import to NBA database
- Verifies the ValueError is raised
- Confirms no bets were inserted to the database
