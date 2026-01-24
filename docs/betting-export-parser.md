# Betting Export Parser

The betting export parser automatically matches bets from your betting app export to games in the database and assigns game IDs.

## Quick Start

### Single Sport
If you have a CSV with only NBA bets:
```bash
python -m src.cli.pipeline betting parse-export --csv data/raw/nba_bets.csv --sport nba
```

### Auto-Detect (Mixed Sports)
If you have a CSV with multiple sports (NBA, NHL, NCAAB, NCAAF):
```bash
python -m src.cli.pipeline betting parse-export --csv data/raw/all_bets.csv
```

The parser will automatically:
1. Detect which sports are in the CSV
2. Match bets to games in the corresponding database
3. Create separate output CSVs for each sport with `_with_ids` suffix:
   - `all_bets_nba_with_ids.csv`
   - `all_bets_nhl_with_ids.csv`
   - etc.

## What It Does

The parser:
1. **Reads** your betting app export CSV
2. **Normalizes** column names (handles special characters like `/` and spaces)
3. **Maps** team codes (e.g., "DAL", "SAC") to full team names using sport-specific mappings
4. **Looks up** games in the database by date and team names
5. **Adds** a `game_id` column to matched bets
6. **Outputs** a new CSV with game IDs populated where matches were found

## Output

The output CSV has all original columns plus a `game_id` column:
- **Matched rows**: Have a game_id value (e.g., `nba:2025-26:2025-12-27:1d407990241f`)
- **Unmatched rows**: Have `NaN` for game_id (game doesn't exist in database yet)

The parser reports:
```
Results by sport:
  NBA: 17/115 matched
    [WARNING] 93 unmatched
```

This means 17 out of 115 NBA bets found matching games.

## Why Bets Don't Match

Bets won't match if:
1. **Game not scheduled yet** - The game hasn't been entered into the database
2. **Team code not supported** - The team code isn't in the mapping for that sport
3. **Typo in team code** - The CSV has a misspelled team abbreviation

The parser automatically handles timezone differences: a bet timestamped at 00:00 UTC on Dec 30 for a game on Dec 29 local time will still match correctly (checks ±1 day).

## Supported Team Codes

### NBA (28 teams)
ATL, BOS, BRK, CHA, CHI, CLE, DAL, DEN, DET, GSW, HOU, IND, LAC, LAL, MEM, MIA, MIL, MIN, NOP, NYK, OKC, ORL, PHI, PHX, POR, SAC, SAS, TOR, UTA, WAS

### NCAAF (DRAFT)
Current mapping is a sample. Contact dev to extend.

### NCAAB
Mapped to "CBB" database.

### NHL
(Full mapping available in src/parsers/betting_app.py)

## Import to Database

After parsing, import matched bets into the database. Sport and season are auto-detected from the game_id column:

```bash
python -m src.cli.pipeline betting import-csv --csv data/raw/betshistory_nba_with_ids.csv
```

Only bets with a `game_id` will be imported. Unmatched bets are skipped.

If you want to override the sport or season:
```bash
python -m src.cli.pipeline betting import-csv \
  --csv data/raw/betshistory_nba_with_ids.csv \
  --sport nba \
  --season 2025-26
```

## Options

```
--csv PATH            Path to betting app export CSV (required)
--sport SPORT         Sport code (nba, nhl, ncaab, ncaaf)
                      If omitted, auto-detects from 'League' column
--season SEASON       Season code (default: 2025-26)
--db PATH             Override default database location
--output PATH         Override output directory (default: same as --csv)
```

## Workflow

1. **Export** bets from your betting app
2. **Run parser**: `python -m src.cli.pipeline betting parse-export --csv bets.csv`
3. **Review output**: Open `bets_nba_with_ids.csv`, check unmatched bets
4. **Import matched**: `python -m src.cli.pipeline betting import-csv --csv bets_nba_with_ids.csv`

The import command will automatically detect sport and season from the game_id column and import all matched bets.

