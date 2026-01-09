# CLV ingestion (closing lines)

Use the `clv-csv` betting subcommand to import closing lines/odds into the per-sport SQLite DB and backfill existing bets with CLV values. This keeps reports and PnL scenarios populated even when closing data arrives after bets are logged.

## Quick start

```bash
python -m src.cli.pipeline betting clv-csv \
  --sport nba --season 2025-26 \
  --csv data/raw/nba_closing_lines.csv \
  --default-market-type ML
```

- Requires either a `game_id` column or resolvable `team_home`/`team_away` + `game_date` columns for each row.
- By default, the command updates any matching bets with the latest CLV snapshot (disable with `--no-update-bets`).

## CSV schema

Minimal header (aliases allowed):

- `market_type` (aliases: none; `--default-market-type` can fill missing values)
- `selection`
- `close_line` (optional for moneyline)
- `close_odds` (or `odds`)
- `game_id` **or** `team_home`/`team_away` (aliases: `home_team`, `away_team`) and `game_date` (YYYY-MM-DD)
- `captured_at` (optional; ISO-8601; falls back to the CLI `--captured-at` or current UTC)

Rows with missing `selection`/`market_type` or invalid odds are rejected and reported in the summary output.

## Output

The CLI prints counts for inserted snapshots, rejected rows, and bets that were backfilled. CLV columns propagate into:

- `bets.clv_close_odds` / `bets.clv_close_line`
- `report` workbooks (CLV sheet and PnL scenarios)
