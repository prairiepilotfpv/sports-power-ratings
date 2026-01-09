# Market Bets Logger

This lightweight CLI turns reviewed staging rows into `bets` entries with stake presets and duplicate safeguards. Use it after resolving matches via `market-review`.

## Workflow
1. Run OCR or CSV import so rows land in `market_snapshot_staging`.
2. Resolve matches with `market-review` (accept/reject) so rows are `matched` and have a `game_id`.
3. Pivot into bets:
   ```bash
   python -m src.cli.pipeline market-bets --sport nba --season 2025-26 --stake-preset unit --unit-stake 1.0
   ```
4. Optionally run `betting report` or `betting settle-bets` later.

## Options
- `--status`: match_status filters (default `matched`; use `all` for everything).
- `--review-run-id`: attach to bets (defaults to `staging-<timestamp>`).
- `--stake-preset`: `half`, `unit`, `double` multipliers applied to `--unit-stake`.
- `--default-book`: fallback when staging rows lack `book`.
- `--disable-auto-hold`: turn off duplicate detection.
- `--dry-run`: summarize without writing.

## Auto-hold detection
The logger checks for duplicate markets from the same `image_path` with the same market/selection. The first instance is kept; subsequent duplicates are tagged with `hold_reason="duplicate_in_image"` on the staging row and skipped from insertion. Rows without an `image_path` are never auto-held. This prevents double-logging when OCR emits repeated lines from one screenshot.

## CLV backfill
If a closing-line snapshot exists for the same `game_id`/`market_type`/`selection`, the inserted bet is populated with `clv_close_odds` and `clv_close_line`. To ingest closing lines after bets are logged, use `python -m src.cli.pipeline betting clv-csv ...` (see [docs/market-clv.md](docs/market-clv.md)).
