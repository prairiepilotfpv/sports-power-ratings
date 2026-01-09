# Market Review CLI

Review and resolve OCR staging rows created by `market-ocr` or CSV imports. The command works against the per-sport/per-season SQLite DB and lets you list pending rows and accept or reject matches.

## Listing rows

- Default shows only `needs_review` rows:  
  `python -m src.cli.pipeline market-review --sport nba --season 2025-26`
- Show all match statuses:  
  `python -m src.cli.pipeline market-review --sport nba --season 2025-26 --status all`
- Limit output:  
  `python -m src.cli.pipeline market-review --sport nba --season 2025-26 --limit 20`

Each row prints `id`, `match_status`, `match_confidence`, teams, market, odds/line, book, captured_at, and current `game_id` (if any).

## Accepting or rejecting

- Accept and persist a match (requires a `game_id`):  
  `python -m src.cli.pipeline market-review --sport nba --season 2025-26 --accept 12 --game-id 2024-12-01-lal-lac --match-confidence 0.95`
- Reject and clear a match:  
  `python -m src.cli.pipeline market-review --sport nba --season 2025-26 --reject 12`

Notes:
- `match_confidence` defaults to 1.0 when accepting and 0.0 when rejecting if not provided.
- `game_id` should already exist in the sport/season DB; the command does not auto-create games.
- After acceptance, rows are marked `matched` and become eligible for `market-commit`.
