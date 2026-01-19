# Schedule workbook markets (README)

This update makes the schedule Excel export market-aware while keeping one sheet per model.

- Sheets: every model sheet now repeats each game three times, once per `params_market` in the order ML, SPREAD, TOTAL. Sorting and grouping respect this order.
- Dashboard: also carries three rows per game/model with `params_market` placed immediately after `date` to keep markets visually grouped.
- Metadata: model sheet metadata includes a `prediction_hash` for the multi-market sheet plus market-prefixed keys (e.g., `ML.params_source`). Data starts below row 50 to give the larger metadata block room.
- Bets: BETS logic and ensemble sourcing are unchanged; it still consumes the ML/SPREAD/TOTAL forecast sources directly, not the dashboard/model sheets.
- Row counts: expect roughly 3x more rows on schedule and dashboard sheets; use `params_market` to filter if you only want ML.
