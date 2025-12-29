# TODO
## ASAP: End-to-end daily pipeline (must-have)
- Provide a single pipeline to update the different leagues as needed with CSV files from Sports Reference and other sources, ie, a way to indiciate what sport and season with the relevent file to keep the databases up to date. 
- Ensure ingest accepts Sports-Reference CSV/text updates to refresh the database daily.
- Produce consistent outputs: team power rankings + daily matchups (including projections/spreads) from the current schedule.
- Add human-usable output: a clean Excel worksheet that shows the up-to-date calendar with projected matchups.

## Next: Model accuracy improvements
- Enhance Bradley-Terry to include home-court advantage.
- Incorporate MOV into the model (beyond post-fit scaling).

## Later: Additional models and betting utilities
- Add additional power ranking models using the same data source.
- Aggregate multiple model outputs into a combined rankings view (beyond Bradley-Terry).
- Add a pricing mechanism (e.g., "bet to win" target calculator).

## Documentation cleanup
- Create an exhaustive list of CLI commands and flags (e.g., --sport, --season, --input, --db).
