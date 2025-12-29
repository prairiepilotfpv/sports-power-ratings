# TODO
## ASAP: End-to-end daily pipeline (must-have)
- Ensure ingest accepts Sports-Reference CSV/text updates to refresh the database daily.
- Produce consistent outputs: team power rankings + daily matchups (including projections/spreads) from the current schedule.
- Add human-usable output: a clean Excel worksheet that shows the up-to-date calendar with projected matchups.

## Next: Model accuracy improvements
- Enhance Bradley-Terry to include home-court advantage.
- Incorporate MOV into the model (beyond post-fit scaling).

## Later: Additional models and betting utilities
- Add additional power ranking models using the same data source.
- Add a pricing mechanism (e.g., "bet to win" target calculator).

## Documentation cleanup
- Create an exhaustive list of CLI commands and flags (e.g., --sport, --season, --input, --db).


- if the update is ran multiple times with no new data, it should return the same data, no doubling up or
- add rank column to excel output
- make sure I don't need the "import sports reference" sectoin as it's kind of outdated. 
- remove "notes" from schedule output
- remove "model" from schedule output - will either run model by model or use aggregated overview
- explain what "home_rating" and "away_rating" mean?
- projected spread still incorrect: need to make sure system is generating projected scores and creating a projected total
- make sure that the calender is updated so that played games go from "scheduled" to "final" when updated
- make sure that the calender updates home and away points as it updates as well. 
- make sure it's updating properly and I'm running the correct command to update it
