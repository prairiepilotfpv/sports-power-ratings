# Market Review CLI (retired)

`market-review` used to list and resolve staging rows, but it now immediately raises `ValueError` with guidance to use `betting market-csv`/`market_lines`. The command remains for reference but no longer performs staging review; instead, rely on the CSV import diagnostics and the `market_line_import_errors` table.
