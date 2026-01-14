Action paste parser

This tool converts the middle "Action" paste block into a CSV you can copy into the betting staging workflow.

Usage (recommended):

- Quick parse and CSV write:

```
python -m src.cli.action_paste --in markettest.txt --out outputs/paste_parsed/mymarkets.csv
```

- Also write opens JSON (optional):

```
python -m src.cli.action_paste --in markettest.txt --out outputs/paste_parsed/mymarkets.csv --include-opens-json opens.json
```

Notes:
- Output CSV columns match the staging expectations and produce exactly 6 rows per game in this order: ML away, ML home, spread away, spread home, total over, total under.
- ML rows use `line=0` to satisfy the import constraint.
- The parser is deterministic and will raise a `ValueError` if required tokens are missing for a matchup.
