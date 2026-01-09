# Bet evaluation guardrails and controls

This repo now enforces a small safety net before any bets or EV summaries touch live predictions.

## Guardrail layer

- Predictions with a `margin_sd` outside the configured range (default 5–30 points) are filtered before any metrics, reports, or review workbooks are built. This protects aggregation layers from models that under- or over-state uncertainty.
- The new helper in `src/pipelines/guardrails.py` is reused by the backtester and can be applied to other betting pipelines by passing `sport` ("nba" by default) or setting `enabled_sports=None` to apply globally.
- Guardrail hits emit a log message with the number of excluded rows so you can audit why a prediction was skipped.

## TOOR variance fix

- The TOOR model now clamps its `margin_sd` output to the minimum guardrail value (or the default fallback when no residuals are available) so those predictions survive the filter.
- This avoids a common scenario where tiny residuals (<5 points) previously got filtered out, resulting in empty review runs or backtests.

## Tests

- `make test -k guardrails` covers the shared helper and its NBA/default behavior.
- `make test -k toor_margin_sd` proves TOOR respects the safety floor before predictions are emitted.

Add a reference to this doc when explaining the betting review/workbook workflow so downstream consumers know why certain rows disappear from EV sheets.
