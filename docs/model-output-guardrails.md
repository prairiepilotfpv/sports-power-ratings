# Model Output Safety & EV Guardrails

What changed
- Centralized prediction validation in `src/eval/validation.py` (defaults tuned for NBA).
- EV aggregation helper in `src/eval/evaluator.py` that applies validation, model weights, and Poisson divergence guard before computing probabilities/EV.
- Backtests now drop invalid prediction rows before metric aggregation.

Validator checks (defaults)
- `margin_sd` in [5, 30]; `total_sd` in [8, 35].
- Projected scores in [50, 170]; projected totals must match score sum within ±2.
- Win-probability fields must be within [0, 1].
- `margin_dist_assumption` must be one of `normal_approx|empirical|none`.
- Invalid rows surface reasons such as `invalid_margin_sd`, `invalid_total_sd`, `missing_score`, `prob_out_of_bounds`, `total_inconsistent`.

EV aggregation rules
- Default weights: spreads/moneyline -> BT/ELO/GSSD = 1, Poisson = 0, TOOR = 0. Totals -> BT/ELO/GSSD = 1, Poisson = 0.25, TOOR = 0.
- Poisson divergence guard: if Poisson total differs from the median of other models by >8 points, Poisson weight is cut to 0.1 for that game/total.
- Spread cover probabilities are priced from `margin_mean` + `margin_sd`; `projected_spread` is ignored by design.
- Debug mode returns per-model probabilities, weights, breakeven probability, and EV; optional `debug_output_path` writes a CSV.

How to use
```python
from eval.evaluator import evaluate_market_rows

opps_df, debug_df = evaluate_market_rows(predictions_df, markets_df, debug=True)
```
- `predictions_df` should contain the schedule/export columns (margin_mean, margin_sd, total_mean, total_sd, model_p_home_win, projected_home_score/away_score, etc.).

Projections & exporters
- Recommendation: projection authors and exporter code should populate the canonical projection columns at the top level of the prediction row: `projected_home_score`, `projected_away_score`, `projected_total`, `margin_mean`, `margin_sd`, `total_mean`, `total_sd`, `model_p_home_win`, `normal_p_home_win`, `win_prob_source`, and `margin_dist_assumption`.
- Rationale: downstream guardrails and tuning expect these top-level fields. Storing projections only inside `extra` (or a serialized blob) can lead to validation mismatches or accidental exclusion of model outputs. The test `tests/test_extra_projection_expansion.py` ensures `extra`-only projections are promoted and validated; prefer emitting top-level fields to avoid needing promotions.
- `markets_df` should contain game_id, market_type (moneyline/spread/total), selection, line, odds, and home/away team labels for spread/ML routing.
- Pass `include_excluded_reason=True` to stamp rows that could not be priced.

TOOR margin_sd note
- The intermittent `margin_sd=1.0` cases originate from the stored `margin_std` metric feeding the schedule projection context; when that metric is learned as ~1 from small samples, it propagates to exports. The new validator excludes sub-5.0 rows so they cannot affect EV/backtests.
