# Bradley-Terry Heads

# Core
- Bradley-Terry is a W/L-only rating model: it fits team strengths so logistic probabilities between opponents match observed wins and losses.
- Predictions include the HFA logit and temperature scaling directly on the rating differential; no margin or total data feeds the core probability head.

# Canonical probability rule (bradley-terry)
- `model_p_home_win`, `p_home_win`, and `projected_win_prob` must always be the direct Bradley-Terry probability (ratings + HFA).
- `win_prob_source` must read `direct` whenever the BT logistic probability populates those fields.
- `normal_p_home_win` is the margin-normal approximation only; it may coexist in outputs but must never replace the canonical probability fields.

# Margin head
- `margin_mean` and `margin_sd` describe the forecasted margin distribution (home minus away).
- The implied spread equals `-margin_mean`; consumers can flip signs if they emit home/away spreads instead of margins.
- Normal-derived wins (via `normal_p_home_win`) come from the Gaussian assumption on the margin and must only be used when `win_prob_source` indicates `bt_margin_normal`.

# Total head
- `total_mean` and `total_sd` describe the expected combined points.
- These values come from the calibration that links the BT rating differential to total outcomes.

# Derived scores
- `projected_home_score = (total_mean + margin_mean)/2`
- `projected_away_score = (total_mean - margin_mean)/2`
- Derived scores maintain consistency with the margin and total heads even when optional fields are absent.
- Use them only as convenience: the margin/total heads remain the canonical distributional outputs.

Stay aligned with this contract when wiring schedule exports, matchups, backtests, and ensembles so every Bradley-Terry consumer sees a single authoritative probability.
