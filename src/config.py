from __future__ import annotations

# Logistic scale (points) used to convert projected spreads into win probabilities.
# The spread is defined as away_minus_home, so lower (more negative) values favor the home team.
DEFAULT_WIN_PROB_K = 6.566641127986305

# Rolling window (games) for empirical residual calibration of margin/total uncertainty.
CALIBRATION_RESIDUAL_GAMES = 300
MIN_CALIBRATION_SAMPLES = 25

# Safe global fallbacks when residual samples are insufficient to estimate spread/total variance.
DEFAULT_MARGIN_SD_FALLBACK = 12.0
DEFAULT_TOTAL_SD_FALLBACK = 20.0

# League-level default total mean used when insufficient data exists
# Conservative NBA baseline; used only as a structural fallback
DEFAULT_TOTAL_MEAN_FALLBACK = 220.0

# League-level default margin spread (NBA-oriented) used when residuals are missing or unstable.
# Chosen to be a conservative, realistic baseline rather than an artificially low variance.
LEAGUE_MARGIN_SD_DEFAULT = 13.5

# Prediction guardrails applied before EV/metrics calculations (NBA-specific but configurable).
MARGIN_SD_GUARDRAIL_MIN = 5.0
MARGIN_SD_GUARDRAIL_MAX = 30.0
TOTAL_SD_GUARDRAIL_MIN = 8.0
TOTAL_SD_GUARDRAIL_MAX = 35.0
PROJECTED_SCORE_MIN = 50.0
PROJECTED_SCORE_MAX = 170.0
PROJECTED_TOTAL_TOLERANCE = 2.0
