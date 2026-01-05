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
