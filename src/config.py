from __future__ import annotations

# Logistic scale (points) used to convert projected spreads into win probabilities.
# The spread is defined as away_minus_home, so lower (more negative) values favor the home team.
DEFAULT_WIN_PROB_K = 6.566641127986305

# Rolling window (games) for empirical residual calibration of margin/total uncertainty.
CALIBRATION_RESIDUAL_GAMES = 300
MIN_CALIBRATION_SAMPLES = 25

# Poisson overdispersion (κ) default for analytic head variance.
DEFAULT_POISSON_OVERDISPERSION = 1.0

# Variance calibrator constraints and regularization defaults
VARIANCE_CALIBRATION_C_MIN = 0.6
VARIANCE_CALIBRATION_C_MAX = 1.6
VARIANCE_CALIBRATION_TAU_MAX = 10.0
VARIANCE_CALIBRATION_LAMBDA_C = 0.1
VARIANCE_CALIBRATION_LAMBDA_TAU = 0.01
VARIANCE_CALIBRATION_CLIP_RATE_THRESHOLD = 0.05

# Safe global fallbacks when residual samples are insufficient to estimate spread/total variance.
DEFAULT_MARGIN_SD_FALLBACK = 12.0
DEFAULT_TOTAL_SD_FALLBACK = 22.0  # Increased from 20.0 to reduce over-confidence in BETS by flattening normal dist

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

# NHL tie-mass defaults for Poisson model
# When insufficient OT data exists, fall back to these values
NHL_OT_HOME_WIN_FALLBACK = 0.5
NHL_OT_MIN_SAMPLES = 20  # Minimum OT games required for empirical estimate

# Health gate thresholds for uncertainty calibration
CLIP_RATE_WARN_THRESHOLD = 0.20
CLIP_RATE_ERROR_THRESHOLD = 0.50
# Formal heads system feature flag.
# When enabled, uses explicit heads framework for model derivations (Elo, TOOR, GSSD).
# When disabled (default), uses legacy projection engine derivation.
# Cannot be enabled simultaneously with legacy projection engine derivation.
HEADS_MODE_ENABLED = False

# ===== PHASE 6: ENSEMBLE GOVERNANCE =====
# Per-market model allowlists: single source of truth for which models are used in each market.
# These are used by:
# - forecast generation (which models are built)
# - ensemble weight resolution (which models are considered)
# - validator/audit reporting

MARKET_MODEL_ALLOWLISTS: dict[str, list[str]] = {
    "ML": ["bradley-terry", "elo", "gssd", "poisson", "toor"],
    "SPREAD": ["bradley-terry", "elo", "gssd", "poisson", "toor"],
    "TOTAL": ["bradley-terry", "elo", "gssd", "poisson", "toor"],
}

# Weight governance policy constants
# MIN_WEIGHT_EPS: weights below this are clamped to 0 then renormalized
MIN_WEIGHT_EPS: float = 0.01

# MIN_NEFF: minimum effective model count (Neff = 1 / sum(w_i^2))
# If Neff < MIN_NEFF in strict mode, raise RuntimeError instead of silently falling back
MIN_NEFF: float = 1.5

# ENSEMBLE_STRICT_MODE: when True, fail instead of silently degrading
# (e.g., weight collapse, Neff below threshold, insufficient model coverage)
# When False, allow fallback to uniform weights but log loudly
ENSEMBLE_STRICT_MODE: bool = False