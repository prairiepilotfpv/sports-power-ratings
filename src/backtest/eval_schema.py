"""Shared evaluation schema constants for backtests.

All evaluation frames should keep these columns even when values are missing
so merges/concats cannot drop required fields.
"""
from __future__ import annotations

from typing import Set

# Actual outcomes required for evaluation. Values may be missing but the
# columns must always exist.
REQUIRED_ACTUAL_COLS: Set[str] = {
    "home_score",
    "away_score",
    "home_win",
    "actual_margin",
    "actual_total",
}

# Prediction-side fields expected by metric computation.
REQUIRED_PRED_COLS: Set[str] = {
    "p_home_win",
    "pred_margin",
    "pred_total",
    "margin_sd",
    "total_sd",
    "total_mean",
    "margin_mean",
}

# Combined set used by backtest runner and helpers.
REQUIRED_EVAL_COLUMNS: Set[str] = REQUIRED_ACTUAL_COLS | REQUIRED_PRED_COLS
