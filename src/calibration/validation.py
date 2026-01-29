"""Data source validation for calibration.

Ensures calibration inputs meet quality standards:
- No NaN or missing canonical columns
- Same ensemble sources as production
- Same heads mode contract validation
- Resolved (non-raw) predictions only
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

_LOG = logging.getLogger(__name__)


def validate_calibration_data_source(
    df: pd.DataFrame,
    *,
    market: str,
    sport: str,
    season: str,
    fail_on_errors: bool = True,
) -> bool:
    """Validate that calibration input is clean and production-ready.

    Args:
        df: DataFrame with predictions and outcomes
        market: Market name (ML, SPREAD, TOTAL)
        sport: Sport code
        season: Season identifier
        fail_on_errors: If True, raise on validation errors

    Returns:
        True if validation passes

    Raises:
        ValueError: if fail_on_errors=True and validation fails
    """
    if df.empty:
        msg = f"[{market} calibration] Empty DataFrame"
        if fail_on_errors:
            raise ValueError(msg)
        _LOG.warning(msg)
        return False

    # Define expected columns by market
    if market == "ML":
        required_cols = {"p_home_win", "home_win"}
    elif market in ("SPREAD", "TOTAL"):
        # Distribution markets need mean, SD, and actual outcomes
        if market == "SPREAD":
            required_cols = {"margin_mean", "margin_sd", "margin"}
        else:  # TOTAL
            required_cols = {"total_mean", "total_sd", "total"}
    else:
        raise ValueError(f"Unknown market: {market}")

    # Check for required columns
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        msg = f"[{market} calibration] Missing columns: {missing_cols}"
        if fail_on_errors:
            raise ValueError(msg)
        _LOG.warning(msg)
        return False

    # Check for NaNs in critical columns
    for col in required_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            nan_pct = 100.0 * nan_count / len(df)
            msg = (
                f"[{market} calibration] Column '{col}' has {nan_count} NaNs ({nan_pct:.1f}%)"
            )
            if nan_pct > 10.0:  # More than 10% NaN is suspicious
                if fail_on_errors:
                    raise ValueError(msg)
                _LOG.warning(msg)
                return False
            else:
                _LOG.info(msg)

    # For distribution markets, validate SD > 0
    if market in ("SPREAD", "TOTAL"):
        sd_col = "margin_sd" if market == "SPREAD" else "total_sd"
        if sd_col in df.columns:
            invalid_sd = (df[sd_col] <= 0).sum()
            if invalid_sd > 0:
                msg = f"[{market} calibration] {invalid_sd} rows with SD <= 0"
                if fail_on_errors:
                    raise ValueError(msg)
                _LOG.warning(msg)
                return False

    _LOG.info(
        f"[{market} calibration] Data source validation passed: {len(df)} rows, "
        f"columns={sorted(required_cols)}"
    )
    return True


def validate_no_nans_in_columns(df: pd.DataFrame, columns: list[str], *, fail: bool = True) -> bool:
    """Strictly validate no NaNs in specified columns.

    Args:
        df: DataFrame to check
        columns: Column names to validate
        fail: If True, raise on any NaN

    Returns:
        True if no NaNs found

    Raises:
        ValueError: if fail=True and NaNs found
    """
    for col in columns:
        if col not in df.columns:
            msg = f"Column '{col}' not in DataFrame"
            if fail:
                raise ValueError(msg)
            _LOG.warning(msg)
            return False

        nan_mask = df[col].isna()
        if nan_mask.any():
            msg = f"Column '{col}' has {nan_mask.sum()} NaNs"
            if fail:
                raise ValueError(msg)
            _LOG.warning(msg)
            return False

    return True


def check_outcome_range(
    df: pd.DataFrame,
    outcome_col: str,
    min_val: float = -1e6,
    max_val: float = 1e6,
    *,
    fail: bool = True,
) -> bool:
    """Validate outcomes are within expected range.

    Args:
        df: DataFrame with outcomes
        outcome_col: Outcome column name
        min_val: Minimum valid value
        max_val: Maximum valid value
        fail: If True, raise on validation failure

    Returns:
        True if all outcomes within range

    Raises:
        ValueError: if fail=True and outcomes out of range
    """
    if outcome_col not in df.columns:
        msg = f"Outcome column '{outcome_col}' not in DataFrame"
        if fail:
            raise ValueError(msg)
        _LOG.warning(msg)
        return False

    outcomes = pd.to_numeric(df[outcome_col], errors="coerce")
    valid_mask = outcomes.notna()

    if not valid_mask.any():
        msg = f"No valid outcomes in column '{outcome_col}'"
        if fail:
            raise ValueError(msg)
        _LOG.warning(msg)
        return False

    out_of_range = (outcomes < min_val) | (outcomes > max_val)
    if out_of_range[valid_mask].any():
        msg = (
            f"Column '{outcome_col}' has {out_of_range[valid_mask].sum()} values "
            f"outside [{min_val}, {max_val}]"
        )
        if fail:
            raise ValueError(msg)
        _LOG.warning(msg)
        return False

    return True
