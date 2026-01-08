"""Odds parsing and EV calculations.

Utilities for parsing American odds, converting to implied probabilities,
and computing expected value (EV) given model probabilities.
"""

from __future__ import annotations

import math


def american_to_implied(odds: int | str) -> float:
    """Convert American odds (e.g., -150, +200 or "-150") to implied probability (0.0-1.0).

    Raises ValueError for odds == 0 or non-numeric values.
    """
    if isinstance(odds, str):
        s = odds.strip()
        s = s.replace("O", "0").replace("o", "0")
        s = s.replace("−", "-").replace("–", "-")
        try:
            odds = int(s)
        except Exception as exc:
            raise ValueError(f"Invalid odds: {odds}") from exc
    if odds == 0:
        raise ValueError("odds cannot be zero")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return -odds / (-odds + 100.0)


def payout_per_unit(odds: int | float) -> float:
    """Return net payout on a win per unit stake for American odds.

    e.g., odds +200 -> return 2.0 (win returns 2x stake profit), odds -150 -> return 0.666...
    """
    odds = int(odds)
    if odds > 0:
        return odds / 100.0
    else:
        return 100.0 / abs(odds)


def expected_value(implied_prob: float | None, model_prob: float, stake: float = 1.0, odds: int | None = None) -> float:
    """Compute EV per unit stake given model_prob.

    If `odds` provided (American), compute EV using American payout. If `implied_prob`
    provided and `odds` None, compute EV using implied probability as payout.
    Returns net EV (positive means expected profit per unit stake).
    """
    if odds is not None:
        payout = payout_per_unit(odds)
        return (model_prob * payout) - (1 - model_prob)
    if implied_prob is None or implied_prob <= 0:
        raise ValueError("implied_prob must be provided and > 0 when odds is not given")
    # payout approx is 1 / implied_prob - 1 (net payout), so EV = model_prob * payout - (1 - model_prob)
    payout = (1.0 / implied_prob) - 1.0
    return (model_prob * payout) - (1 - model_prob)


def roi_from_profit_and_stake(profit: float, stake: float) -> float | None:
    if stake == 0:
        return None
    return profit / stake
