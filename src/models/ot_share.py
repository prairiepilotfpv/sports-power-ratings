"""Empirical overtime home-win share for Poisson tie-mass adjustment.

This module provides the empirical home-win share from overtime/shootout games
for sports like NHL where ties don't exist in final outcomes but can occur
in regulation. The tie mass from Skellam distribution is split using this
empirical rate rather than a fixed 50-50 split.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from functools import lru_cache
from pathlib import Path
from typing import Tuple

from config import NHL_OT_HOME_WIN_FALLBACK, NHL_OT_MIN_SAMPLES

_LOG = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def get_ot_home_win_share(
    db_path: str | Path,
    sport: str,
    season: str,
) -> Tuple[float, str]:
    """Return (home_win_share, source) for overtime/shootout games.

    Queries completed games with overtime=1 to compute empirical
    home-win fraction. Returns fallback for non-NHL sports or if
    insufficient samples exist.

    This function is cached per (db_path, sport, season) to avoid
    repeated database queries during a schedule run.

    Args:
        db_path: Path to SQLite database
        sport: Sport code (e.g., "nhl", "nba")
        season: Season identifier (e.g., "2024-25")

    Returns:
        Tuple of (home_win_share, source_label) where:
        - home_win_share: Float in [0, 1] representing P(home wins | OT)
        - source_label: One of "empirical_ot", "fallback_insufficient_data",
          "fallback_error", or "default_non_nhl"
    """
    # Only use empirical OT share for NHL
    if sport.lower() != "nhl":
        return (0.5, "default_non_nhl")

    try:
        db_path_resolved = Path(db_path)
        if not db_path_resolved.exists():
            _LOG.warning(
                "Database not found at %s for OT share query; using fallback %.3f",
                db_path,
                NHL_OT_HOME_WIN_FALLBACK,
            )
            return (NHL_OT_HOME_WIN_FALLBACK, "fallback_db_not_found")

        with closing(sqlite3.connect(db_path_resolved)) as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN home_score > away_score THEN 1 ELSE 0 END) as home_wins
                FROM games
                WHERE overtime = 1
                  AND home_score IS NOT NULL
                  AND away_score IS NOT NULL
                """,
            ).fetchone()

            total, home_wins = row[0] or 0, row[1] or 0

            if total >= NHL_OT_MIN_SAMPLES:
                share = float(home_wins) / float(total)
                _LOG.info(
                    "Using empirical OT home-win share for %s/%s: %.3f (%d/%d games)",
                    sport,
                    season,
                    share,
                    home_wins,
                    total,
                )
                return (share, "empirical_ot")
            else:
                _LOG.info(
                    "Insufficient OT games for %s/%s (%d < %d), using fallback %.3f",
                    sport,
                    season,
                    total,
                    NHL_OT_MIN_SAMPLES,
                    NHL_OT_HOME_WIN_FALLBACK,
                )
                return (NHL_OT_HOME_WIN_FALLBACK, "fallback_insufficient_data")
    except Exception as e:
        _LOG.warning(
            "Failed to query OT share for %s/%s: %s; using fallback %.3f",
            sport,
            season,
            e,
            NHL_OT_HOME_WIN_FALLBACK,
        )
        return (NHL_OT_HOME_WIN_FALLBACK, "fallback_error")


def clear_ot_share_cache() -> None:
    """Clear cached OT share values.

    Useful for testing or when database content has changed.
    """
    get_ot_home_win_share.cache_clear()
