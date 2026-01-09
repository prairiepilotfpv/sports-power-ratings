"""Market staging review helpers.

Orchestrates manual review of `market_snapshot_staging` rows:
- list rows filtered by match_status
- accept matches by persisting a game_id
- reject rows to clear pending matches
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.data import betting_repository as br


def list_staging_rows(db_path: str | Path, *, match_statuses: Sequence[str] | None = None, limit: int | None = None) -> list[dict]:
    """Return staging rows filtered by match_status (or all when omitted)."""
    statuses = None
    if match_statuses:
        statuses = [s.strip() for s in match_statuses if s and s.strip()]
    return br.list_staging_rows(db_path, match_statuses=statuses, limit=limit)


def accept_match(
    db_path: str | Path,
    *,
    staging_id: int,
    game_id: str,
    match_confidence: float | None = None,
) -> dict:
    """Mark a staging row as matched and persist its game_id."""
    return br.update_staging_match(
        db_path,
        staging_id=staging_id,
        match_status="matched",
        game_id=game_id,
        match_confidence=match_confidence,
    )


def reject_match(db_path: str | Path, *, staging_id: int) -> dict:
    """Mark a staging row as unmatched and clear game_id."""
    return br.update_staging_match(
        db_path,
        staging_id=staging_id,
        match_status="unmatched",
        game_id=None,
        match_confidence=0.0,
    )


__all__ = [
    "list_staging_rows",
    "accept_match",
    "reject_match",
]
