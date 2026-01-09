"""Pivot reviewed staging rows into bet entries with simple stake presets.

Features:
- Filters staging rows by match_status (default: matched)
- Auto-detects duplicate markets from the same image and tags them with hold_reason
- Applies stake presets (half/unit/double) against a base unit stake
- Upserts into `bets` table to remain idempotent
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import sqlite3

from src.data import betting_repository as br


STAKE_PRESETS = {
    "half": 0.5,
    "unit": 1.0,
    "double": 2.0,
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_review_run(review_run_id: str | None) -> str:
    if review_run_id:
        return review_run_id
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"staging-{ts}"


def _normalized_statuses(match_statuses: Sequence[str] | None) -> list[str] | None:
    if not match_statuses:
        return None
    normalized = [s.strip() for s in match_statuses if s and s.strip()]
    return normalized or None


def _compute_stake(unit_stake: float, stake_preset: str) -> float:
    multiplier = STAKE_PRESETS.get(stake_preset, 1.0)
    return unit_stake * multiplier


def _auto_hold_duplicates(rows: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """Split rows into (kept, held) based on duplicate detection.

    Duplicate key: (image_path, market_type, normalized selection). The first
    occurrence is kept; later duplicates are held.
    """
    seen: set[tuple[Optional[str], Optional[str], str]] = set()
    kept: list[dict] = []
    held: list[dict] = []
    # Process in insertion order (ascending id) so earliest is kept, later duplicates held
    for row in sorted(rows, key=lambda r: r.get("id")):
        image_path = row.get("image_path")
        if not image_path:
            # Without an image path we cannot reliably detect duplicate screenshots; keep row.
            kept.append(row)
            continue
        key = (
            image_path,
            row.get("market_type"),
            (row.get("selection") or "").lower(),
        )
        if key in seen:
            held.append(row)
        else:
            seen.add(key)
            kept.append(row)
    return kept, held


def pivot_staging_to_bets(
    db_path: str | Path,
    *,
    review_run_id: str | None = None,
    match_statuses: Sequence[str] | None = ("matched",),
    limit: int | None = None,
    stake_preset: str = "unit",
    unit_stake: float = 1.0,
    default_book: str | None = None,
    auto_hold_duplicates: bool = True,
    dry_run: bool = False,
) -> dict:
    """Pivot staging rows into bets.

    Returns summary dict: {"inserted": int, "held": int, "skipped": int, "review_run_id": str, "stake": float}
    """
    br.init_db(db_path)
    statuses = _normalized_statuses(match_statuses)
    rows = br.list_staging_rows(db_path, match_statuses=statuses, limit=limit)
    if not rows:
        return {"inserted": 0, "held": 0, "skipped": 0, "review_run_id": _resolve_review_run(review_run_id), "stake": _compute_stake(unit_stake, stake_preset)}

    # Filter invalid rows early
    valid_rows: list[dict] = []
    skipped = 0
    for row in rows:
        if not row.get("game_id") or not row.get("market_type"):
            skipped += 1
            continue
        odds = row.get("odds")
        if odds is None or odds == 0:
            skipped += 1
            continue
        valid_rows.append(row)

    kept_rows = valid_rows
    held_rows: list[dict] = []
    if auto_hold_duplicates:
        kept_rows, held_rows = _auto_hold_duplicates(valid_rows)
        if not dry_run:
            held_ids = [hr["id"] for hr in held_rows]
            if held_ids:
                placeholders = ", ".join(["?"] * len(held_ids))
                with sqlite3.connect(Path(db_path)) as conn:
                    conn.execute(
                        f"UPDATE market_snapshot_staging SET hold_reason = 'duplicate_in_image' WHERE id IN ({placeholders})",
                        held_ids,
                    )
                    conn.commit()
                # double-tag via helper to ensure consistency on legacy DBs/migrations
                for hid in held_ids:
                    br.tag_staging_hold(db_path, staging_id=hid, reason="duplicate_in_image")

    stake_value = _compute_stake(unit_stake, stake_preset)
    resolved_review_run_id = _resolve_review_run(review_run_id)

    if dry_run:
        return {
            "inserted": len(kept_rows),
            "held": len(held_rows),
            "skipped": skipped,
            "review_run_id": resolved_review_run_id,
            "stake": stake_value,
        }

    inserted = 0
    with sqlite3.connect(Path(db_path)) as conn:
        cur = conn.cursor()
        for row in kept_rows:
            book = row.get("book") or default_book
            game_id = row.get("game_id")
            market_type = row.get("market_type")
            selection = row.get("selection")
            line = row.get("line")
            odds = row.get("odds")
            logged_at = _utcnow_iso()

            # attach latest CLV snapshot if present
            clv = br.get_latest_clv(db_path, game_id=game_id, market_type=market_type, selection=selection)
            clv_line = clv.get("close_line") if clv else None
            clv_odds = clv.get("close_odds") if clv else None

            cur.execute(
                """
                INSERT INTO bets (
                    review_run_id, game_id, market_type, selection, line, odds, stake, book, logged_at, status, source_opportunity_id, clv_close_odds, clv_close_line
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
                ON CONFLICT(review_run_id, game_id, market_type, selection) DO UPDATE SET
                    stake = excluded.stake,
                    odds = excluded.odds,
                    line = excluded.line,
                    book = excluded.book,
                    logged_at = excluded.logged_at,
                    status = excluded.status,
                    source_opportunity_id = excluded.source_opportunity_id,
                    clv_close_odds = excluded.clv_close_odds,
                    clv_close_line = excluded.clv_close_line
                """,
                (
                    resolved_review_run_id,
                    game_id,
                    market_type,
                    selection,
                    line,
                    odds,
                    stake_value,
                    book,
                    logged_at,
                    None,
                    clv_odds,
                    clv_line,
                ),
            )
            inserted += 1
        conn.commit()

    # Safety: ensure no lingering connections hold file locks on Windows
    sqlite3.connect(Path(db_path)).close()

    return {
        "inserted": inserted,
        "held": len(held_rows),
        "skipped": skipped,
        "review_run_id": resolved_review_run_id,
        "stake": stake_value,
    }


__all__ = [
    "pivot_staging_to_bets",
]
