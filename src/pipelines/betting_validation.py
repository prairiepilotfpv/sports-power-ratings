"""Pre-flight validation checks for betting pipelines."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
import sqlite3

from pipelines import opportunities as opportunities_pipeline
from data import betting_repository as br


def run_preflight_validation(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str,
    date: str,
    snapshot_run_id: str | None = None,
    snapshot_date: str | None = None,
    min_snapshots: int = 1,
) -> dict:
    """Run pre-flight checks before generating review workbooks.

    Returns a dict with integrity, predictions, and snapshot counts.
    """
    if snapshot_run_id is None and snapshot_date is None:
        raise ValueError("Provide snapshot_run_id or snapshot_date for snapshot validation.")

    br.init_db(db_path)
    path = Path(db_path)

    results: dict[str, dict] = {}
    with closing(sqlite3.connect(path)) as conn:
        integrity_row = conn.execute("PRAGMA integrity_check").fetchone()
        integrity_detail = integrity_row[0] if integrity_row else "unknown"
        results["integrity"] = {
            "ok": integrity_detail == "ok",
            "detail": integrity_detail,
        }

        game_rows = conn.execute(
            """
            SELECT game_id
            FROM games
            WHERE sport = ? AND season = ? AND date(date) = date(?)
              AND game_id IS NOT NULL
            """,
            (sport, season, date),
        ).fetchall()
        game_ids = [row[0] for row in game_rows if row[0]]
        results["games"] = {"count": len(game_ids)}

        snapshot_query = """
            SELECT COUNT(*)
            FROM market_snapshots ms
            LEFT JOIN games g ON ms.game_id = g.game_id
            WHERE g.sport = ? AND g.season = ?
        """
        snapshot_params: list[object] = [sport, season]
        if snapshot_run_id is not None:
            snapshot_query += " AND ms.snapshot_run_id = ?"
            snapshot_params.append(snapshot_run_id)
        if snapshot_date is not None:
            snapshot_query += " AND date(ms.captured_at) = date(?)"
            snapshot_params.append(snapshot_date)
        snapshot_count = conn.execute(snapshot_query, snapshot_params).fetchone()[0] or 0
        results["snapshots"] = {
            "count": int(snapshot_count),
            "min_required": int(min_snapshots),
            "ok": int(snapshot_count) >= int(min_snapshots),
        }

    if not game_ids:
        results["predictions"] = {"count": 0, "ok": False}
        return results

    predictions = opportunities_pipeline.load_schedule_predictions(
        path,
        sport=sport,
        season=season,
        model=model,
        game_ids=game_ids,
    )
    pred_count = int(len(predictions.index))
    results["predictions"] = {"count": pred_count, "ok": pred_count > 0}
    return results


__all__ = ["run_preflight_validation"]
