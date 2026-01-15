from __future__ import annotations

from pathlib import Path
from typing import Optional
import csv
from datetime import datetime, timezone

import pandas as pd

from data import betting_repository as br


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def import_from_csv(
    csv_path: str | Path,
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    snapshot_run_id: Optional[str] = None,
    book: Optional[str] = None,
    capture_field: Optional[str] = "captured_at",
):
    """Import market rows from a parser-generated CSV into market_snapshots/staging.

    - Expected CSV columns: team_home_raw, team_away_raw, game_date, market_type, selection, line, odds
    - If a row resolves to a game_id (via fuzzy resolver) it will be inserted into
      `market_snapshots` (snapshot_run_id param used). Otherwise it will be inserted
      into `market_snapshot_staging` for later review.

    Returns dict with counts: inserted, staged, rejected
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    required = [
        "team_home_raw",
        "team_away_raw",
        "game_date",
        "market_type",
        "selection",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")

    inserted = 0
    staged = 0
    rejected = 0

    for _, row in df.iterrows():
        home_raw = row.get("team_home_raw")
        away_raw = row.get("team_away_raw")
        game_date = row.get("game_date")
        market_type = row.get("market_type")
        selection = row.get("selection")
        line = row.get("line") if "line" in row else None
        odds = row.get("odds") if "odds" in row else None
        captured_at = row.get(capture_field) if capture_field in row else None

        try:
            # attempt to resolve to a game_id
            res = br.resolve_staging_to_game(
                db_path,
                sport=sport,
                season=season,
                team_home_raw=str(home_raw) if home_raw is not None else "",
                team_away_raw=str(away_raw) if away_raw is not None else "",
                game_date=str(game_date) if game_date is not None else None,
            )
        except Exception:
            res = {"game_id": None, "match_status": "unmatched", "match_confidence": 0.0}

        game_id = res.get("game_id")
        match_status = res.get("match_status")

        try:
            if match_status == "matched" and game_id:
                # insert into market_snapshots
                snap_id = br.add_market_snapshot(
                    db_path,
                    snapshot_run_id=snapshot_run_id or f"action-import-{_utcnow_iso()}",
                    captured_at=captured_at or _utcnow_iso(),
                    book=book or "parser",
                    market_type=str(market_type),
                    selection=str(selection),
                    line=None if pd.isna(line) else float(line),
                    odds=None if pd.isna(odds) else int(odds) if odds is not None else None,
                    game_id=game_id,
                    source_staging_id=None,
                )
                inserted += 1
            else:
                # write to staging for review
                st = {
                    "source": "action_parser",
                    "captured_at": captured_at or _utcnow_iso(),
                    "image_path": None,
                    "raw_text": None,
                    "book": book or "parser",
                    "market_type": market_type,
                    "selection": selection,
                    "line": None if pd.isna(line) else float(line) if line is not None else None,
                    "odds": None if pd.isna(odds) else int(odds) if odds is not None else None,
                    "team_home_raw": home_raw,
                    "team_away_raw": away_raw,
                    "game_date": game_date,
                    "match_status": match_status,
                    "match_confidence": res.get("match_confidence"),
                    "game_id": game_id,
                    "hold_reason": None,
                }
                br.add_staging_row(db_path, **st)
                staged += 1
        except Exception:
            rejected += 1

    return {"inserted": inserted, "staged": staged, "rejected": rejected}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", dest="csv", required=True)
    parser.add_argument("--sport", required=True)
    parser.add_argument("--season", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--snapshot-run-id", dest="snapshot_run_id", help="Optional snapshot_run_id to attach")
    parser.add_argument("--book", help="Optional book name to attach")
    args = parser.parse_args()
    res = import_from_csv(args.csv, args.db, sport=args.sport, season=args.season, snapshot_run_id=args.snapshot_run_id, book=args.book)
    print(f"action-import: inserted={res.get('inserted')} staged={res.get('staged')} rejected={res.get('rejected')}")
