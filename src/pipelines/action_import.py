from __future__ import annotations

from pathlib import Path
from typing import Optional
import csv
from datetime import datetime, timezone

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

from data import betting_repository as br
from utils import identity as idu
from utils.normalization import (
    normalize_market_type_value,
    normalize_team_label,
    normalize_total_selection,
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ActionMarketRow(BaseModel):
    """Validated market row from Action CSV inputs."""

    team_home_raw: str = Field(min_length=1)
    team_away_raw: str = Field(min_length=1)
    game_date: str | None = None
    market_type: str
    selection: str = Field(min_length=1)
    line: float | None = None
    odds: int | None = None
    captured_at: str | None = None

    @field_validator("market_type", mode="before")
    @classmethod
    def _normalize_market_type(cls, value: object) -> str:
        normalized = normalize_market_type_value(str(value) if value is not None else None)
        if not normalized:
            raise ValueError("Unsupported market_type value.")
        return normalized


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
    alias_map = idu.load_alias_map(sport)

    for _, row in df.iterrows():
        try:
            line_value = row.get("line") if "line" in row else None
            if line_value is not None and pd.isna(line_value):
                line_value = None
            odds_value = row.get("odds") if "odds" in row else None
            if odds_value is not None and pd.isna(odds_value):
                odds_value = None
            selection_value = row.get("selection")
            if selection_value is not None and pd.isna(selection_value):
                selection_value = None
            parsed = ActionMarketRow(
                team_home_raw=str(row.get("team_home_raw") or ""),
                team_away_raw=str(row.get("team_away_raw") or ""),
                game_date=str(row.get("game_date")) if row.get("game_date") is not None else None,
                market_type=row.get("market_type"),
                selection=str(selection_value or ""),
                line=line_value,
                odds=odds_value,
                captured_at=row.get(capture_field) if capture_field in row else None,
            )
        except ValidationError:
            rejected += 1
            continue

        home_raw = parsed.team_home_raw
        away_raw = parsed.team_away_raw
        game_date = parsed.game_date
        market_type = parsed.market_type
        selection = parsed.selection
        line = parsed.line
        odds = parsed.odds
        captured_at = parsed.captured_at

        home_match = normalize_team_label(home_raw, alias_map=alias_map) or home_raw
        away_match = normalize_team_label(away_raw, alias_map=alias_map) or away_raw
        selection_match = selection
        if market_type in {"ML", "spread"}:
            selection_match = (
                normalize_team_label(selection, alias_map=alias_map) or selection
            )
        if market_type == "total":
            selection_match = normalize_total_selection(selection) or selection
        effective_snapshot_run_id = snapshot_run_id or br.default_snapshot_run_id(
            sport=sport,
            season=season,
            game_date=str(game_date) if game_date is not None else None,
            captured_at=str(captured_at) if captured_at is not None else None,
            prefix="action-import",
        )

        try:
            # attempt to resolve to a game_id
            res = br.resolve_staging_to_game(
                db_path,
                sport=sport,
                season=season,
                team_home_raw=str(home_match) if home_match is not None else "",
                team_away_raw=str(away_match) if away_match is not None else "",
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
                    snapshot_run_id=effective_snapshot_run_id,
                    captured_at=captured_at or _utcnow_iso(),
                    book=book or "parser",
                    market_type=str(market_type),
                    selection=str(selection_match),
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
                    "selection": selection_match,
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
