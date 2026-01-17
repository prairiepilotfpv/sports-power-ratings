"""Market-line storage and import helpers."""

from __future__ import annotations

import csv
import json
import logging
import re
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Iterable

from . import repository as base_repo
from . import teams as team_repo
from src.utils.game_id import make_game_id

logger = logging.getLogger(__name__)

_ODDS_RE = re.compile(r"^[+-]?\d+$")


def normalize_game_date(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.split("T", 1)[0].split(" ", 1)[0]
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%Y%m%d",
        "%m%d%Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _normalize_market_type(value: str | None) -> str | None:
    if not value:
        return None
    norm = str(value).strip().lower()
    if norm in {"ml", "moneyline", "money line", "money_line"}:
        return "ML"
    if norm in {"spread", "spreads", "pointspread", "point spread", "handicap", "spreadline", "line"}:
        return "spread"
    if norm in {"total", "totals", "overunder", "over/under", "ou", "o/u", "total_line"}:
        return "total"
    return None


def _normalize_total_selection(value: str | None) -> str | None:
    if not value:
        return None
    norm = str(value).strip().lower()
    if norm in {"over", "o", "ov"}:
        return "Over"
    if norm in {"under", "u", "un"}:
        return "Under"
    return None


def _parse_line(value: str | float | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        filtered = "".join(ch for ch in text if ch in "-.0123456789")
        try:
            return float(filtered) if filtered else None
        except ValueError:
            return None


def _parse_odds(value: str | int | None) -> int | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if not _ODDS_RE.match(s):
        return None
    try:
        odds = int(s)
    except ValueError:
        return None
    if odds == 0 or odds < -1000 or odds > 1000:
        return None
    return odds


def _build_date_filter(date_filter: str | Iterable[str] | None) -> set[str] | None:
    if date_filter is None:
        return None
    if isinstance(date_filter, str):
        normalized = normalize_game_date(date_filter)
        return {normalized} if normalized else set()
    dates: set[str] = set()
    for value in date_filter:
        normalized = normalize_game_date(value)
        if normalized:
            dates.add(normalized)
    return dates


def _record_failure(
    conn: sqlite3.Connection,
    *,
    sport: str,
    season: str,
    row: dict,
    reason: str,
    details: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO market_line_import_errors (
            sport, season, row_data, failure_reason, failure_details, created_at
        ) VALUES (?, ?, ?, ?, ?, datetime('now'))
        """,
        (
            sport,
            season,
            json.dumps(row, default=str),
            reason,
            details,
        ),
    )


def _resolve_game_id(
    conn: sqlite3.Connection,
    *,
    sport: str,
    season: str,
    game_date: str,
    home_team_id: int,
    away_team_id: int,
) -> tuple[str | None, str | None]:
    home_name = team_repo.get_canonical_name(conn, team_id=home_team_id)
    away_name = team_repo.get_canonical_name(conn, team_id=away_team_id)
    if not home_name or not away_name:
        return None, "team_unmatched"

    # prefer deterministic gid lookup
    try:
        gid = make_game_id(sport, season, game_date, away_name, home_name)
        row = conn.execute(
            "SELECT game_id FROM games WHERE sport = ? AND season = ? AND game_id = ?",
            (sport, season, gid),
        ).fetchone()
        if row:
            return row[0], None
    except Exception:
        # fall back to legacy text-match
        pass

    # fallback: match existing rows by exact canonical team names and date
    rows = conn.execute(
        """
        SELECT game_id
        FROM games
        WHERE sport = ? AND season = ? AND date = ? AND home_team = ? AND away_team = ? AND game_id IS NOT NULL
        """,
        (sport, season, game_date, home_name, away_name),
    ).fetchall()
    if not rows:
        return None, "game_unmatched"
    if len(rows) > 1:
        return None, "ambiguous_game_match"
    return rows[0][0], None


def _upsert_market_line(
    conn: sqlite3.Connection,
    *,
    sport: str,
    season: str,
    game_id: str,
    game_date: str,
    market_type: str,
    selection_team_id: int | None,
    selection: str | None,
    line: float | None,
    odds: int,
    book: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO market_lines (
            sport, season, game_id, game_date, market_type,
            selection_team_id, selection, line, odds, book, imported_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT DO UPDATE SET
            odds = excluded.odds,
            imported_at = datetime('now')
        """,
        (
            sport,
            season,
            game_id,
            game_date,
            market_type,
            selection_team_id,
            selection,
            line,
            odds,
            book,
        ),
    )


def import_market_csv(
    db_path: str | Path,
    *,
    csv_path: str | Path,
    sport: str,
    season: str,
    default_book: str | None = None,
    date_filter: str | Iterable[str] | None = None,
) -> dict:
    """Import market CSV rows into the market_lines table with diagnostics."""
    base_repo.init_db(db_path)
    rows_loaded = 0
    inserted = 0
    unmatched = 0
    filtered = 0
    reason_counts: Counter[str] = Counter()
    examples: list[dict] = []
    date_filter_set = _build_date_filter(date_filter)
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with closing(sqlite3.connect(Path(db_path))) as conn, csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader, start=1):
            rows_loaded += 1
            game_date_raw = row.get("game_date")
            iso_date = normalize_game_date(game_date_raw)
            if not iso_date:
                unmatched += 1
                reason = "date_parse_failed"
                reason_counts[reason] += 1
                _record_failure(conn, sport=sport, season=season, row=row, reason=reason, details=str(game_date_raw))
                _append_example(
                    examples,
                    row,
                    reason=reason,
                    details=f"raw date={game_date_raw}",
                    row_index=idx,
                )
                continue
            if date_filter_set and iso_date not in date_filter_set:
                filtered += 1
                continue

            market_type = _normalize_market_type(row.get("market_type"))
            if not market_type:
                unmatched += 1
                reason = "market_invalid"
                reason_counts[reason] += 1
                _record_failure(conn, sport=sport, season=season, row=row, reason=reason, details="invalid market_type")
                _append_example(examples, row, reason=reason, details="invalid market_type", row_index=idx)
                continue

            odds = _parse_odds(row.get("odds"))
            if odds is None:
                unmatched += 1
                reason = "market_invalid"
                reason_counts[reason] += 1
                _record_failure(conn, sport=sport, season=season, row=row, reason=reason, details="invalid odds")
                _append_example(examples, row, reason=reason, details="invalid odds", row_index=idx)
                continue

            home_team_id = team_repo.resolve_team_id(conn, sport=sport, season=season, raw_team_name=row.get("team_home_raw"))
            away_team_id = team_repo.resolve_team_id(conn, sport=sport, season=season, raw_team_name=row.get("team_away_raw"))
            if not home_team_id or not away_team_id:
                unmatched += 1
                reason = "team_unmatched"
                details = "home" if not home_team_id else "away"
                reason_counts[reason] += 1
                _record_failure(conn, sport=sport, season=season, row=row, reason=reason, details=f"{details}_team_unmatched")
                _append_example(examples, row, reason=reason, details=f"{details}_team_unmatched", row_index=idx)
                continue

            selection_team_id: int | None = None
            selection_text: str | None = None
            line_value: float | None = None
            selection_raw = row.get("selection")

            if market_type == "ML":
                selection_team_id = team_repo.resolve_team_id(conn, sport=sport, season=season, raw_team_name=selection_raw)
                if not selection_team_id:
                    unmatched += 1
                    reason = "team_unmatched"
                    reason_counts[reason] += 1
                    _record_failure(conn, sport=sport, season=season, row=row, reason=reason, details="selection_team_unmatched")
                    _append_example(examples, row, reason=reason, details="selection_team_unmatched", row_index=idx)
                    continue
                selection_text = team_repo.get_canonical_name(conn, team_id=selection_team_id) or str(selection_raw).strip()
                if not selection_text:
                    selection_text = None
            elif market_type == "spread":
                selection_team_id = team_repo.resolve_team_id(conn, sport=sport, season=season, raw_team_name=selection_raw)
                if not selection_team_id:
                    unmatched += 1
                    reason = "team_unmatched"
                    reason_counts[reason] += 1
                    _record_failure(conn, sport=sport, season=season, row=row, reason=reason, details="selection_team_unmatched")
                    _append_example(examples, row, reason=reason, details="selection_team_unmatched", row_index=idx)
                    continue
                line_value = _parse_line(row.get("line"))
                if line_value is None:
                    unmatched += 1
                    reason = "market_invalid"
                    reason_counts[reason] += 1
                    _record_failure(conn, sport=sport, season=season, row=row, reason=reason, details="missing spread line")
                    _append_example(examples, row, reason=reason, details="missing spread line", row_index=idx)
                    continue
                selection_text = team_repo.get_canonical_name(conn, team_id=selection_team_id) or str(selection_raw).strip()
                if not selection_text:
                    selection_text = None
            else:
                # total
                selection_text = _normalize_total_selection(selection_raw)
                if not selection_text:
                    unmatched += 1
                    reason = "selection_invalid"
                    reason_counts[reason] += 1
                    _record_failure(conn, sport=sport, season=season, row=row, reason=reason, details="invalid total selection")
                    _append_example(examples, row, reason=reason, details="invalid total selection", row_index=idx)
                    continue
                line_value = _parse_line(row.get("line"))
                if line_value is None:
                    unmatched += 1
                    reason = "market_invalid"
                    reason_counts[reason] += 1
                    _record_failure(conn, sport=sport, season=season, row=row, reason=reason, details="missing total line")
                    _append_example(examples, row, reason=reason, details="missing total line", row_index=idx)
                    continue

            game_id, game_reason = _resolve_game_id(
                conn,
                sport=sport,
                season=season,
                game_date=iso_date,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
            )
            if not game_id:
                unmatched += 1
                reason = game_reason or "game_unmatched"
                reason_counts[reason] += 1
                _record_failure(conn, sport=sport, season=season, row=row, reason=reason, details="game lookup")
                _append_example(examples, row, reason=reason, details="game lookup", row_index=idx)
                continue

            try:
                _upsert_market_line(
                    conn,
                    sport=sport,
                    season=season,
                    game_id=game_id,
                    game_date=iso_date,
                    market_type=market_type,
                    selection_team_id=selection_team_id,
                    selection=selection_text,
                    line=line_value,
                    odds=odds,
                    book=row.get("book") or default_book,
                )
            except Exception as exc:
                logger.exception("Failed to upsert market line: %s", exc)
                unmatched += 1
                reason = "market_invalid"
                reason_counts[reason] += 1
                _record_failure(conn, sport=sport, season=season, row=row, reason=reason, details=str(exc))
                _append_example(examples, row, reason=reason, details="upsert failure", row_index=idx)
                continue
            inserted += 1
        conn.commit()
    return {
        "rows_loaded": rows_loaded,
        "inserted": inserted,
        "unmatched": unmatched,
        "date_filtered": filtered,
        "unmatched_reasons": dict(reason_counts),
        "unmatched_examples": examples[:10],
    }


def _append_example(
    examples: list[dict],
    row: dict,
    *,
    reason: str,
    details: str,
    row_index: int,
) -> None:
    if len(examples) >= 10:
        return
    examples.append(
        {
            "row_index": row_index,
            "team_home_raw": row.get("team_home_raw"),
            "team_away_raw": row.get("team_away_raw"),
            "game_date": row.get("game_date"),
            "market_type": row.get("market_type"),
            "selection": row.get("selection"),
            "line": row.get("line"),
            "odds": row.get("odds"),
            "reason": reason,
            "details": details,
        }
    )
