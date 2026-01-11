"""Persistence helpers for betting/review pipelines.

This module implements an idempotent `init_db` function that will create the
betting-specific tables within the per-sport/per-season SQLite database.

Schema implemented (see project notes):
- review_runs
- market_snapshot_staging
- market_snapshots
- forecast_snapshots
- opportunities
- bets

The functions here are conservative: they create tables if missing and provide
basic helpers for opening the DB and ensuring schema exists. Higher-level
CRUD and transaction helpers will be added as pipelines are implemented.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Iterable, Sequence, List, Dict, Any, Optional
from datetime import datetime, timezone
import logging
import uuid

from .paths import db_path_for
from . import repository as base_repo
from .migrations import apply_migrations


logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS review_runs (
    id TEXT PRIMARY KEY,
    sport TEXT,
    season TEXT,
    model TEXT,
    created_at TEXT,
    source_market_snapshot_id TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS market_snapshot_staging (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT,
    captured_at TEXT,
    image_path TEXT,
    raw_text TEXT,
    book TEXT,
    market_type TEXT,
    selection TEXT,
    line REAL,
    odds INTEGER,
    team_home_raw TEXT,
    team_away_raw TEXT,
    game_date TEXT,
    match_status TEXT,
    match_confidence REAL,
    game_id TEXT,
    hold_reason TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_run_id TEXT,
    captured_at TEXT,
    book TEXT,
    market_type TEXT,
    selection TEXT,
    line REAL,
    odds INTEGER,
    game_id TEXT,
    source_staging_id INTEGER,
    created_at TEXT,
    UNIQUE(snapshot_run_id, game_id, market_type, selection)
);

CREATE TABLE IF NOT EXISTS forecast_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_run_id TEXT,
    model TEXT,
    game_id TEXT,
    projected_spread REAL,
    projected_total REAL,
    projected_home_score REAL,
    projected_away_score REAL,
    win_prob_home REAL,
    created_at TEXT,
    UNIQUE(review_run_id, game_id, model)
);

CREATE TABLE IF NOT EXISTS opportunities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_run_id TEXT,
    game_id TEXT,
    market_type TEXT,
    selection TEXT,
    line REAL,
    odds INTEGER,
    implied_prob REAL,
    model_prob REAL,
    edge REAL,
    ev REAL,
    source_market_snapshot_id INTEGER,
    created_at TEXT,
    UNIQUE(review_run_id, game_id, market_type, selection)
);

CREATE TABLE IF NOT EXISTS bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_run_id TEXT,
    game_id TEXT,
    market_type TEXT,
    selection TEXT,
    line REAL,
    odds INTEGER,
    stake REAL,
    book TEXT,
    logged_at TEXT,
    clv_close_odds INTEGER,
    clv_close_line REAL,
    status TEXT,
    outcome TEXT,
    profit REAL,
    source_opportunity_id INTEGER,
    UNIQUE(review_run_id, game_id, market_type, selection)
);

CREATE TABLE IF NOT EXISTS clv_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT,
    market_type TEXT,
    selection TEXT,
    close_line REAL,
    close_odds INTEGER,
    captured_at TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS prediction_exclusions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_run_id TEXT,
    game_id TEXT,
    model TEXT,
    excluded_reason TEXT,
    created_at TEXT,
    UNIQUE(review_run_id, game_id, model, excluded_reason)
);

CREATE INDEX IF NOT EXISTS idx_market_snapshots_game_id ON market_snapshots(game_id);
CREATE INDEX IF NOT EXISTS idx_forecast_snapshots_review_run ON forecast_snapshots(review_run_id);
CREATE INDEX IF NOT EXISTS idx_opportunities_review_run ON opportunities(review_run_id);
CREATE INDEX IF NOT EXISTS idx_bets_review_run ON bets(review_run_id);
CREATE INDEX IF NOT EXISTS idx_prediction_exclusions_review_run ON prediction_exclusions(review_run_id);
"""


def init_db(db_path: str | Path) -> None:
    """Ensure the betting tables exist in the per-sport/per-season DB.

    This is additive and idempotent: it will call the base repository's
    `init_db` (to ensure `games` and other core tables exist) and then apply
    the betting schema. Use this during pipelines startup or in tests.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure base schema (games, model_metrics, etc.) exists before adding betting
    base_repo.init_db(path)
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(SCHEMA)
        apply_migrations(conn)
        conn.commit()


def connect_for(sport: str, season: str) -> sqlite3.Connection:
    """Return a sqlite3 connection for the sport/season DB.

    Caller is responsible for closing the connection or using a context manager.
    """
    return sqlite3.connect(db_path_for(sport, season))


def resolve_staging_to_game(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    team_home_raw: str | None,
    team_away_raw: str | None,
    game_date: str | None = None,
) -> dict:
    """Attempt to resolve OCR staging home/away team strings to a canonical game_id.

    Returns a dict with keys: game_id (or None), match_confidence (0.0-1.0),
    match_status in ("matched","needs_review","unmatched"), and
    candidate (the matched game_id if any).

    Strategy:
    - Load games for sport/season
    - Load alias map and identity thresholds
    - For each game, compute match score for home and away teams (via
      normalize/alias/fuzzy). Use the minimum of the two as the pair score.
    - Penalize pair score if given game_date is >1 day away from scheduled
      game date.
    - Choose best candidate; apply thresholds to set match_status.
    """
    from datetime import date
    from . import repository as repo
    from src.utils import identity as idu

    candidates = repo.load_games(db_path, sport=sport, season=season)

    alias_map = idu.load_alias_map(sport)
    thresholds = idu.load_identity_config()
    auto_thr = float(thresholds.get("auto", 0.9))
    needs_review_thr = float(thresholds.get("needs_review", 0.75))

    best = None
    best_score = 0.0
    best_game_id = None

    for g in candidates:
        # compute home score
        home_match = None
        home_score = 0.0
        away_match = None
        away_score = 0.0

        # alias direct
        hm_alias = idu.resolve_team_alias(team_home_raw or "", alias_map)
        if hm_alias is not None and hm_alias == g.home_team:
            home_match = g.home_team
            home_score = 1.0
        else:
            hm, hs = idu.fuzzy_match_team(team_home_raw or "", [g.home_team])
            home_match = hm
            home_score = hs

        am_alias = idu.resolve_team_alias(team_away_raw or "", alias_map)
        if am_alias is not None and am_alias == g.away_team:
            away_match = g.away_team
            away_score = 1.0
        else:
            am, ascr = idu.fuzzy_match_team(team_away_raw or "", [g.away_team])
            away_match = am
            away_score = ascr

        pair_score = min(home_score, away_score)

        # also consider swapped (home/away reversed) and penalize slightly
        swapped_home_score = 0.0
        swapped_away_score = 0.0
        shm, shm_score = idu.fuzzy_match_team(team_home_raw or "", [g.away_team])
        sam, sam_score = idu.fuzzy_match_team(team_away_raw or "", [g.home_team])
        swapped_pair_score = min(shm_score, sam_score) - 0.15

        effective_score = max(pair_score, swapped_pair_score)

        # penalize by date mismatch > 1 day if game_date provided
        if game_date:
            try:
                parsed = date.fromisoformat(game_date)
                gdate = g.date
                delta = abs((parsed - gdate).days)
                if delta > 1:
                    effective_score -= 0.2
            except Exception:
                # ignore parse issues
                pass

        if effective_score > best_score:
            best_score = effective_score
            best = g
            best_game_id = g.game_id

    status = "unmatched"
    if best_score >= auto_thr:
        status = "matched"
    elif best_score >= needs_review_thr:
        status = "needs_review"

    return {
        "game_id": best_game_id,
        "match_confidence": round(float(best_score), 3),
        "match_status": status,
    }


# --- Timestamp helpers and review_run CRUD ---

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_review_run(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str,
    notes: str | None = None,
    id: str | None = None,
    created_at: str | None = None,
) -> str:
    """Create a review_run record and return its id (generated if not provided).

    Accepts an explicit `created_at` timestamp to avoid drift when running
    multi-step commands; defaults to current UTC timestamp if not provided.
    """
    if id is None:
        id = uuid.uuid4().hex
    if created_at is None:
        created_at = _utcnow_iso()
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO review_runs (
                id, sport, season, model, created_at, source_market_snapshot_id, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (id, sport, season, model, created_at, None, notes),
        )
        conn.commit()
    return id


def add_staging_row(db_path: str | Path, **kwargs) -> int:
    """Insert a staging row. Returns the staging id.

    Ensures `created_at` is set to a consistent UTC timestamp if not provided.
    """
    init_db(db_path)
    fields = [
        "source",
        "captured_at",
        "image_path",
        "raw_text",
        "book",
        "market_type",
        "selection",
        "line",
        "odds",
        "team_home_raw",
        "team_away_raw",
        "game_date",
        "match_status",
        "match_confidence",
        "game_id",
        "created_at",
    ]
    if "created_at" not in kwargs or kwargs.get("created_at") is None:
        kwargs["created_at"] = _utcnow_iso()
    _log_high_risk_staging(kwargs)
    vals = [kwargs.get(f) for f in fields]
    with closing(sqlite3.connect(Path(db_path))) as conn:
        cur = conn.execute(
            f"INSERT INTO market_snapshot_staging ({', '.join(fields)}) VALUES ({', '.join(['?']*len(fields))})",
            vals,
        )
        conn.commit()
        return cur.lastrowid


def _log_high_risk_staging(row: dict) -> None:
    reasons: list[str] = []
    odds = row.get("odds")
    try:
        odds_val = int(odds) if odds is not None else None
    except Exception:
        odds_val = None

    if odds_val is not None:
        if odds_val == 0:
            reasons.append("odds_zero")
        if abs(odds_val) > 400:
            reasons.append("odds_outside_typical_range")

    match_status = (row.get("match_status") or "").strip().lower()
    game_id = row.get("game_id")
    if match_status == "matched" and not game_id:
        reasons.append("missing_game_id_for_matched")

    if not reasons:
        return
    logger.warning(
        "High-risk staging row: %s (source=%s market_type=%s selection=%s odds=%s line=%s match_status=%s)",
        ",".join(reasons),
        row.get("source"),
        row.get("market_type"),
        row.get("selection"),
        odds,
        row.get("line"),
        match_status or None,
    )


def _get_staging_rows(conn: sqlite3.Connection, staging_ids: Sequence[int]) -> List[Dict[str, Any]]:
    q = f"SELECT id, match_status, match_confidence, game_id, source, market_type, selection, line, odds FROM market_snapshot_staging WHERE id IN ({', '.join(['?']*len(staging_ids))})"
    rows = conn.execute(q, staging_ids).fetchall()
    cols = [c[0] for c in conn.execute("PRAGMA table_info(market_snapshot_staging)").fetchall()]
    # Map rows to dicts (keep minimal fields)
    result = []
    for r in rows:
        result.append({
            "id": r[0],
            "match_status": r[1],
            "match_confidence": r[2],
            "game_id": r[3],
            "source": r[4],
            "market_type": r[5],
            "selection": r[6],
            "line": r[7],
            "odds": r[8],
        })
    return result


def commit_market_snapshots(db_path: str | Path, *, snapshot_run_id: str, staging_ids: Sequence[int], force: bool = False) -> int:
    """Commit matched staging rows into `market_snapshots`.

    - If any staging row has match_status == 'needs_review' and force == False, raise ValueError.
    - Rows with match_status != 'matched' (e.g., 'unmatched') are skipped.
    - Insert or replace into `market_snapshots` to preserve idempotency based on UNIQUE constraint.
    Returns number of rows committed.
    """
    if not staging_ids:
        return 0
    init_db(db_path)
    committed = 0
    with closing(sqlite3.connect(Path(db_path))) as conn:
        conn.row_factory = sqlite3.Row
        rows = _get_staging_rows(conn, staging_ids)
        needs_review = [r["id"] for r in rows if r["match_status"] == "needs_review"]
        if needs_review and not force:
            raise ValueError(f"Staging rows {needs_review} are marked needs_review; pass force=True to override")

        for r in rows:
            if r["match_status"] != "matched" and not (force and r["match_status"] in ("needs_review", "matched")):
                # Skip unmatched rows
                continue
            # Insert or replace into market_snapshots
            conn.execute(
                """
                INSERT OR REPLACE INTO market_snapshots (
                    snapshot_run_id, captured_at, book, market_type, selection, line, odds, game_id, source_staging_id, created_at
                ) VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    snapshot_run_id,
                    r.get("source"),
                    r.get("market_type"),
                    r.get("selection"),
                    r.get("line"),
                    r.get("odds"),
                    r.get("game_id"),
                    r.get("id"),
                ),
            )
            conn.execute("UPDATE market_snapshot_staging SET match_status = 'committed' WHERE id = ?", (r["id"],))
            committed += 1
        conn.commit()
    return committed


def export_needs_review(db_path: str | Path, *, sport: str | None = None, season: str | None = None) -> List[Dict[str, Any]]:
    """Return staging rows that are marked needs_review for optional sport/season scope."""
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        q = "SELECT * FROM market_snapshot_staging WHERE match_status = 'needs_review'"
        rows = conn.execute(q).fetchall()
        cols = [c[0] for c in conn.execute("PRAGMA table_info(market_snapshot_staging)").fetchall()]
        results = []
        for r in rows:
            results.append({cols[i]: r[i] for i in range(len(cols))})
        return results


def list_staging_rows(
    db_path: str | Path,
    *,
    match_statuses: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return staging rows filtered by match_status values (default: all).

    Results are ordered by created_at DESC then id DESC to surface the most recent
    entries first. When limit is provided, rows are truncated accordingly.
    """
    init_db(db_path)
    statuses: List[str] | None = None
    if match_statuses:
        statuses = [s.strip() for s in match_statuses if s and s.strip()]
        if not statuses:
            statuses = None

    with closing(sqlite3.connect(Path(db_path))) as conn:
        conn.row_factory = sqlite3.Row
        base = (
            "SELECT id, match_status, match_confidence, game_id, source, captured_at, "
            "book, market_type, selection, line, odds, team_home_raw, team_away_raw, game_date, raw_text, image_path, hold_reason "
            "FROM market_snapshot_staging"
        )
        params: list[Any] = []
        if statuses:
            placeholders = ", ".join(["?"] * len(statuses))
            base = f"{base} WHERE match_status IN ({placeholders})"
            params.extend(statuses)
        base = f"{base} ORDER BY datetime(created_at) DESC, id DESC"
        if limit:
            base = f"{base} LIMIT ?"
            params.append(int(limit))

        rows = conn.execute(base, params).fetchall()
        result = [dict(r) for r in rows]

        # Auto-tag duplicates by image/market/selection to expose hold reasons even if not yet persisted
        seen: set[tuple[str | None, str | None, str | None]] = set()
        # Iterate from oldest to newest so the first occurrence is kept and later ones flagged
        for row in reversed(result):
            key = (
                row.get("image_path"),
                row.get("market_type"),
                (row.get("selection") or "").lower() if row.get("selection") else None,
            )
            if key in seen and not row.get("hold_reason"):
                row["hold_reason"] = "duplicate_in_image"
            else:
                seen.add(key)
        return result


def update_staging_match(
    db_path: str | Path,
    *,
    staging_id: int,
    match_status: str,
    game_id: Optional[str],
    match_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """Update a staging row's match decision and game_id.

    - match_status must be one of matched|needs_review|unmatched
    - game_id is required when setting matched
    - match_confidence defaults to 1.0 for matched rows and 0.0 otherwise
    Returns the updated row as a dict for downstream logging/tests.
    """
    allowed_status = {"matched", "needs_review", "unmatched"}
    status = match_status.strip().lower()
    if status not in allowed_status:
        raise ValueError(f"Unsupported match_status: {match_status}")
    if status == "matched" and (game_id is None or str(game_id).strip() == ""):
        raise ValueError("game_id is required when marking a staging row as matched")

    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        conn.row_factory = sqlite3.Row
        existing = conn.execute(
            "SELECT id FROM market_snapshot_staging WHERE id = ?",
            (staging_id,),
        ).fetchone()
        if existing is None:
            raise ValueError(f"Staging row {staging_id} does not exist")

        confidence = (
            float(match_confidence)
            if match_confidence is not None
            else (1.0 if status == "matched" else 0.0)
        )
        conn.execute(
            "UPDATE market_snapshot_staging SET match_status = ?, match_confidence = ?, game_id = ? WHERE id = ?",
            (status, confidence, game_id, staging_id),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id, match_status, match_confidence, game_id, source, captured_at, book, market_type, selection, line, odds, team_home_raw, team_away_raw, game_date FROM market_snapshot_staging WHERE id = ?",
            (staging_id,),
        ).fetchone()
        return dict(row)


def tag_staging_hold(
    db_path: str | Path,
    *,
    staging_id: int,
    reason: str,
) -> dict:
    """Set hold_reason on a staging row for auto-hold tagging.

    Returns updated row including hold_reason for logging/tests.
    """
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        conn.row_factory = sqlite3.Row
        existing = conn.execute(
            "SELECT id FROM market_snapshot_staging WHERE id = ?",
            (staging_id,),
        ).fetchone()
        if existing is None:
            raise ValueError(f"Staging row {staging_id} does not exist")
        conn.execute(
            "UPDATE market_snapshot_staging SET hold_reason = ? WHERE id = ?",
            (reason, staging_id),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id, match_status, match_confidence, game_id, source, captured_at, book, market_type, selection, line, odds, team_home_raw, team_away_raw, game_date, hold_reason FROM market_snapshot_staging WHERE id = ?",
            (staging_id,),
        ).fetchone()
        return dict(row)


# Lightweight convenience helpers will be added as pipelines are implemented.


__all__ = [
    "init_db",
    "connect_for",
    "create_review_run",
    "add_staging_row",
    "commit_market_snapshots",
    "export_needs_review",
    "list_staging_rows",
    "update_staging_match",
    "tag_staging_hold",
    "resolve_staging_to_game",
    "get_opportunities_with_game_info",
    "get_prediction_exclusions",
    "add_clv_snapshot",
    "get_latest_clv",
    "import_clv_csv",
    "add_prediction_exclusions",
    "update_bets_with_clv",
]


def _clean_odds_field(raw: str | int | None) -> int | None:
    """Apply OCR cleanup heuristics and convert to integer odds, or return None."""
    if raw is None:
        return None
    s = str(raw).strip()
    # normalize unicode minus/dash to ASCII
    s = s.replace("−", "-").replace("–", "-")
    # fix common OCR: O or o to 0 when it makes sense (for numbers with signs)
    s = s.replace("O", "0").replace("o", "0")
    # strip stray characters like trailing ) or whitespace
    s = s.strip().strip(")",)
    # ensure leading + sign is present for positive odds
    if s.startswith("+") or s.startswith("-"):
        try:
            val = int(s)
            return val
        except ValueError:
            # try removing non-digit
            digits = ''.join(ch for ch in s if ch in '+-0123456789')
            try:
                return int(digits)
            except Exception:
                return None
    else:
        # maybe it's a number without sign, treat as positive
        try:
            return int(s)
        except Exception:
            digits = ''.join(ch for ch in s if ch in '0123456789')
            if digits:
                return int(digits)
            return None


def _clean_line_field(raw: str | float | None) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip()
    s = s.replace("−", "-").replace("–", "-")
    s = s.replace("O", "0").replace("o", "0")
    s = s.strip().strip(")",)
    try:
        return float(s)
    except Exception:
        # attempt to remove non-numeric except dot and minus
        filtered = ''.join(ch for ch in s if ch in '-.0123456789')
        try:
            return float(filtered) if filtered else None
        except Exception:
            return None


def import_market_csv(
    db_path: str | Path,
    *,
    csv_path: str | Path,
    snapshot_run_id: str,
    sport: str,
    season: str,
    default_book: str | None = None,
    commit_matched: bool = True,
) -> dict:
    """Import a CSV of markets. Returns summary dict.

    Expected CSV columns (some optional):
    - source (optional)
    - captured_at (optional)
    - book (optional)
    - market_type (ML/spread/total)
    - selection
    - line
    - odds
    - team_home
    - team_away
    - game_date (YYYY-MM-DD)
    - game_id (optional)

    Behavior:
    - Validate numeric fields (odds integer in [-1000,1000], line float)
    - If game_id supplied, insert into market_snapshots directly
    - Otherwise resolve via identity subsystem; matched rows inserted into market_snapshots if commit_matched=True, others go to market_snapshot_staging with match_status set
    - Returns counts: {committed: N, staged: M, rejected: K}
    """
    import csv

    init_db(db_path)
    committed = 0
    staged = 0
    rejected = 0
    with open(csv_path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            source = row.get('source') or 'csv'
            captured_at = row.get('captured_at')
            book = row.get('book') or default_book
            market_type = row.get('market_type')
            selection = row.get('selection')
            line = _clean_line_field(row.get('line'))
            odds = _clean_odds_field(row.get('odds'))
            team_home_raw = row.get('team_home') or row.get('home_team')
            team_away_raw = row.get('team_away') or row.get('away_team')
            game_date = row.get('game_date')
            game_id = row.get('game_id')

            # Basic validation
            if not selection or not team_home_raw or not team_away_raw:
                rejected += 1
                continue
            if odds is None or odds == 0 or odds < -1000 or odds > 1000:
                rejected += 1
                continue
            if line is None:
                rejected += 1
                continue

            if game_id:
                # insert directly to market_snapshots
                with closing(sqlite3.connect(Path(db_path))) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO market_snapshots (snapshot_run_id, captured_at, book, market_type, selection, line, odds, game_id, source_staging_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (snapshot_run_id, captured_at or _utcnow_iso(), book, market_type, selection, line, odds, game_id, None, _utcnow_iso()),
                    )
                    conn.commit()
                    committed += 1
            else:
                # attempt resolve
                res = resolve_staging_to_game(db_path, sport=sport, season=season, team_home_raw=team_home_raw, team_away_raw=team_away_raw, game_date=game_date)
                if res.get('match_status') == 'matched' and commit_matched:
                    with closing(sqlite3.connect(Path(db_path))) as conn:
                        conn.execute(
                            "INSERT OR REPLACE INTO market_snapshots (snapshot_run_id, captured_at, book, market_type, selection, line, odds, game_id, source_staging_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (snapshot_run_id, captured_at or _utcnow_iso(), book, market_type, selection, line, odds, res.get('game_id'), None, _utcnow_iso()),
                        )
                        conn.commit()
                        committed += 1
                else:
                    # insert into staging for review
                    add_staging_row(
                        db_path,
                        source=source,
                        captured_at=captured_at or _utcnow_iso(),
                        image_path=None,
                        raw_text=None,
                        book=book,
                        market_type=market_type,
                        selection=selection,
                        line=line,
                        odds=odds,
                        team_home_raw=team_home_raw,
                        team_away_raw=team_away_raw,
                        game_date=game_date,
                        match_status=res.get('match_status') if res else 'unmatched',
                        match_confidence=res.get('match_confidence') if res else 0.0,
                        game_id=res.get('game_id') if res else None,
                    )
                    staged += 1
    return {"committed": committed, "staged": staged, "rejected": rejected}


def get_opportunities_with_game_info(db_path: str | Path, *, review_run_id: str) -> list[dict]:
    """Return opportunities joined with game metadata for a review_run_id.

    Columns returned include: opportunity_id, review_run_id, game_id, market_type,
    selection, line, odds, implied_prob, model_prob, edge, ev, source_market_snapshot_id,
    created_at, date, home_team, away_team
    """
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        q = """
            SELECT
                o.id AS opportunity_id,
                o.review_run_id,
                o.game_id,
                o.market_type,
                o.selection,
                o.line,
                o.odds,
                o.implied_prob,
                o.model_prob,
                o.edge,
                o.ev,
                o.source_market_snapshot_id,
                o.created_at AS opportunity_created_at,
                g.date AS game_date,
                g.home_team,
                g.away_team
            FROM opportunities o
            LEFT JOIN games g ON o.game_id = g.game_id
            WHERE o.review_run_id = ?
            ORDER BY g.date, g.away_team, g.home_team
        """
        rows = conn.execute(q, (review_run_id,)).fetchall()
        cols = [c[0] for c in conn.execute("PRAGMA table_info(opportunities)").fetchall()]
        result = []
        for r in rows:
            result.append({
                "opportunity_id": r[0],
                "review_run_id": r[1],
                "game_id": r[2],
                "market_type": r[3],
                "selection": r[4],
                "line": r[5],
                "odds": r[6],
                "implied_prob": r[7],
                "model_prob": r[8],
                "edge": r[9],
                "ev": r[10],
                "source_market_snapshot_id": r[11],
                "created_at": r[12],
                "date": r[13],
                "home_team": r[14],
                "away_team": r[15],
            })
        return result


def add_prediction_exclusions(
    db_path: str | Path,
    *,
    review_run_id: str,
    exclusions: Iterable[dict],
    created_at: str | None = None,
) -> int:
    """Insert prediction exclusions for a review run. Returns inserted count."""
    rows = list(exclusions)
    if not rows:
        return 0
    init_db(db_path)
    timestamp = created_at or _utcnow_iso()
    inserted = 0
    with closing(sqlite3.connect(Path(db_path))) as conn:
        for row in rows:
            conn.execute(
                """
                INSERT INTO prediction_exclusions (
                    review_run_id, game_id, model, excluded_reason, created_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(review_run_id, game_id, model, excluded_reason) DO UPDATE SET
                    created_at = excluded.created_at
                """,
                (
                    review_run_id,
                    row.get("game_id"),
                    row.get("model"),
                    row.get("excluded_reason"),
                    timestamp,
                ),
            )
            inserted += 1
        conn.commit()
    return inserted


def get_prediction_exclusions(db_path: str | Path, *, review_run_id: str) -> list[dict]:
    """Return prediction exclusions for the given review run."""
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        rows = conn.execute(
            """
            SELECT game_id, model, excluded_reason
            FROM prediction_exclusions
            WHERE review_run_id = ?
            ORDER BY game_id, model, excluded_reason
            """,
            (review_run_id,),
        ).fetchall()
        return [
            {"game_id": r[0], "model": r[1], "excluded_reason": r[2]}
            for r in rows
        ]


def add_clv_snapshot(
    db_path: str | Path,
    *,
    game_id: str,
    market_type: str,
    selection: str,
    close_line: float | None,
    close_odds: int | None,
    captured_at: str | None = None,
) -> int:
    """Persist a close-line snapshot for a game/market.

    Returns the inserted snapshot id.
    """
    init_db(db_path)
    if captured_at is None:
        captured_at = _utcnow_iso()
    with closing(sqlite3.connect(Path(db_path))) as conn:
        cur = conn.execute(
            "INSERT INTO clv_snapshots (game_id, market_type, selection, close_line, close_odds, captured_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (game_id, market_type, selection, close_line, close_odds, captured_at, _utcnow_iso()),
        )
        conn.commit()
        return cur.lastrowid


def get_latest_clv(db_path: str | Path, *, game_id: str, market_type: str, selection: str) -> dict | None:
    """Return the most recent CLV snapshot for the given keys, or None."""
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        cur = conn.execute(
            "SELECT close_line, close_odds, captured_at FROM clv_snapshots WHERE game_id = ? AND market_type = ? AND selection = ? ORDER BY datetime(captured_at) DESC LIMIT 1",
            (game_id, market_type, selection),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {"close_line": row[0], "close_odds": row[1], "captured_at": row[2]}


def update_bets_with_clv(
    db_path: str | Path,
    *,
    game_id: str | None = None,
    market_type: str | None = None,
    selection: str | None = None,
) -> int:
    """Apply latest CLV snapshots to bets for the given filters.

    Filters are optional; when omitted the latest snapshot per key is applied
    to all bets that have a matching game/market/selection.
    Returns the number of bet rows updated.
    """
    init_db(db_path)
    updated = 0
    with closing(sqlite3.connect(Path(db_path))) as conn:
        # Collect latest snapshot per key (ordered by captured_at then id for stability)
        q = """
            SELECT game_id, market_type, selection, close_line, close_odds
            FROM (
                SELECT game_id, market_type, selection, close_line, close_odds, captured_at, id,
                       ROW_NUMBER() OVER(PARTITION BY game_id, market_type, selection ORDER BY datetime(captured_at) DESC, id DESC) AS rn
                FROM clv_snapshots
        """
        params: list = []
        clauses: list[str] = []
        if game_id:
            clauses.append("game_id = ?")
            params.append(game_id)
        if market_type:
            clauses.append("market_type = ?")
            params.append(market_type)
        if selection:
            clauses.append("selection = ?")
            params.append(selection)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += """
            ) WHERE rn = 1
        """
        latest = conn.execute(q, params).fetchall()
        for gid, mtype, sel, cline, codds in latest:
            res = conn.execute(
                "UPDATE bets SET clv_close_line = ?, clv_close_odds = ? WHERE game_id = ? AND market_type = ? AND selection = ?",
                (cline, codds, gid, mtype, sel),
            )
            if res.rowcount is not None and res.rowcount > 0:
                updated += int(res.rowcount)
        conn.commit()
    return updated


def import_clv_csv(
    db_path: str | Path,
    *,
    csv_path: str | Path,
    sport: str,
    season: str,
    default_market_type: str | None = None,
    default_captured_at: str | None = None,
    update_bets: bool = True,
) -> dict:
    """Import close-line snapshots from a CSV.

    Expected columns (some optional):
    - market_type (ML/spread/total) [optional if default_market_type provided]
    - selection
    - close_line (optional for ML)
    - close_odds
    - team_home/home_team (for match resolution when game_id absent)
    - team_away/away_team
    - game_date (YYYY-MM-DD)
    - game_id (optional; skips resolution)
    - captured_at (optional; defaults to current UTC)

    Returns summary counts: {"snapshots": N, "bets_updated": M, "rejected": K}
    """
    import csv

    init_db(db_path)
    snapshots = 0
    bets_updated = 0
    rejected = 0
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            market_type = row.get("market_type") or default_market_type
            selection = row.get("selection")
            close_line = _clean_line_field(row.get("close_line") or row.get("line"))
            close_odds = _clean_odds_field(row.get("close_odds") or row.get("odds"))
            captured_at = row.get("captured_at") or default_captured_at
            game_id = row.get("game_id")
            team_home_raw = row.get("team_home") or row.get("home_team")
            team_away_raw = row.get("team_away") or row.get("away_team")
            game_date = row.get("game_date")

            # Basic validation
            if not selection or not market_type:
                rejected += 1
                continue
            if close_odds is None or close_odds == 0 or close_odds < -1000 or close_odds > 1000:
                rejected += 1
                continue

            resolved_game_id = game_id
            if not resolved_game_id:
                res = resolve_staging_to_game(
                    db_path,
                    sport=sport,
                    season=season,
                    team_home_raw=team_home_raw,
                    team_away_raw=team_away_raw,
                    game_date=game_date,
                )
                if not res or res.get("match_status") != "matched" or not res.get("game_id"):
                    rejected += 1
                    continue
                resolved_game_id = res.get("game_id")

            add_clv_snapshot(
                db_path,
                game_id=resolved_game_id,
                market_type=market_type,
                selection=selection,
                close_line=close_line,
                close_odds=close_odds,
                captured_at=captured_at,
            )
            snapshots += 1

            if update_bets:
                bets_updated += update_bets_with_clv(
                    db_path,
                    game_id=resolved_game_id,
                    market_type=market_type,
                    selection=selection,
                )

    return {"snapshots": snapshots, "bets_updated": bets_updated, "rejected": rejected}
