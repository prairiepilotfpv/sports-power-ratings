"""Team identity helpers for canonicalizing names and managing aliases."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Iterable

_ALIAS_CONFIG_PATH = Path("data/config/team_aliases.json")
_ALIAS_CACHE: dict | None = None
_ALIAS_CLEANER = re.compile(r"[^a-z0-9 ]+")


def _normalize_alias_text(value: str | None) -> str | None:
    if not value:
        return None
    normalized = str(value).strip().lower()
    normalized = _ALIAS_CLEANER.sub("", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip() or None


def _load_alias_config() -> dict[str, dict[str, list[str]]]:
    global _ALIAS_CACHE
    if _ALIAS_CACHE is not None:
        return _ALIAS_CACHE
    if not _ALIAS_CONFIG_PATH.exists():
        _ALIAS_CACHE = {}
        return _ALIAS_CACHE
    try:
        data = json.loads(_ALIAS_CONFIG_PATH.read_text())
    except Exception:
        data = {}
    _ALIAS_CACHE = data
    return _ALIAS_CACHE


def ensure_team(
    conn: sqlite3.Connection,
    *,
    sport: str,
    season: str,
    canonical_name: str,
    source: str = "schedule",
) -> int | None:
    name = str(canonical_name).strip()
    if not name:
        return None
    row = conn.execute(
        "SELECT id FROM teams WHERE sport = ? AND season = ? AND canonical_name = ?",
        (sport, season, name),
    ).fetchone()
    if row:
        team_id = row[0]
    else:
        cur = conn.execute(
            "INSERT INTO teams (sport, season, canonical_name) VALUES (?, ?, ?)",
            (sport, season, name),
        )
        team_id = cur.lastrowid
    ensure_alias(conn, sport=sport, season=season, team_id=team_id, alias_text=name, source=source)
    return team_id


def ensure_alias(
    conn: sqlite3.Connection,
    *,
    sport: str,
    season: str,
    team_id: int,
    alias_text: str,
    source: str,
) -> None:
    normalized = _normalize_alias_text(alias_text)
    if not normalized:
        return
    conn.execute(
        "INSERT OR IGNORE INTO team_aliases (sport, season, team_id, alias_text, source) VALUES (?, ?, ?, ?, ?)",
        (sport, season, team_id, normalized, source),
    )


def resolve_team_id(
    conn: sqlite3.Connection, *, sport: str, season: str, raw_team_name: str | None
) -> int | None:
    if not raw_team_name:
        return None
    normalized = _normalize_alias_text(raw_team_name)
    if not normalized:
        return None
    row = conn.execute(
        "SELECT team_id FROM team_aliases WHERE sport = ? AND season = ? AND alias_text = ?",
        (sport, season, normalized),
    ).fetchone()
    return row[0] if row else None


def get_team_id(
    conn: sqlite3.Connection, *, sport: str, season: str, canonical_name: str | None
) -> int | None:
    if not canonical_name:
        return None
    row = conn.execute(
        "SELECT id FROM teams WHERE sport = ? AND season = ? AND canonical_name = ?",
        (sport, season, canonical_name.strip()),
    ).fetchone()
    return row[0] if row else None


def get_canonical_name(conn: sqlite3.Connection, *, team_id: int | None) -> str | None:
    if not team_id:
        return None
    row = conn.execute(
        "SELECT canonical_name FROM teams WHERE id = ?",
        (int(team_id),),
    ).fetchone()
    return row[0] if row else None


def ensure_aliases_from_config(conn: sqlite3.Connection, *, sport: str, season: str) -> None:
    data = _load_alias_config()
    sport_key = sport.lower()
    aliases_for_sport = data.get(sport_key) or {}
    if not aliases_for_sport:
        return
    for canonical, alias_list in aliases_for_sport.items():
        team_id = get_team_id(conn, sport=sport, season=season, canonical_name=canonical)
        if not team_id:
            continue
        ensure_alias(conn, sport=sport, season=season, team_id=team_id, alias_text=canonical, source="config")
        if isinstance(alias_list, list):
            for alias in alias_list:
                ensure_alias(conn, sport=sport, season=season, team_id=team_id, alias_text=alias, source="config")


def ensure_teams_for_games(conn: sqlite3.Connection, games: Iterable) -> None:
    pending: dict[tuple[str, str], set[str]] = {}
    for game in games:
        sport = getattr(game, "sport", None)
        season = getattr(game, "season", None)
        if not sport or not season:
            continue
        key = (sport, season)
        names = pending.setdefault(key, set())
        home = getattr(game, "home_team", None)
        away = getattr(game, "away_team", None)
        if home:
            names.add(home)
        if away:
            names.add(away)
    for (sport, season), names in pending.items():
        for name in names:
            ensure_team(conn, sport=sport, season=season, canonical_name=name, source="schedule")
        ensure_aliases_from_config(conn, sport=sport, season=season)
