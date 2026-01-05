"""Normalization helpers for parsed game results."""

from __future__ import annotations

from typing import Iterable, List

from ingest.schema import GameResult


def normalize_games(
    games: Iterable[GameResult],
    sport: str | None = None,
    season: str | None = None,
    division: str | None = None,
    conference: str | None = None,
) -> List[GameResult]:
    """Normalize GameResult entries and backfill sport/season defaults."""
    normalized: List[GameResult] = []
    for game in games:
        # Skip rows without core identity fields.
        if not game.home_team or not game.away_team or not game.date:
            continue
        normalized.append(
            GameResult(
                date=game.date,
                home_team=game.home_team.strip(),
                away_team=game.away_team.strip(),
                home_score=game.home_score,
                away_score=game.away_score,
                neutral=game.neutral,
                overtime=game.overtime,
                game_id=game.game_id,
                sport=game.sport or sport,
                season=game.season or season,
                division=game.division or division,
                conference=game.conference or conference,
                notes=game.notes,
            )
        )
    return normalized
