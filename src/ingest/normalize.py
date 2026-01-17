"""Normalization helpers for parsed game results."""

from __future__ import annotations

from typing import Iterable, List

from ingest.schema import GameResult
from src.utils.game_id import make_game_id


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
        # backfill sport/season and compute deterministic game_id when missing
        sport_val = game.sport or sport
        season_val = game.season or season
        home_clean = game.home_team.strip()
        away_clean = game.away_team.strip()
        computed_game_id = game.game_id
        if not computed_game_id and sport_val and season_val:
            try:
                computed_game_id = make_game_id(sport_val, season_val, game.date, away_clean, home_clean)
            except Exception:
                computed_game_id = None

        normalized.append(
            GameResult(
                date=game.date,
                start_time=game.start_time,
                home_team=home_clean,
                away_team=away_clean,
                home_score=game.home_score,
                away_score=game.away_score,
                neutral=game.neutral,
                overtime=game.overtime,
                decision_type=game.decision_type,
                game_id=computed_game_id,
                sport=sport_val,
                season=season_val,
                division=game.division or division,
                conference=game.conference or conference,
                notes=game.notes,
            )
        )
    return normalized
