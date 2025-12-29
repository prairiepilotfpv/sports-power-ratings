from __future__ import annotations

from typing import Iterable, List

from ingest.schema import GameResult


def normalize_games(
    games: Iterable[GameResult],
    sport: str | None = None,
    season: str | None = None,
) -> List[GameResult]:
    normalized: List[GameResult] = []
    for game in games:
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
                notes=game.notes,
            )
        )
    return normalized
