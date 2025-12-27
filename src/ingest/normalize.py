from __future__ import annotations

from typing import Iterable, List

from ingest.schema import GameResult


def normalize_games(games: Iterable[GameResult]) -> List[GameResult]:
    normalized: List[GameResult] = []
    for game in games:
        if not game.home_team or not game.away_team:
            continue
        if game.home_score is None or game.away_score is None:
            continue
        normalized.append(game)
    return normalized
