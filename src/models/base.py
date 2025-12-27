from __future__ import annotations

from typing import Any, Iterable, Mapping, Protocol


class PowerRatingModel(Protocol):
    """Protocol for power rating models."""

    def fit(self, games: Iterable[Mapping[str, Any]]) -> None:
        """Fit the model on iterable game results."""

    def predict(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        """Predict win probability and point spread for the home team."""

    def rankings(self) -> list[tuple[str, float]]:
        """Return sorted team ratings, high to low."""
