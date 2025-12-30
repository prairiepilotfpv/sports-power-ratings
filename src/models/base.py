from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import isnan
from typing import Any, Iterable, Mapping, Protocol


@dataclass(frozen=True)
class ModelMetadata:
    name: str
    version: str
    supports_margin: bool
    supports_total: bool
    supports_win_prob: bool


@dataclass
class GamePrediction:
    game_id: str
    date: str
    home_team: str
    away_team: str
    p_home_win: float
    pred_margin: float | None = None
    pred_total: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.p_home_win = validate_probability(self.p_home_win, field_name="p_home_win")
        self.pred_margin = normalize_optional_float(self.pred_margin)
        self.pred_total = normalize_optional_float(self.pred_total)


class BaseModel(ABC):
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        raise NotImplementedError

    @abstractmethod
    def fit(self, games_df: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, upcoming_games_df: Any) -> list[GamePrediction]:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> BaseModel:
        raise NotImplementedError


class PowerRatingModel(Protocol):
    """Protocol for power rating models."""

    def fit(self, games: Iterable[Mapping[str, Any]]) -> None:
        """Fit the model on iterable game results."""

    def rankings(self) -> list[tuple[str, float]]:
        """Return sorted team ratings, high to low."""


def validate_probability(value: float | None, *, field_name: str = "probability") -> float:
    if value is None:
        raise ValueError(f"{field_name} is required.")
    try:
        prob = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a float between 0 and 1.") from exc
    if prob < 0.0 or prob > 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1, got {prob}.")
    return prob


def normalize_optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and isnan(value):
        return None
    return float(value)


def require_columns(df: Any, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in getattr(df, "columns", [])]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
