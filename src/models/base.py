"""Base interfaces and helpers for prediction models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping, Protocol


REQUIRED_PREDICTION_METADATA_KEYS = ("model_id", "model_version", "params")


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata describing a model's capabilities."""

    model_id: str
    model_version: str
    params: Mapping[str, Any]
    supports_margin: bool
    supports_total: bool
    supports_win_prob: bool

    def identity_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "params": dict(self.params),
        }


@dataclass
class GamePrediction:
    """Unified prediction record for downstream pipelines."""

    game_id: str
    date: str
    home_team: str
    away_team: str
    p_home_win: float
    win_prob_dist: list[dict[str, float]] | None = None
    pred_margin: float | None = None
    pred_total: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.p_home_win = validate_probability(self.p_home_win, field_name="p_home_win")
        self.win_prob_dist = validate_win_prob_dist(self.win_prob_dist)
        self.pred_margin = normalize_optional_float(self.pred_margin)
        self.pred_total = normalize_optional_float(self.pred_total)
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dict.")
        missing = [
            key for key in REQUIRED_PREDICTION_METADATA_KEYS if key not in self.metadata
        ]
        if missing:
            raise ValueError(f"metadata missing required keys: {', '.join(missing)}")


class BaseModel(ABC):
    """Base interface for predictive models in the system."""

    @abstractmethod
    def metadata(self) -> ModelMetadata:
        raise NotImplementedError

    @property
    def model_id(self) -> str:
        return self.metadata().model_id

    @property
    def model_version(self) -> str:
        return self.metadata().model_version

    @property
    def params(self) -> Mapping[str, Any]:
        return self.metadata().params

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

    def metadata(self) -> ModelMetadata:
        """Return metadata describing the model."""

    def fit(self, games: Iterable[Mapping[str, Any]]) -> None:
        """Fit the model on iterable game results."""

    def rankings(self) -> list[tuple[str, float]]:
        """Return sorted team ratings, high to low."""


def resolve_model_identity(model: Any) -> dict[str, Any]:
    """Extract required metadata fields from a model instance."""
    if hasattr(model, "metadata") and callable(getattr(model, "metadata")):
        meta = model.metadata()
        if isinstance(meta, ModelMetadata):
            return meta.identity_dict()
        if isinstance(meta, dict):
            return {
                "model_id": meta.get("model_id"),
                "model_version": meta.get("model_version"),
                "params": meta.get("params"),
            }

    model_id = getattr(model, "model_id", None)
    model_version = getattr(model, "model_version", None)
    params = getattr(model, "params", None)
    if callable(params):
        params = params()

    if model_id is None or model_version is None or params is None:
        raise ValueError("Model is missing required identity fields.")
    return {
        "model_id": model_id,
        "model_version": model_version,
        "params": params,
    }


def validate_probability(
    value: float | None, *, field_name: str = "probability"
) -> float:
    """Ensure probabilities are valid floats in [0, 1]."""
    if value is None:
        raise ValueError(f"{field_name} is required.")
    try:
        prob = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a float between 0 and 1.") from exc
    if prob < 0.0 or prob > 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1, got {prob}.")
    return prob


def validate_win_prob_dist(
    dist: list[dict[str, float]] | None,
    *,
    field_name: str = "win_prob_dist",
) -> list[dict[str, float]] | None:
    """Validate and normalize a win-probability distribution."""
    if dist is None:
        return None
    if not isinstance(dist, list) or not dist:
        raise ValueError(f"{field_name} must be a non-empty list.")
    normalized: list[dict[str, float]] = []
    total = 0.0
    for bucket in dist:
        if not isinstance(bucket, Mapping):
            raise ValueError(f"{field_name} entries must be mappings.")
        if "p_home_win" not in bucket or "weight" not in bucket:
            raise ValueError(
                f"{field_name} entries must include 'p_home_win' and 'weight'."
            )
        p_home_win = validate_probability(
            bucket.get("p_home_win"), field_name="p_home_win"
        )
        try:
            weight = float(bucket.get("weight"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} weight must be a float.") from exc
        if weight < 0:
            raise ValueError(f"{field_name} weight must be non-negative.")
        if not math.isfinite(weight):
            raise ValueError(f"{field_name} weight must be finite.")
        normalized.append({"p_home_win": p_home_win, "weight": weight})
        total += weight
    if total <= 0:
        raise ValueError(f"{field_name} weights must sum to a positive value.")
    normalized = sorted(normalized, key=lambda item: item["p_home_win"])
    if abs(total - 1.0) > 1e-6:
        normalized = [
            {"p_home_win": item["p_home_win"], "weight": item["weight"] / total}
            for item in normalized
        ]
    return normalized


def normalize_optional_float(value: float | None) -> float | None:
    """Normalize NaN floats to None for easier downstream checks."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


def require_columns(df: Any, required: Iterable[str]) -> None:
    """Validate that a DataFrame-like object has required columns."""
    missing = [
        column for column in required if column not in getattr(df, "columns", [])
    ]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
