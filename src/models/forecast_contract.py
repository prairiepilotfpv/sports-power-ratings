"""Game-level forecast contract for matchup-level markets.

The rankings contract remains team-level and rating-oriented; the forecast
contract focuses on per-game, per-market payloads so downstream pipelines can
consume native matchup outputs when available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any, Mapping, Literal

from models.base import validate_probability

ForecastSource = Literal["native", "derived", "missing"]

NORMAL_FAMILY = "normal"


@dataclass(frozen=True)
class MLForecast:
    """Matchup-level ML forecast payload."""

    p_home_win: float
    p_away_win: float | None = None
    source: ForecastSource = "native"

    def __post_init__(self) -> None:
        validate_probability(self.p_home_win, field_name="p_home_win")
        if self.p_away_win is not None:
            validate_probability(self.p_away_win, field_name="p_away_win")


@dataclass(frozen=True)
class SpreadForecast:
    """Spread forecast payload (margin distribution)."""

    margin_mean: float
    margin_sd: float
    dist_family: str = NORMAL_FAMILY
    source: ForecastSource = "native"

    def __post_init__(self) -> None:
        object.__setattr__(self, "margin_mean", float(self.margin_mean))
        sd = float(self.margin_sd)
        if sd <= 0 or not math.isfinite(sd):
            raise ValueError("margin_sd must be a positive finite number.")
        object.__setattr__(self, "margin_sd", sd)


@dataclass(frozen=True)
class TotalForecast:
    """Total forecast payload (combined points distribution)."""

    total_mean: float
    total_sd: float
    dist_family: str = NORMAL_FAMILY
    source: ForecastSource = "native"

    def __post_init__(self) -> None:
        object.__setattr__(self, "total_mean", float(self.total_mean))
        sd = float(self.total_sd)
        if sd <= 0 or not math.isfinite(sd):
            raise ValueError("total_sd must be a positive finite number.")
        object.__setattr__(self, "total_sd", sd)


@dataclass(frozen=True)
class ForecastContract:
    """Per-game, per-market forecast payload with provenance."""

    game_id: str | None
    date: str
    home_team: str
    away_team: str
    model_id: str
    model_version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    ml: MLForecast | None = None
    spread: SpreadForecast | None = None
    total: TotalForecast | None = None
    source_ml: ForecastSource = "missing"
    source_spread: ForecastSource = "missing"
    source_total: ForecastSource = "missing"
    projected_home_score: float | None = None
    projected_away_score: float | None = None
    projected_total: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
