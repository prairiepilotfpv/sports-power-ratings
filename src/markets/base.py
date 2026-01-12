from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List


class Market(str, Enum):
    ML = "ML"
    SPREAD = "spread"
    TOTAL = "total"


@dataclass(frozen=True)
class MarketSpec:
    market: Market
    required_fields: List[str]
    primary_metric_name: str

    def ensemble_weights_path(self, sport: str, season: str, ensemble_id: str) -> Path:
        return Path("outputs") / "ensembles" / sport / season / f"{ensemble_id}.json"

    def calibrator_dir(self, sport: str, season: str, source_id: str) -> Path:
        # Market-specific calibrator directory. Falls under outputs/calibrators.
        return Path("outputs") / "calibrators" / sport / season / source_id / self.market.value
