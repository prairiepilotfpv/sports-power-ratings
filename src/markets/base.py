from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List


class Market(str, Enum):
    """Prediction target for a betting market.

    A "market" is the type of line being modeled (ML, spread, total). We keep
    ensemble weights and calibrators scoped per-market so future tuning can
    evolve independently without changing workbook contracts.
    """

    ML = "ML"
    SPREAD = "spread"
    TOTAL = "total"


@dataclass(frozen=True)
class MarketSpec:
    """Market-specific contracts for prediction inputs and artifacts."""

    market: Market
    required_fields: List[str]
    primary_metric_name: str

    def ensemble_weights_path(self, sport: str, season: str, ensemble_id: str) -> Path:
        # Store ensemble weights per-market to keep tuning isolated by target.
        return (
            Path("outputs")
            / "ensembles"
            / sport
            / season
            / self.market.value
            / f"{ensemble_id}.json"
        )

    def calibrator_dir(self, sport: str, season: str, source_id: str) -> Path:
        # Market-specific calibrator directory. Falls under outputs/calibrators.
        return Path("outputs") / "calibrators" / sport / season / source_id / self.market.value
