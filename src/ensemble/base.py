"""Base types for ensembles."""

from __future__ import annotations

from enum import Enum
from typing import Protocol


class Market(str, Enum):
    ML = "ML"
    SPREAD = "spread"
    TOTAL = "total"


class BaseEnsemble(Protocol):
    @property
    def ensemble_id(self) -> str:
        ...

    @property
    def market(self) -> Market:
        ...
