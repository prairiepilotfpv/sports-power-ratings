"""Base types for ensembles."""

from __future__ import annotations

from typing import Protocol

from markets.base import Market


class BaseEnsemble(Protocol):
    @property
    def ensemble_id(self) -> str:
        ...

    @property
    def market(self) -> Market:
        ...
