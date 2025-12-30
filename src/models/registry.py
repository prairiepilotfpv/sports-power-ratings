from __future__ import annotations

from typing import Type

from models.base import BaseModel, PowerRatingModel
from models.bradley_terry import BradleyTerry
from models.bradley_terry_hfa import BradleyTerryHFA


_REGISTRY: dict[str, Type[PowerRatingModel]] = {
    "bradley-terry": BradleyTerry,
}

_BACKTEST_REGISTRY: dict[str, Type[BaseModel]] = {
    "bradley_terry_hfa": BradleyTerryHFA,
}


def get_model(name: str) -> Type[PowerRatingModel]:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {name}") from exc


def list_models() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_backtest_model(name: str) -> Type[BaseModel]:
    try:
        return _BACKTEST_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported backtest model: {name}") from exc


def list_backtest_models() -> list[str]:
    return sorted(_BACKTEST_REGISTRY.keys())
