from __future__ import annotations

import re
from typing import Type

from models.base import BaseModel, PowerRatingModel
from models.bradley_terry import BradleyTerry
from models.bradley_terry_hfa import BradleyTerryHFA
from models.toor import TOORModel, TOORPowerRating


_REGISTRY: dict[str, Type[PowerRatingModel]] = {
    "bradley-terry": BradleyTerry,
    "toor": TOORPowerRating,
}

_MODEL_ABBREVIATIONS: dict[str, str] = {
    "bradley-terry": "bt",
    "toor": "toor",
}

_BACKTEST_REGISTRY: dict[str, Type[BaseModel]] = {
    "bradley_terry_hfa": BradleyTerryHFA,
    "toor": TOORModel,
}


def get_model(name: str) -> Type[PowerRatingModel]:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {name}") from exc


def list_models() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_model_abbreviation(name: str) -> str:
    if name in _MODEL_ABBREVIATIONS:
        return _MODEL_ABBREVIATIONS[name]
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", name) if part]
    if not parts:
        return name
    return "".join(part[0] for part in parts).lower()


def get_backtest_model(name: str) -> Type[BaseModel]:
    try:
        return _BACKTEST_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported backtest model: {name}") from exc


def list_backtest_models() -> list[str]:
    return sorted(_BACKTEST_REGISTRY.keys())
