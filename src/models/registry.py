from __future__ import annotations

import re
from typing import Type

from models.base import BaseModel, PowerRatingModel
from models.bradley_terry import BradleyTerry
from models.bradley_terry_hfa import BradleyTerryHFA
from models.gssd import GSSDPowerRating
from models.toor import TOORModel, TOORPowerRating


_REGISTRY: dict[str, Type[PowerRatingModel]] = {
    "bradley-terry": BradleyTerry,
    "gssd": GSSDPowerRating,
    "toor": TOORPowerRating,
}

_MODEL_ABBREVIATIONS: dict[str, str] = {
    "bradley-terry": "bt",
    "gssd": "gssd",
    "toor": "toor",
}

_BACKTEST_REGISTRY: dict[str, Type[BaseModel]] = {
    "bradley_terry_hfa": BradleyTerryHFA,
    "toor": TOORModel,
}


def get_model(name: str) -> Type[PowerRatingModel]:
    name = normalize_model_name(name)
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {name}") from exc


def list_models() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_model_abbreviation(name: str) -> str:
    name = normalize_model_name(name)
    if name in _MODEL_ABBREVIATIONS:
        return _MODEL_ABBREVIATIONS[name]
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", name) if part]
    if not parts:
        return name
    return "".join(part[0] for part in parts).lower()


def get_backtest_model(name: str) -> Type[BaseModel]:
    name = normalize_model_name(name)
    try:
        return _BACKTEST_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported backtest model: {name}") from exc


def list_backtest_models() -> list[str]:
    return sorted(_BACKTEST_REGISTRY.keys())


def normalize_model_name(name: str) -> str:
    return name.strip().lower()
