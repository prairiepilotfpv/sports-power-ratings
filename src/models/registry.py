from __future__ import annotations

from typing import Type

from models.base import PowerRatingModel
from models.bradley_terry import BradleyTerry


_REGISTRY: dict[str, Type[PowerRatingModel]] = {
    "bradley-terry": BradleyTerry,
}


def get_model(name: str) -> Type[PowerRatingModel]:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {name}") from exc


def list_models() -> list[str]:
    return sorted(_REGISTRY.keys())
