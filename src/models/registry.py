from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import re
from typing import Type

from models.base import BaseModel, PowerRatingModel
from models.bradley_terry import BradleyTerryBacktest
from models.bradley_terry_calibrated_hfa import BradleyTerryCalibratedHFA
from models.bradley_terry_hfa import BradleyTerryHFA
from models.elo import EloModel
from models.gssd import GSSDModel
from models.poisson import PoissonModel
from models.toor import TOORModel


def normalize_model_name(name: str) -> str:
    return name.strip().lower()


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: str
    abbreviation: str
    is_experimental: bool = False
    is_deprecated: bool = False
    tags: tuple[str, ...] = ()
    required_module: str | None = None


_MODEL_SPECS: dict[str, ModelSpec] = {
    "bradley-terry": ModelSpec(
        name="bradley-terry",
        path="models.bradley_terry.BradleyTerry",
        abbreviation="bt",
    ),
    "elo": ModelSpec(
        name="elo",
        path="models.elo.EloPowerRating",
        abbreviation="elo",
    ),
    "gssd": ModelSpec(
        name="gssd",
        path="models.gssd.GSSDPowerRating",
        abbreviation="gssd",
        required_module="ssat",
    ),
    "poisson": ModelSpec(
        name="poisson",
        path="models.poisson.PoissonPowerRating",
        abbreviation="pois",
    ),
    "toor": ModelSpec(
        name="toor",
        path="models.toor.TOORPowerRating",
        abbreviation="toor",
    ),
}

_DYNAMIC_MODELS: dict[str, Type[PowerRatingModel]] = {}
_DYNAMIC_ABBREVIATIONS: dict[str, str] = {}

_BACKTEST_REGISTRY: dict[str, Type[BaseModel]] = {
    "bradley-terry": BradleyTerryBacktest,
    "bradley_terry_hfa": BradleyTerryHFA,
    "bradley_terry_calibrated_hfa": BradleyTerryCalibratedHFA,
    "elo": EloModel,
    "gssd": GSSDModel,
    "poisson": PoissonModel,
    "toor": TOORModel,
}


@dataclass(frozen=True)
class ModelMetadata:
    is_experimental: bool = False
    is_deprecated: bool = False
    tags: tuple[str, ...] = ()


def _default_experimental_flag(name: str) -> bool:
    normalized = normalize_model_name(name)
    # Treat any variant that references HFA explicitly as experimental by default.
    return "hfa" in normalized


_MODEL_METADATA: dict[str, ModelMetadata] = {}
for spec_name, spec in _MODEL_SPECS.items():
    _MODEL_METADATA[spec_name] = ModelMetadata(
        is_experimental=spec.is_experimental or _default_experimental_flag(spec_name),
        is_deprecated=spec.is_deprecated,
        tags=spec.tags,
    )

_BACKTEST_METADATA: dict[str, ModelMetadata] = {}
for name in _BACKTEST_REGISTRY.keys():
    _BACKTEST_METADATA[name] = ModelMetadata(
        is_experimental=_default_experimental_flag(name),
        tags=("backtest",),
    )


def _model_available(spec: ModelSpec) -> bool:
    if spec.required_module is None:
        return True
    return importlib.util.find_spec(spec.required_module) is not None


def _load_model_class(spec: ModelSpec) -> Type[PowerRatingModel]:
    module_name, class_name = spec.path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_model(name: str) -> Type[PowerRatingModel]:
    name = normalize_model_name(name)
    if name in _DYNAMIC_MODELS:
        return _DYNAMIC_MODELS[name]
    if name not in _MODEL_SPECS:
        raise ValueError(f"Unsupported model: {name}")
    spec = _MODEL_SPECS[name]
    if not _model_available(spec):
        raise ValueError(
            f"Model {name!r} requires optional dependency '{spec.required_module}'."
        )
    return _load_model_class(spec)


def list_models() -> list[str]:
    static = [name for name, spec in _MODEL_SPECS.items() if _model_available(spec)]
    return sorted(set(static + list(_DYNAMIC_MODELS.keys())))


def get_model_metadata(name: str) -> ModelMetadata:
    normalized = normalize_model_name(name)
    if normalized in _MODEL_METADATA:
        return _MODEL_METADATA[normalized]
    if normalized in _BACKTEST_METADATA:
        return _BACKTEST_METADATA[normalized]
    return ModelMetadata(is_experimental=_default_experimental_flag(normalized))


def is_experimental_model(name: str) -> bool:
    return get_model_metadata(name).is_experimental


def get_model_abbreviation(name: str) -> str:
    name = normalize_model_name(name)
    if name in _DYNAMIC_ABBREVIATIONS:
        return _DYNAMIC_ABBREVIATIONS[name]
    spec = _MODEL_SPECS.get(name)
    if spec is not None:
        return spec.abbreviation
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


def register_model(
    name: str,
    model_cls: Type[PowerRatingModel],
    *,
    abbreviation: str | None = None,
    metadata: ModelMetadata | None = None,
) -> None:
    """Register a power rating model (primarily for tests)."""
    normalized = normalize_model_name(name)
    _DYNAMIC_MODELS[normalized] = model_cls
    if abbreviation:
        _DYNAMIC_ABBREVIATIONS[normalized] = abbreviation
    resolved_metadata = metadata or ModelMetadata(
        is_experimental=_default_experimental_flag(normalized)
    )
    _MODEL_METADATA[normalized] = resolved_metadata


def unregister_model(name: str) -> None:
    """Remove a dynamically registered model."""
    normalized = normalize_model_name(name)
    _DYNAMIC_MODELS.pop(normalized, None)
    _DYNAMIC_ABBREVIATIONS.pop(normalized, None)
    _MODEL_METADATA.pop(normalized, None)
