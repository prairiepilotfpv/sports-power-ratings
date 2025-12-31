from __future__ import annotations

from typing import Dict, Type

from ingest.base import IngestSource
from ingest.sources import SportsReferenceSource


_REGISTRY: Dict[str, Type[IngestSource]] = {
    SportsReferenceSource.name: SportsReferenceSource,
}


def get_ingest_source(name: str) -> Type[IngestSource]:
    normalized = name.strip().lower()
    try:
        return _REGISTRY[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported ingest source: {name}") from exc


def list_ingest_sources() -> list[str]:
    return sorted(_REGISTRY.keys())


def register_ingest_source(name: str, source_cls: Type[IngestSource]) -> None:
    """Register a new ingest source (primarily for tests)."""
    _REGISTRY[name.strip().lower()] = source_cls


def unregister_ingest_source(name: str) -> None:
    """Remove an ingest source from the registry."""
    _REGISTRY.pop(name.strip().lower(), None)
