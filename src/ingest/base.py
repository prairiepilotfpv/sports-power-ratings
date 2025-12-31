"""Interfaces for ingest sources."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ingest.schema import GameResult


class IngestSource(Protocol):
    """Protocol describing an ingest source."""

    name: str

    def load_path(
        self,
        path: str | Path,
        *,
        sport: str | None = None,
        season: str | None = None,
        format_hint: str | None = None,
    ) -> list[GameResult]:
        """Load games from a file path."""

    def load_text(
        self,
        text: str,
        *,
        sport: str | None = None,
        season: str | None = None,
    ) -> list[GameResult]:
        """Load games from raw text."""
