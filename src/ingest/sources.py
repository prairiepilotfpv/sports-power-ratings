"""Concrete ingest sources."""

from __future__ import annotations

from pathlib import Path

from ingest.schema import GameResult
from ingest.sports_reference import parse_sr_csv, parse_sr_csv_text, parse_sr_html


class SportsReferenceSource:
    """Ingest source for Sports-Reference CSV/HTML exports."""

    name = "sports-reference"

    def load_path(
        self,
        path: str | Path,
        *,
        sport: str | None = None,
        season: str | None = None,
        format_hint: str | None = None,
    ) -> list[GameResult]:
        resolved = Path(path)
        if format_hint == "html" or resolved.suffix.lower() in {".html", ".htm"}:
            return parse_sr_html(resolved, sport=sport, season=season)
        return parse_sr_csv(resolved, sport=sport, season=season)

    def load_text(
        self,
        text: str,
        *,
        sport: str | None = None,
        season: str | None = None,
    ) -> list[GameResult]:
        return parse_sr_csv_text(text, sport=sport, season=season)
