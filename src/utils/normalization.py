"""Normalization helpers for incoming pipeline data."""

from __future__ import annotations

import re
from typing import Iterable

from markets.base import Market
from utils import identity as idu

_MARKET_TOKEN_CLEANER = re.compile(r"[^a-z0-9]+")


def normalize_market_type(value: str | None) -> Market | None:
    """Normalize market type labels into a Market enum."""
    if value is None:
        return None
    normalized = _MARKET_TOKEN_CLEANER.sub("", str(value).strip().lower())
    if normalized in {"ml", "moneyline", "moneylinebet", "moneylinewager", "moneylineodds"}:
        return Market.ML
    if normalized in {"spread", "spreads", "pointspread", "pointspreadline", "handicap", "line", "spreadline"}:
        return Market.SPREAD
    if normalized in {"total", "totals", "overunder", "overunderline", "overunderodds", "totalline", "ou", "overunderbet"}:
        return Market.TOTAL
    return None


def normalize_market_type_value(value: str | None) -> str | None:
    """Normalize market type labels into storage-friendly strings."""
    market = normalize_market_type(value)
    return market.value if market is not None else None


def normalize_evaluation_market_type(value: str | None) -> str | None:
    """Normalize market type labels into evaluator-friendly strings."""
    market = normalize_market_type(value)
    if market is None:
        return None
    if market == Market.ML:
        return "moneyline"
    return market.value


def normalize_total_selection(value: str | None) -> str | None:
    """Normalize total selections to Over/Under labels."""
    if not value:
        return None
    norm = str(value).strip().lower()
    if norm in {"over", "o", "ov"}:
        return "Over"
    if norm in {"under", "u", "un"}:
        return "Under"
    return None


def normalize_team_label(
    value: str | None,
    *,
    alias_map: dict[str, Iterable[str]] | None = None,
) -> str | None:
    """Normalize a team label using configured alias mappings."""
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if alias_map:
        resolved = idu.resolve_team_alias(raw, alias_map)
        if resolved:
            return resolved
    return raw
