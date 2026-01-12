"""Helpers for reading persisted calibrator artifacts.

Calibrators are expected under outputs/calibrators/<sport>/<season>/<model>/*.joblib
with fallbacks to outputs/calibrators/<sport>/<model> and outputs/calibrators/<model>.
"""

from __future__ import annotations

from pathlib import Path

from .base import BaseCalibrator
from markets.registry import get_market_spec


def _latest_joblib(directory: Path) -> Path | None:
    try:
        candidates = list(directory.glob("*.joblib"))
    except Exception:
        return None
    if not candidates:
        return None
    try:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    except Exception:
        return None


def load_latest_calibrator(
    *,
    sport: str,
    season: str,
    model: str,
    market: str | None = "ML",
) -> BaseCalibrator | None:
    """Load the most recent calibrator artifact for a model/season.

    This prefers market-specific calibrator directories produced by `MarketSpec.calibrator_dir` but
    falls back to legacy locations to remain backwards-compatible.
    """
    base_dir = Path("outputs") / "calibrators"
    # Prefer market-specific directory when present using MarketSpec; fall back to legacy paths.
    market_dir = None
    if market is not None:
        try:
            spec = get_market_spec(market)
            market_dir = spec.calibrator_dir(sport, season, model)
        except Exception:
            market_dir = None

    search_dirs = []
    if market_dir is not None:
        search_dirs.append(market_dir)
    # Legacy fallbacks
    search_dirs.extend(
        [
            base_dir / sport / season / model,
            base_dir / sport / model,
            base_dir / model,
        ]
    )
    for directory in search_dirs:
        latest = _latest_joblib(directory)
        if latest is None:
            continue
        try:
            return BaseCalibrator.load(latest)
        except Exception:
            return None
    return None
