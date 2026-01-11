"""Helpers for reading persisted calibrator artifacts.

Calibrators are expected under outputs/calibrators/<sport>/<season>/<model>/*.joblib
with fallbacks to outputs/calibrators/<sport>/<model> and outputs/calibrators/<model>.
"""

from __future__ import annotations

from pathlib import Path

from .base import BaseCalibrator


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
    market: str = "ML",
) -> BaseCalibrator | None:
    """Load the most recent calibrator artifact for a model/season/market."""
    _ = market
    base_dir = Path("outputs") / "calibrators"
    search_dirs = [
        base_dir / sport / season / model,
        base_dir / sport / model,
        base_dir / model,
    ]
    for directory in search_dirs:
        latest = _latest_joblib(directory)
        if latest is None:
            continue
        try:
            return BaseCalibrator.load(latest)
        except Exception:
            return None
    return None
