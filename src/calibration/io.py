"""Helpers for reading and writing persisted calibrator artifacts.

Calibrators are expected under outputs/calibrators/<sport>/<season>/<source_id>/<market>/
with calibrator.pkl (joblib serialized) and metadata.json files.
"""

from __future__ import annotations

from pathlib import Path
import joblib
import json
import logging
from datetime import datetime

_LOG = logging.getLogger(__name__)


def save_calibrator(
    calibrator,
    output_dir: Path | str,
    market: str | None = None,
    metadata: dict | None = None,
) -> Path:
    """Save a calibrator to disk.
    
    Args:
        calibrator: The calibrator object to save
        output_dir: Directory to save to
        market: Optional market name for metadata
        metadata: Optional metadata dict (added to calibrator attributes)
    
    Returns:
        Path to saved calibrator file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save calibrator object
    calibrator_path = output_dir / "calibrator.pkl"
    joblib.dump(calibrator, str(calibrator_path))
    _LOG.info(f"[save_calibrator] Saved calibrator to {calibrator_path}")
    
    # Save metadata
    meta_dict = {
        "saved_at": datetime.now().isoformat(),
        "market": market or "unknown",
        "method": getattr(calibrator, "method", "unknown"),
    }
    if metadata:
        meta_dict.update(metadata)
    
    # Also store as attribute on calibrator
    calibrator.metadata = meta_dict
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(meta_dict, f, indent=2, default=str)
    _LOG.info(f"[save_calibrator] Saved metadata to {metadata_path}")
    
    return calibrator_path


def load_latest_calibrator(
    *,
    sport: str,
    season: str,
    model: str,
    market: str = "ML",
    source_id: str = "historical",
) -> any:
    """Load the most recent calibrator for a market.
    
    Args:
        sport: Sport code (e.g., 'nba', 'nfl')
        season: Season identifier (e.g., '2025-26')
        model: Model name (e.g., 'bradley-terry')
        market: Market type ("ML", "spread", "total")
        source_id: Source identifier (default: "historical")
    
    Returns:
        Loaded calibrator object, or None if not found
    """
    # Construct path: outputs/calibrators/{sport}/{season}/{source_id}/{market}/
    calib_dir = Path("outputs") / "calibrators" / sport / season / source_id / market
    
    if not calib_dir.exists():
        _LOG.debug(f"[load_latest_calibrator] Calibrator directory not found: {calib_dir}")
        return None
    
    # Find latest calibrator.pkl
    pkl_files = sorted(calib_dir.glob("calibrator.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pkl_files:
        _LOG.debug(f"[load_latest_calibrator] No calibrator files in {calib_dir}")
        return None
    
    latest_pkl = pkl_files[0]
    try:
        calibrator = joblib.load(str(latest_pkl))
        
        # Load metadata if available
        metadata_path = calib_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                calibrator.metadata = metadata
        else:
            calibrator.metadata = {"market": market, "source_id": source_id}
        
        _LOG.info(f"[load_latest_calibrator] Loaded {market} calibrator from {latest_pkl}")
        return calibrator
    except Exception as e:
        _LOG.warning(f"[load_latest_calibrator] Failed to load {latest_pkl}: {e}")
        return None
