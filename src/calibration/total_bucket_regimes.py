"""TOTAL bucket regime-conditioned calibration.

Phase 10 implementation for fitting and applying separate TOTAL calibrators
per total_bucket regime (low, mid, high based on predicted total_mean).

This module provides:
1. Deterministic regime labeling via total_bucket_regime()
2. Manifest schema and persistence
3. Regime-conditioned fitting helpers
4. Apply-time bucket routing
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

_LOG = logging.getLogger(__name__)


# ============================================================================
# REGIME LABELING
# ============================================================================


def total_bucket_regime(
    total_mean: float | None,
    *,
    thresholds: tuple[float, float] = (210, 225),
) -> str | None:
    """Deterministically assign a total_bucket regime label.
    
    Maps predicted total_mean to regime:
    - 'low':  total_mean < thresholds[0]
    - 'mid':  thresholds[0] <= total_mean < thresholds[1]
    - 'high': total_mean >= thresholds[1]
    
    Args:
        total_mean: Predicted total mean value
        thresholds: (low_cutoff, mid_cutoff) tuple (default: (210, 225))
    
    Returns:
        Regime label ('low', 'mid', 'high') or None if total_mean is None/NaN
    """
    if total_mean is None:
        return None
    try:
        mean_val = float(total_mean)
        if pd.isna(mean_val):
            return None
    except (ValueError, TypeError):
        return None
    
    low_cutoff, mid_cutoff = thresholds
    if mean_val < low_cutoff:
        return "low"
    elif mean_val < mid_cutoff:
        return "mid"
    else:
        return "high"


def label_dataframe_with_total_bucket(
    df: pd.DataFrame,
    total_col: str = "total_mean",
    thresholds: tuple[float, float] = (210, 225),
) -> pd.DataFrame:
    """Add a 'total_bucket' column to DataFrame.
    
    Args:
        df: Input DataFrame (must contain total_col)
        total_col: Column name for total mean values
        thresholds: (low_cutoff, mid_cutoff) tuple
    
    Returns:
        DataFrame with new 'total_bucket' column
    """
    df = df.copy()
    if total_col not in df.columns:
        _LOG.warning("[total_bucket] Column '%s' not found; creating empty regime column", total_col)
        df["total_bucket"] = None
    else:
        df["total_bucket"] = df[total_col].apply(
            lambda x: total_bucket_regime(x, thresholds=thresholds)
        )
    return df


# ============================================================================
# MANIFEST DATA STRUCTURE
# ============================================================================


@dataclass(frozen=True)
class RegimeSampleInfo:
    """Statistics for a single bucket."""
    bucket: str
    sample_count: int
    has_calibrator: bool
    created_at: str | None = None


@dataclass(frozen=True)
class TotalBucketManifest:
    """Manifest for regime-conditioned TOTAL calibrators.
    
    Tracks:
    - Regime configuration (low/mid/high thresholds)
    - Per-bucket sample counts and calibrator status
    - Fit window (start/end dates)
    - Global calibrator availability
    - Artifact filenames
    - Timestamp for provenance
    """
    
    sport: str
    season: str
    source: str
    market: str = "total"
    
    # Regime configuration
    low_threshold: float = 210.0
    mid_threshold: float = 225.0
    min_samples_per_bucket: int = 200
    
    # Sample info per bucket
    samples_global: int = 0
    samples_low: int = 0
    samples_mid: int = 0
    samples_high: int = 0
    
    # Calibrator availability flags
    has_global_mean: bool = False
    has_global_variance: bool = False
    has_bucket_mean: Dict[str, bool] = field(default_factory=lambda: {"low": False, "mid": False, "high": False})
    has_bucket_variance: Dict[str, bool] = field(default_factory=lambda: {"low": False, "mid": False, "high": False})
    
    # Fit window (ISO format dates or None)
    fit_start_date: str | None = None
    fit_end_date: str | None = None
    
    # Artifact files (relative paths)
    calibrator_global_mean: str | None = None
    calibrator_global_variance: str | None = None
    calibrators_bucket_mean: Dict[str, str] = field(default_factory=dict)  # bucket -> path
    calibrators_bucket_variance: Dict[str, str] = field(default_factory=dict)  # bucket -> path
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = "phase10"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TotalBucketManifest:
        """Load manifest from dictionary."""
        # Filter to recognized fields to allow forward compatibility
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in fields}
        return cls(**filtered)


# ============================================================================
# MANIFEST PERSISTENCE
# ============================================================================


def save_total_bucket_manifest(
    manifest: TotalBucketManifest,
    output_dir: Path | str,
) -> Path:
    """Save manifest to JSON.
    
    Args:
        manifest: Manifest to save
        output_dir: Directory to save to
    
    Returns:
        Path to saved manifest file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = output_dir / "regime_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2, default=str)
    
    _LOG.info(
        "[total_bucket] Saved manifest to %s",
        manifest_path,
    )
    return manifest_path


def load_total_bucket_manifest(
    output_dir: Path | str,
) -> TotalBucketManifest | None:
    """Load manifest from JSON.
    
    Args:
        output_dir: Directory containing regime_manifest.json
    
    Returns:
        Loaded manifest or None if not found
    """
    output_dir = Path(output_dir)
    manifest_path = output_dir / "regime_manifest.json"
    
    if not manifest_path.exists():
        _LOG.debug("[total_bucket] Manifest not found: %s", manifest_path)
        return None
    
    try:
        with open(manifest_path, "r") as f:
            data = json.load(f)
        manifest = TotalBucketManifest.from_dict(data)
        _LOG.info(
            "[total_bucket] Loaded manifest: %d/%d/%d samples (low/mid/high)",
            manifest.samples_low,
            manifest.samples_mid,
            manifest.samples_high,
        )
        return manifest
    except Exception as e:
        _LOG.warning("[total_bucket] Failed to load manifest: %s", e)
        return None


# ============================================================================
# BUCKET ROUTING
# ============================================================================


def select_calibrator_by_bucket(
    total_mean: float | None,
    manifest: TotalBucketManifest,
    calibrators: Dict[str, Any],
) -> tuple[str, Any | None]:
    """Select the appropriate calibrator for a row.
    
    Args:
        total_mean: Predicted total mean
        manifest: Manifest with bucket config and availability info
        calibrators: Dict mapping 'global_mean', 'global_variance', 'low_mean', 'low_variance', etc.
    
    Returns:
        Tuple of (source_label, calibrator_object)
        source_label is one of: 'bucket_low', 'bucket_mid', 'bucket_high', 'global', None
    """
    # Determine bucket
    thresholds = (manifest.low_threshold, manifest.mid_threshold)
    bucket = total_bucket_regime(total_mean, thresholds=thresholds)
    
    if bucket is None:
        return ("global", calibrators.get("global_variance"))
    
    # Try bucket-specific calibrator
    bucket_key = f"bucket_{bucket}_variance"
    if bucket_key in calibrators and calibrators[bucket_key] is not None:
        return (bucket, calibrators[bucket_key])
    
    # Fall back to global
    return ("global", calibrators.get("global_variance"))


def compute_bucket_usage_stats(
    df: pd.DataFrame,
    applied_by_bucket: Dict[str, int],
) -> Dict[str, int]:
    """Aggregate calibrator usage by bucket.
    
    Args:
        df: Schedule DataFrame (with total_bucket column if available)
        applied_by_bucket: Dict tracking count of rows by bucket
    
    Returns:
        Dict with per-bucket usage stats
    """
    stats = {"bucket_low": 0, "bucket_mid": 0, "bucket_high": 0, "global": 0}
    stats.update(applied_by_bucket)
    return stats
