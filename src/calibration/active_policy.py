"""Active TOTAL calibration policy management.

Maintains a single source of truth for which TOTAL calibration mode is active:
- global: Use global TOTAL mean + variance calibrators
- total_bucket: Use regime-conditioned bucket calibrators with optional global fallback

Schema:
    {
        "market": "total",
        "mode": "global" | "total_bucket",
        "global": {"path": "<path to global calibrator artifact or directory>"},
        "total_bucket": {"manifest_path": "<path to bucket manifest json>"},
        "promoted_at": "<ISO UTC>",
        "notes": "optional"
    }
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_LOG = logging.getLogger(__name__)

# Data root for policy files
_DATA_ROOT = Path("data")


def get_total_policy_path(sport: str, season: str) -> Path:
    """Return path to active TOTAL policy file.
    
    Args:
        sport: Sport code (e.g., 'nba')
        season: Season identifier (e.g., '2025-26')
        
    Returns:
        Path to active.json (may not exist)
    """
    policy_dir = _DATA_ROOT / "calibrators" / sport / season / "historical" / "total"
    return policy_dir / "active.json"


def load_total_active_policy(sport: str, season: str) -> Optional[dict[str, Any]]:
    """Load the active TOTAL calibration policy if it exists.
    
    Args:
        sport: Sport code
        season: Season identifier
        
    Returns:
        Policy dict if file exists and is valid, None otherwise
        
    Raises:
        ValueError: If policy file exists but is invalid JSON or fails validation
    """
    policy_path = get_total_policy_path(sport, season)
    
    if not policy_path.exists():
        return None
    
    try:
        with open(policy_path, "r") as f:
            policy = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in active policy {policy_path}: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load active policy from {policy_path}: {e}")
    
    # Validate before returning
    validate_total_policy(policy)
    return policy


def save_total_active_policy(
    sport: str, 
    season: str, 
    policy: dict[str, Any],
) -> Path:
    """Save the active TOTAL calibration policy.
    
    Creates parent directories as needed.
    Validates policy before writing.
    
    Args:
        sport: Sport code
        season: Season identifier
        policy: Policy dict with required fields
        
    Returns:
        Path where policy was written
        
    Raises:
        ValueError: If policy fails validation
        OSError: If unable to write file
    """
    validate_total_policy(policy)
    
    policy_path = get_total_policy_path(sport, season)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(policy_path, "w") as f:
        json.dump(policy, f, indent=2)
    
    return policy_path


def validate_total_policy(policy: dict[str, Any]) -> None:
    """Validate TOTAL policy structure and required fields.
    
    Rules:
    - market must be "total"
    - mode must be "global" or "total_bucket"
    - If mode=global: global.path is required
    - If mode=total_bucket: total_bucket.manifest_path is required
    - promoted_at must be ISO UTC format (optional in new policies, validated if present)
    
    Args:
        policy: Policy dict to validate
        
    Raises:
        ValueError: If policy is invalid with clear error message
    """
    if not isinstance(policy, dict):
        raise ValueError("Policy must be a dict")
    
    market = policy.get("market")
    if market != "total":
        raise ValueError(f"policy.market must be 'total', got '{market}'")
    
    mode = policy.get("mode")
    if mode not in ("global", "total_bucket"):
        raise ValueError(f"policy.mode must be 'global' or 'total_bucket', got '{mode}'")
    
    if mode == "global":
        global_cfg = policy.get("global")
        if not global_cfg or not isinstance(global_cfg, dict):
            raise ValueError("policy.global must be a dict when mode='global'")
        if not global_cfg.get("path"):
            raise ValueError("policy.global.path is required when mode='global'")
    
    elif mode == "total_bucket":
        bucket_cfg = policy.get("total_bucket")
        if not bucket_cfg or not isinstance(bucket_cfg, dict):
            raise ValueError("policy.total_bucket must be a dict when mode='total_bucket'")
        if not bucket_cfg.get("manifest_path"):
            raise ValueError("policy.total_bucket.manifest_path is required when mode='total_bucket'")
    
    # If promoted_at is present, validate ISO format
    if "promoted_at" in policy:
        promoted_at = policy.get("promoted_at")
        if promoted_at:
            try:
                datetime.fromisoformat(promoted_at)
            except (ValueError, TypeError):
                raise ValueError(
                    f"policy.promoted_at must be ISO UTC format, got '{promoted_at}'"
                )


def create_global_policy(
    sport: str,
    season: str,
    global_path: str | Path,
    notes: str = "",
) -> dict[str, Any]:
    """Create a global TOTAL calibration policy.
    
    Args:
        sport: Sport code
        season: Season identifier
        global_path: Path to global calibrator artifact or directory
        notes: Optional notes
        
    Returns:
        Policy dict ready to save
    """
    policy = {
        "market": "total",
        "mode": "global",
        "global": {
            "path": str(global_path),
        },
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }
    if notes:
        policy["notes"] = notes
    
    validate_total_policy(policy)
    return policy


def create_total_bucket_policy(
    sport: str,
    season: str,
    manifest_path: str | Path,
    global_path: Optional[str | Path] = None,
    notes: str = "",
) -> dict[str, Any]:
    """Create a total_bucket TOTAL calibration policy.
    
    Args:
        sport: Sport code
        season: Season identifier
        manifest_path: Path to bucket manifest JSON
        global_path: Optional path to global calibrator for fallback
        notes: Optional notes
        
    Returns:
        Policy dict ready to save
    """
    policy = {
        "market": "total",
        "mode": "total_bucket",
        "total_bucket": {
            "manifest_path": str(manifest_path),
        },
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }
    
    # Include global path if provided (for fallback)
    if global_path:
        policy["global"] = {
            "path": str(global_path),
        }
    
    if notes:
        policy["notes"] = notes
    
    validate_total_policy(policy)
    return policy
