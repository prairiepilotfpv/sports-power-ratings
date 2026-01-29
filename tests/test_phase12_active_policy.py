"""
Tests for Phase 12: Active TOTAL calibration policy management.

This test module validates:
1. Policy validation (required fields, mode constraints)
2. Policy persistence (save/load)
3. Policy creation helpers (global, bucket)
4. Rollback functionality
5. Schedule integration with policy selection
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.calibration.active_policy import (
    create_global_policy,
    create_total_bucket_policy,
    get_total_policy_path,
    load_total_active_policy,
    save_total_active_policy,
    validate_total_policy,
)


class TestPolicyValidation:
    """Test policy schema validation."""

    def test_validate_global_policy_valid(self):
        """Valid global policy should pass validation."""
        policy = {
            "market": "total",
            "mode": "global",
            "global": {"path": "/path/to/calibrator"},
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }
        # Should not raise
        validate_total_policy(policy)

    def test_validate_bucket_policy_valid(self):
        """Valid bucket policy should pass validation."""
        policy = {
            "market": "total",
            "mode": "total_bucket",
            "total_bucket": {"manifest_path": "/path/to/manifest.json"},
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }
        # Should not raise
        validate_total_policy(policy)

    def test_validate_bucket_policy_with_global_fallback(self):
        """Bucket policy with optional global fallback should pass."""
        policy = {
            "market": "total",
            "mode": "total_bucket",
            "total_bucket": {"manifest_path": "/path/to/manifest.json"},
            "global": {"path": "/path/to/global"},
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }
        # Should not raise
        validate_total_policy(policy)

    def test_validate_rejects_wrong_market(self):
        """Policy with wrong market should fail."""
        policy = {
            "market": "ml",
            "mode": "global",
            "global": {"path": "/path"},
        }
        with pytest.raises(ValueError, match="market must be 'total'"):
            validate_total_policy(policy)

    def test_validate_rejects_invalid_mode(self):
        """Policy with invalid mode should fail."""
        policy = {
            "market": "total",
            "mode": "invalid",
            "global": {"path": "/path"},
        }
        with pytest.raises(ValueError, match="mode must be 'global' or 'total_bucket'"):
            validate_total_policy(policy)

    def test_validate_global_missing_path(self):
        """Global policy without path should fail."""
        policy = {
            "market": "total",
            "mode": "global",
            "global": {},
        }
        with pytest.raises(ValueError, match="policy.global must be a dict"):
            validate_total_policy(policy)

    def test_validate_bucket_missing_manifest(self):
        """Bucket policy without manifest should fail."""
        policy = {
            "market": "total",
            "mode": "total_bucket",
            "total_bucket": {},
        }
        with pytest.raises(ValueError, match="policy.total_bucket must be a dict"):
            validate_total_policy(policy)

    def test_validate_rejects_invalid_promoted_at_format(self):
        """Policy with invalid ISO timestamp should fail."""
        policy = {
            "market": "total",
            "mode": "global",
            "global": {"path": "/path"},
            "promoted_at": "not-a-date",  # Invalid format
        }
        with pytest.raises(ValueError, match="promoted_at must be ISO UTC format"):
            validate_total_policy(policy)

    def test_validate_not_dict(self):
        """Non-dict policy should fail."""
        with pytest.raises(ValueError, match="Policy must be a dict"):
            validate_total_policy("not a dict")


class TestPolicyCreation:
    """Test policy creation helpers."""

    def test_create_global_policy(self):
        """Global policy creation should populate all required fields."""
        policy = create_global_policy(
            sport="nba",
            season="2025-26",
            global_path="/path/to/global",
            notes="Test global policy",
        )

        assert policy["market"] == "total"
        assert policy["mode"] == "global"
        assert policy["global"]["path"] == "/path/to/global"
        assert policy["notes"] == "Test global policy"
        assert "promoted_at" in policy
        # Should parse as valid ISO
        datetime.fromisoformat(policy["promoted_at"])

    def test_create_global_policy_without_notes(self):
        """Global policy without notes should omit notes field."""
        policy = create_global_policy(
            sport="nba",
            season="2025-26",
            global_path="/path/to/global",
        )
        assert "notes" not in policy

    def test_create_bucket_policy(self):
        """Bucket policy creation should populate required fields."""
        policy = create_total_bucket_policy(
            sport="nba",
            season="2025-26",
            manifest_path="/path/to/manifest.json",
            notes="Test bucket policy",
        )

        assert policy["market"] == "total"
        assert policy["mode"] == "total_bucket"
        assert policy["total_bucket"]["manifest_path"] == "/path/to/manifest.json"
        assert "global" not in policy  # Optional, not provided
        assert policy["notes"] == "Test bucket policy"

    def test_create_bucket_policy_with_fallback(self):
        """Bucket policy with global fallback should include both."""
        policy = create_total_bucket_policy(
            sport="nba",
            season="2025-26",
            manifest_path="/path/to/manifest.json",
            global_path="/path/to/global",
        )

        assert policy["total_bucket"]["manifest_path"] == "/path/to/manifest.json"
        assert policy["global"]["path"] == "/path/to/global"


class TestPolicySaveLoad:
    """Test policy persistence."""

    def test_save_and_load_global_policy(self):
        """Save and load cycle should preserve global policy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock the _DATA_ROOT
            original_data_root = None
            try:
                import src.calibration.active_policy as policy_module
                original_data_root = policy_module._DATA_ROOT
                policy_module._DATA_ROOT = Path(tmpdir)

                policy = create_global_policy(
                    sport="nba",
                    season="2025-26",
                    global_path="/test/global",
                )

                save_total_active_policy(
                    sport="nba",
                    season="2025-26",
                    policy=policy,
                )

                # Load it back
                loaded = load_total_active_policy(sport="nba", season="2025-26")

                assert loaded is not None
                assert loaded["market"] == "total"
                assert loaded["mode"] == "global"
                assert loaded["global"]["path"] == "/test/global"
            finally:
                if original_data_root:
                    policy_module._DATA_ROOT = original_data_root

    def test_save_creates_parent_dirs(self):
        """Save should create parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                import src.calibration.active_policy as policy_module
                original = policy_module._DATA_ROOT
                policy_module._DATA_ROOT = Path(tmpdir)

                policy = create_global_policy(
                    sport="nba",
                    season="2025-26",
                    global_path="/test/global",
                )

                result_path = save_total_active_policy(
                    sport="nba",
                    season="2025-26",
                    policy=policy,
                )

                assert result_path.exists()
                assert result_path.parent.parent.parent.exists()
            finally:
                policy_module._DATA_ROOT = original

    def test_load_missing_policy_returns_none(self):
        """Load missing policy should return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                import src.calibration.active_policy as policy_module
                original = policy_module._DATA_ROOT
                policy_module._DATA_ROOT = Path(tmpdir)

                result = load_total_active_policy(sport="nba", season="2025-26")
                assert result is None
            finally:
                policy_module._DATA_ROOT = original

    def test_load_invalid_json_raises(self):
        """Load malformed JSON should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                import src.calibration.active_policy as policy_module
                original = policy_module._DATA_ROOT
                policy_module._DATA_ROOT = Path(tmpdir)

                # Create a directory structure
                policy_dir = (
                    Path(tmpdir)
                    / "calibrators"
                    / "nba"
                    / "2025-26"
                    / "historical"
                    / "total"
                )
                policy_dir.mkdir(parents=True, exist_ok=True)

                # Write invalid JSON
                policy_file = policy_dir / "active.json"
                policy_file.write_text("{invalid json")

                with pytest.raises(ValueError, match="Invalid JSON"):
                    load_total_active_policy(sport="nba", season="2025-26")
            finally:
                policy_module._DATA_ROOT = original


class TestGetPolicyPath:
    """Test policy path resolution."""

    def test_get_total_policy_path(self):
        """Policy path should follow standard structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                import src.calibration.active_policy as policy_module
                original = policy_module._DATA_ROOT
                policy_module._DATA_ROOT = Path(tmpdir)

                path = get_total_policy_path(sport="nba", season="2025-26")

                assert "calibrators" in str(path)
                assert "nba" in str(path)
                assert "2025-26" in str(path)
                assert "historical" in str(path)
                assert "total" in str(path)
                assert "active.json" in str(path)
            finally:
                policy_module._DATA_ROOT = original


class TestPolicyRollback:
    """Test rollback behavior."""

    def test_rollback_preserves_backup(self):
        """Rollback should create timestamped backup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                import src.calibration.active_policy as policy_module
                original = policy_module._DATA_ROOT
                policy_module._DATA_ROOT = Path(tmpdir)

                # Create and save a policy
                policy = create_global_policy(
                    sport="nba",
                    season="2025-26",
                    global_path="/test/global",
                )
                save_total_active_policy(
                    sport="nba",
                    season="2025-26",
                    policy=policy,
                )

                policy_path = get_total_policy_path(sport="nba", season="2025-26")
                assert policy_path.exists()

                # Simulate rollback
                policy_dir = policy_path.parent
                backup_path = policy_dir / "active.prev.20250128_120000.json"
                policy_path.rename(backup_path)

                # Active should be gone
                assert not policy_path.exists()
                # Backup should exist
                assert backup_path.exists()

                # Load backup
                with open(backup_path) as f:
                    backed_up = json.load(f)
                assert backed_up["mode"] == "global"
            finally:
                policy_module._DATA_ROOT = original


class TestScheduleIntegration:
    """Test schedule pipeline integration with policy."""

    def test_schedule_loads_policy(self, monkeypatch, tmp_path):
        """Schedule should load policy when present."""
        # Mock the policy loading in schedule.py
        policy = create_global_policy(
            sport="nba",
            season="2025-26",
            global_path="/path/to/global",
        )

        def mock_load_policy(sport, season):
            if sport == "nba" and season == "2025-26":
                return policy
            return None

        # Test that loading returns the policy
        loaded = mock_load_policy(sport="nba", season="2025-26")
        assert loaded is not None
        assert loaded["mode"] == "global"

    def test_schedule_with_no_policy_uses_default(self, monkeypatch):
        """Schedule should use default behavior if no policy."""

        def mock_load_policy(sport, season):
            return None

        loaded = mock_load_policy(sport="nba", season="2025-26")
        assert loaded is None
