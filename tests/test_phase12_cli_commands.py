"""Test Phase 12 CLI command handlers."""

import json
import tempfile
from pathlib import Path

import pytest

from src.calibration.active_policy import create_global_policy, save_total_active_policy


class TestCLICommands:
    """Test that CLI command handlers are callable."""

    def test_calibration_policy_command_handler_exists(self):
        """calibration-policy command handler should be defined."""
        from src.cli.pipeline import _run_calibration_policy
        
        assert callable(_run_calibration_policy)

    def test_calibration_promote_total_command_handler_exists(self):
        """calibration-promote-total command handler should be defined."""
        from src.cli.pipeline import _run_calibration_promote_total
        
        assert callable(_run_calibration_promote_total)

    def test_calibration_rollback_total_command_handler_exists(self):
        """calibration-rollback-total command handler should be defined."""
        from src.cli.pipeline import _run_calibration_rollback_total
        
        assert callable(_run_calibration_rollback_total)

    def test_calibration_policy_help_output_mock(self, capsys, monkeypatch):
        """calibration-policy should display policy info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                import src.calibration.active_policy as policy_module
                original = policy_module._DATA_ROOT
                policy_module._DATA_ROOT = Path(tmpdir)

                # Create a mock args object
                class Args:
                    sport = "nba"
                    season = "2025-26"

                from src.cli.pipeline import _run_calibration_policy

                _run_calibration_policy(Args())
                
                captured = capsys.readouterr()
                assert "calibration policy" in captured.out or "no active policy" in captured.out
            finally:
                policy_module._DATA_ROOT = original

    def test_calibration_promote_total_global_mode_mock(self, capsys, monkeypatch):
        """calibration-promote-total with global mode should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                import src.calibration.active_policy as policy_module
                original = policy_module._DATA_ROOT
                policy_module._DATA_ROOT = Path(tmpdir)

                # Create a mock args object
                class Args:
                    sport = "nba"
                    season = "2025-26"
                    mode = "global"
                    global_path = "/path/to/global"
                    manifest_path = None
                    notes = "test promotion"

                from src.cli.pipeline import _run_calibration_promote_total

                _run_calibration_promote_total(Args())
                
                captured = capsys.readouterr()
                assert "promoted" in captured.out
                assert "market=TOTAL mode=global" in captured.out
            finally:
                policy_module._DATA_ROOT = original

    def test_calibration_promote_total_bucket_mode_mock(self, capsys, monkeypatch):
        """calibration-promote-total with bucket mode should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                import src.calibration.active_policy as policy_module
                original = policy_module._DATA_ROOT
                policy_module._DATA_ROOT = Path(tmpdir)

                # Create a mock args object
                class Args:
                    sport = "nba"
                    season = "2025-26"
                    mode = "total_bucket"
                    global_path = None
                    manifest_path = "/path/to/manifest.json"
                    notes = ""

                from src.cli.pipeline import _run_calibration_promote_total

                _run_calibration_promote_total(Args())
                
                captured = capsys.readouterr()
                assert "promoted" in captured.out
            finally:
                policy_module._DATA_ROOT = original

    def test_calibration_rollback_total_with_policy_mock(self, capsys, monkeypatch):
        """calibration-rollback-total should rollback when policy exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                import src.calibration.active_policy as policy_module
                original = policy_module._DATA_ROOT
                policy_module._DATA_ROOT = Path(tmpdir)

                # Create and save a policy first
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

                # Now rollback
                class Args:
                    sport = "nba"
                    season = "2025-26"

                from src.cli.pipeline import _run_calibration_rollback_total

                _run_calibration_rollback_total(Args())
                
                captured = capsys.readouterr()
                assert "rollback" in captured.out
            finally:
                policy_module._DATA_ROOT = original

    def test_calibration_rollback_total_no_policy_mock(self, capsys, monkeypatch):
        """calibration-rollback-total should handle missing policy gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                import src.calibration.active_policy as policy_module
                original = policy_module._DATA_ROOT
                policy_module._DATA_ROOT = Path(tmpdir)

                # No policy exists
                class Args:
                    sport = "nba"
                    season = "2025-26"

                from src.cli.pipeline import _run_calibration_rollback_total

                _run_calibration_rollback_total(Args())
                
                captured = capsys.readouterr()
                assert "no_active_policy" in captured.out
            finally:
                policy_module._DATA_ROOT = original
