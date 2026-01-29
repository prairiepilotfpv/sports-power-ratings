"""Tests for Phase 10: regime-conditioned TOTAL calibration (total_bucket)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from datetime import datetime

import pytest
import pandas as pd
import numpy as np

from calibration.total_bucket_regimes import (
    total_bucket_regime,
    label_dataframe_with_total_bucket,
    TotalBucketManifest,
    save_total_bucket_manifest,
    load_total_bucket_manifest,
    select_calibrator_by_bucket,
    compute_bucket_usage_stats,
)
from calibration.mean_calibrator import MeanCalibrator
from calibration.distribution import VarianceCalibrator


class TestTotalBucketRegimeLabelingImpl:
    """Test deterministic regime labeling."""
    
    def test_regime_labeling_low(self):
        """Test low bucket assignment."""
        # Default thresholds: (210, 225)
        assert total_bucket_regime(200.0) == "low"
        assert total_bucket_regime(209.9) == "low"
        assert total_bucket_regime(209.99999) == "low"
    
    def test_regime_labeling_mid(self):
        """Test mid bucket assignment."""
        # Between 210 and 225
        assert total_bucket_regime(210.0) == "mid"
        assert total_bucket_regime(215.0) == "mid"
        assert total_bucket_regime(224.999) == "mid"
    
    def test_regime_labeling_high(self):
        """Test high bucket assignment."""
        # >= 225
        assert total_bucket_regime(225.0) == "high"
        assert total_bucket_regime(230.0) == "high"
        assert total_bucket_regime(500.0) == "high"
    
    def test_regime_labeling_custom_thresholds(self):
        """Test custom threshold configuration."""
        thresholds = (100.0, 150.0)
        assert total_bucket_regime(99.0, thresholds=thresholds) == "low"
        assert total_bucket_regime(100.0, thresholds=thresholds) == "mid"
        assert total_bucket_regime(150.0, thresholds=thresholds) == "high"
    
    def test_regime_labeling_none_input(self):
        """Test handling of None input."""
        assert total_bucket_regime(None) is None
        assert total_bucket_regime(np.nan) is None
    
    def test_regime_labeling_invalid_input(self):
        """Test handling of invalid input."""
        assert total_bucket_regime("invalid") is None
        assert total_bucket_regime([1, 2, 3]) is None
    
    def test_regime_deterministic(self):
        """Test determinism: same input always produces same output."""
        for _ in range(5):
            assert total_bucket_regime(215.0) == "mid"
            assert total_bucket_regime(200.0) == "low"
            assert total_bucket_regime(230.0) == "high"
    
    def test_label_dataframe_with_total_bucket(self):
        """Test DataFrame labeling."""
        df = pd.DataFrame({
            "total_mean": [200.0, 215.0, 230.0, 205.0],
            "other": [1, 2, 3, 4],
        })
        
        labeled = label_dataframe_with_total_bucket(df)
        
        assert "total_bucket" in labeled.columns
        assert labeled["total_bucket"].tolist() == ["low", "mid", "high", "low"]
    
    def test_label_dataframe_missing_column(self):
        """Test DataFrame labeling when column is missing."""
        df = pd.DataFrame({"other": [1, 2, 3]})
        
        labeled = label_dataframe_with_total_bucket(df)
        
        assert "total_bucket" in labeled.columns
        assert labeled["total_bucket"].isna().all()


class TestTotalBucketManifest:
    """Test manifest data structure and persistence."""
    
    def test_manifest_creation_default(self):
        """Test creating manifest with defaults."""
        manifest = TotalBucketManifest(
            sport="nba",
            season="2025-26",
            source="historical",
        )
        
        assert manifest.sport == "nba"
        assert manifest.season == "2025-26"
        assert manifest.source == "historical"
        assert manifest.low_threshold == 210.0
        assert manifest.mid_threshold == 225.0
        assert manifest.min_samples_per_bucket == 200
    
    def test_manifest_creation_custom(self):
        """Test creating manifest with custom parameters."""
        manifest = TotalBucketManifest(
            sport="nfl",
            season="2024",
            source="ensemble_total_v1",
            low_threshold=40.0,
            mid_threshold=50.0,
            min_samples_per_bucket=100,
        )
        
        assert manifest.low_threshold == 40.0
        assert manifest.mid_threshold == 50.0
        assert manifest.min_samples_per_bucket == 100
    
    def test_manifest_to_dict(self):
        """Test manifest serialization."""
        manifest = TotalBucketManifest(
            sport="nba",
            season="2025-26",
            source="historical",
            samples_global=1000,
            samples_low=300,
            samples_mid=400,
            samples_high=300,
        )
        
        data = manifest.to_dict()
        
        assert isinstance(data, dict)
        assert data["sport"] == "nba"
        assert data["samples_global"] == 1000
        assert "created_at" in data
    
    def test_manifest_from_dict(self):
        """Test manifest deserialization."""
        data = {
            "sport": "nba",
            "season": "2025-26",
            "source": "historical",
            "low_threshold": 210.0,
            "mid_threshold": 225.0,
            "samples_global": 1000,
            "samples_low": 300,
            "samples_mid": 400,
            "samples_high": 300,
        }
        
        manifest = TotalBucketManifest.from_dict(data)
        
        assert manifest.sport == "nba"
        assert manifest.samples_global == 1000
    
    def test_manifest_roundtrip(self):
        """Test manifest serialize/deserialize roundtrip."""
        original = TotalBucketManifest(
            sport="nba",
            season="2025-26",
            source="historical",
            samples_global=1000,
            samples_low=300,
            samples_mid=400,
            samples_high=300,
            has_global_mean=True,
            has_global_variance=True,
            has_bucket_mean={"low": True, "mid": True, "high": False},
        )
        
        data = original.to_dict()
        restored = TotalBucketManifest.from_dict(data)
        
        assert restored.sport == original.sport
        assert restored.samples_low == original.samples_low
        assert restored.has_bucket_mean == original.has_bucket_mean


class TestManifestPersistence:
    """Test manifest file I/O."""
    
    def test_save_and_load_manifest(self):
        """Test saving and loading manifest from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            manifest = TotalBucketManifest(
                sport="nba",
                season="2025-26",
                source="historical",
                samples_global=1000,
                samples_low=300,
            )
            
            # Save
            saved_path = save_total_bucket_manifest(manifest, output_dir)
            assert saved_path.exists()
            
            # Load
            loaded = load_total_bucket_manifest(output_dir)
            assert loaded is not None
            assert loaded.sport == "nba"
            assert loaded.samples_low == 300
    
    def test_load_missing_manifest(self):
        """Test loading manifest from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loaded = load_total_bucket_manifest(Path(tmpdir))
            assert loaded is None


class TestBucketSelection:
    """Test bucket routing and calibrator selection."""
    
    def test_select_calibrator_by_bucket_low(self):
        """Test selecting low bucket calibrator."""
        manifest = TotalBucketManifest(
            sport="nba",
            season="2025-26",
            source="historical",
            low_threshold=210.0,
            mid_threshold=225.0,
        )
        
        mean_cal = MeanCalibrator()
        var_cal = VarianceCalibrator()
        calibrators = {
            "bucket_low_variance": var_cal,
            "global_variance": None,
        }
        
        source, calib = select_calibrator_by_bucket(200.0, manifest, calibrators)
        
        assert source == "low"
        assert calib is var_cal
    
    def test_select_calibrator_by_bucket_fallback_global(self):
        """Test fallback to global when bucket calibrator unavailable."""
        manifest = TotalBucketManifest(
            sport="nba",
            season="2025-26",
            source="historical",
            low_threshold=210.0,
            mid_threshold=225.0,
        )
        
        var_cal = VarianceCalibrator()
        calibrators = {
            "bucket_low_variance": None,  # Missing
            "global_variance": var_cal,
        }
        
        source, calib = select_calibrator_by_bucket(200.0, manifest, calibrators)
        
        assert source == "global"
        assert calib is var_cal
    
    def test_select_calibrator_by_bucket_no_bucket(self):
        """Test selection when total_mean doesn't map to a bucket."""
        manifest = TotalBucketManifest(
            sport="nba",
            season="2025-26",
            source="historical",
        )
        
        var_cal = VarianceCalibrator()
        calibrators = {
            "global_variance": var_cal,
        }
        
        source, calib = select_calibrator_by_bucket(None, manifest, calibrators)
        
        assert source == "global"
        assert calib is var_cal


class TestBucketUsageStats:
    """Test usage statistics computation."""
    
    def test_compute_usage_stats(self):
        """Test computing bucket usage statistics."""
        df = pd.DataFrame({
            "total_bucket": ["low", "mid", "high", "low"],
        })
        
        applied_by_bucket = {
            "bucket_low": 2,
            "bucket_mid": 1,
            "bucket_high": 1,
            "global": 0,
        }
        
        stats = compute_bucket_usage_stats(df, applied_by_bucket)
        
        assert stats["bucket_low"] == 2
        assert stats["bucket_mid"] == 1
        assert stats["bucket_high"] == 1
        assert stats["global"] == 0


class TestIntegrationScenarios:
    """Integration tests for regime-conditioned calibration."""
    
    def test_deterministic_labeling_idempotent(self):
        """Test that regime labeling is idempotent."""
        df = pd.DataFrame({
            "total_mean": [200.0, 215.0, 230.0] * 10,
        })
        
        labeled1 = label_dataframe_with_total_bucket(df)
        labeled2 = label_dataframe_with_total_bucket(df)
        
        pd.testing.assert_series_equal(
            labeled1["total_bucket"],
            labeled2["total_bucket"],
            check_names=True,
        )
    
    def test_manifest_with_multiple_bucket_files(self):
        """Test manifest with multiple bucket calibrator files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            manifest = TotalBucketManifest(
                sport="nba",
                season="2025-26",
                source="historical",
                calibrator_global_mean="outputs/calibrators/nba/2025-26/historical/global/calibrator.pkl",
                calibrator_global_variance="outputs/calibrators/nba/2025-26/historical/global/calibrator_variance.pkl",
                calibrators_bucket_mean={
                    "low": "outputs/calibrators/nba/2025-26/historical/bucket_low/calibrator.pkl",
                    "mid": "outputs/calibrators/nba/2025-26/historical/bucket_mid/calibrator.pkl",
                },
                calibrators_bucket_variance={
                    "low": "outputs/calibrators/nba/2025-26/historical/bucket_low/calibrator_variance.pkl",
                    "mid": "outputs/calibrators/nba/2025-26/historical/bucket_mid/calibrator_variance.pkl",
                },
            )
            
            saved_path = save_total_bucket_manifest(manifest, output_dir)
            loaded = load_total_bucket_manifest(output_dir)
            
            assert loaded is not None
            assert len(loaded.calibrators_bucket_mean) == 2
            assert loaded.calibrator_global_variance is not None
    
    def test_insufficient_samples_fallback(self):
        """Test that insufficient bucket samples result in no bucket calibrator."""
        # Simulate a manifest where low bucket has insufficient samples
        manifest = TotalBucketManifest(
            sport="nba",
            season="2025-26",
            source="historical",
            min_samples_per_bucket=200,
            samples_low=50,  # Below threshold
            samples_mid=300,
            samples_high=250,
            has_bucket_mean={"low": False, "mid": True, "high": True},
            has_bucket_variance={"low": False, "mid": True, "high": True},
        )
        
        # Low bucket should not be available
        assert not manifest.has_bucket_mean["low"]
        assert not manifest.has_bucket_variance["low"]
        
        # Mid and high should be available
        assert manifest.has_bucket_mean["mid"]
        assert manifest.has_bucket_variance["high"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
