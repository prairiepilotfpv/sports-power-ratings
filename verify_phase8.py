#!/usr/bin/env python
"""
Phase 8 Verification Script: Mean-Variance Separation

Demonstrates that:
1. Mean calibration does NOT modify SD
2. Variance calibration does NOT modify mean
3. Two-stage application preserves both properties
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from calibration.mean_calibrator import MeanCalibrator
from calibration.distribution import VarianceCalibrator

def test_mean_variance_separation():
    """Verify separation of mean and variance calibration."""
    print("\n" + "="*70)
    print("PHASE 8 VERIFICATION: Mean-Variance Calibration Separation")
    print("="*70)
    
    # Create synthetic data with known bias and miscalibrated variance
    np.random.seed(42)
    residuals = np.random.normal(0, 2.0, 100)
    pred_mean = np.array([100.0] * 100)
    pred_sd = np.array([3.0] * 100)  # Underestimated (true is ~2.0)
    actual_value = pred_mean + 5.0 + residuals  # +5 bias
    
    df = pd.DataFrame({
        "pred_mean": pred_mean,
        "pred_sd": pred_sd,
        "actual_value": actual_value,
    })
    
    print("\n[INPUT DATA]")
    print(f"  Samples: {len(df)}")
    print(f"  Pred mean: {df['pred_mean'].iloc[0]:.1f}")
    print(f"  Pred SD: {df['pred_sd'].iloc[0]:.1f}")
    print(f"  Actual bias: +{(df['actual_value'] - df['pred_mean']).mean():.1f}")
    print(f"  Actual std: {(df['actual_value'] - df['pred_mean']).std():.1f}")
    
    # ========== STAGE 1: MEAN CALIBRATION ==========
    print("\n" + "-"*70)
    print("[STAGE 1: MEAN CALIBRATION]")
    print("-"*70)
    
    mean_cal = MeanCalibrator()
    mean_cal.fit(df)
    
    print(f"\n  Fitted parameters:")
    print(f"    Delta (bias shift): {mean_cal.delta:+.4f}")
    print(f"    RMSE before: {mean_cal.rmse_before:.4f}")
    print(f"    RMSE after: {mean_cal.rmse_after:.4f}")
    print(f"    Improvement: {100*(mean_cal.rmse_before - mean_cal.rmse_after)/mean_cal.rmse_before:.1f}%")
    
    mean_result = mean_cal.transform(df[["pred_mean", "pred_sd"]])
    
    print(f"\n  Transformation results:")
    print(f"    Mean changed: {not np.allclose(mean_result['calibrated_mean'], df['pred_mean'])}")
    print(f"    SD changed: {not np.allclose(mean_result['calibrated_sd'], df['pred_sd'])}")
    print(f"    New mean: {mean_result['calibrated_mean'].iloc[0]:.4f}")
    print(f"    New SD (should be unchanged): {mean_result['calibrated_sd'].iloc[0]:.4f}")
    
    # ✓ VERIFICATION
    assert not np.allclose(mean_result['calibrated_mean'], df['pred_mean']), "Mean should change!"
    assert np.allclose(mean_result['calibrated_sd'], df['pred_sd'], atol=1e-6), "SD should NOT change!"
    print(f"\n  ✓ PASS: Mean changed, SD unchanged")
    
    # ========== STAGE 2: VARIANCE CALIBRATION ==========
    print("\n" + "-"*70)
    print("[STAGE 2: VARIANCE CALIBRATION]")
    print("-"*70)
    
    df_after_mean = pd.DataFrame({
        "pred_mean": mean_result["calibrated_mean"],
        "pred_sd": mean_result["calibrated_sd"],
        "actual_value": df["actual_value"],
    })
    
    var_cal = VarianceCalibrator()
    var_cal.fit(df_after_mean)
    
    print(f"\n  Fitted parameters:")
    print(f"    c (scale factor): {var_cal.c:.4f}")
    print(f"    tau (additional noise): {var_cal.tau:.4f}")
    print(f"    Clip rate: {var_cal.clip_rate:.3%}")
    print(f"    Healthy: {var_cal.healthy}")
    
    var_result = var_cal.transform(df_after_mean[["pred_mean", "pred_sd"]])
    
    print(f"\n  Transformation results:")
    print(f"    Mean changed: {not np.allclose(var_result['calibrated_mean'], df_after_mean['pred_mean'], atol=1e-6)}")
    print(f"    SD changed: {not np.allclose(var_result['calibrated_sd'], df_after_mean['pred_sd'])}")
    print(f"    Mean (should be unchanged): {var_result['calibrated_mean'].iloc[0]:.4f}")
    print(f"    SD before: {df_after_mean['pred_sd'].iloc[0]:.4f}")
    print(f"    SD after: {var_result['calibrated_sd'].iloc[0]:.4f}")
    
    # ✓ VERIFICATION
    assert np.allclose(var_result['calibrated_mean'], df_after_mean['pred_mean'], atol=1e-6), "Mean should NOT change!"
    assert not np.allclose(var_result['calibrated_sd'], df_after_mean['pred_sd']), "SD should change!"
    print(f"\n  ✓ PASS: Mean unchanged, SD changed")
    
    # ========== OVERALL VERIFICATION ==========
    print("\n" + "-"*70)
    print("[OVERALL VERIFICATION]")
    print("-"*70)
    
    print(f"\n  Original predictions:")
    print(f"    Mean: {df['pred_mean'].iloc[0]:.4f}")
    print(f"    SD: {df['pred_sd'].iloc[0]:.4f}")
    
    print(f"\n  After mean calibration:")
    print(f"    Mean: {mean_result['calibrated_mean'].iloc[0]:.4f} (Δ={mean_cal.delta:+.4f})")
    print(f"    SD: {mean_result['calibrated_sd'].iloc[0]:.4f} (unchanged)")
    
    print(f"\n  After variance calibration:")
    print(f"    Mean: {var_result['calibrated_mean'].iloc[0]:.4f} (preserved)")
    print(f"    SD: {var_result['calibrated_sd'].iloc[0]:.4f}")
    print(f"           (from {mean_result['calibrated_sd'].iloc[0]:.4f} via c={var_cal.c:.3f}, tau={var_cal.tau:.3f})")
    
    print(f"\n  ✓ VERIFIED: Each stage modifies only its responsibility")
    
    print("\n" + "="*70)
    print("Phase 8 Verification Complete: All separations confirmed ✓")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_mean_variance_separation()
