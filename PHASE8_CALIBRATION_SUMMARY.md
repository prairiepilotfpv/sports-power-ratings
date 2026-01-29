Phase 8: Calibration Tightening & Transparency

COMPLETED OBJECTIVES:
======================

1. SPLIT CALIBRATION RESPONSIBILITIES ✓
   
   NEW: MeanCalibrator (src/calibration/mean_calibrator.py)
   - Shifts total_mean only (does not modify total_sd)
   - Fits additive bias: delta = mean(residuals) / (1 + lambda)
   - Stores pre/post RMSE and mean values in metadata
   - Validates all inputs (fails on NaN, SD <= 0)
   - Respects delta_min/max guardrails
   
   REFACTORED: VarianceCalibrator (src/calibration/distribution.py)
   - Explicitly documents "mean untouched" contract
   - Logs pre/post SD median and % change
   - Records c, tau, clip_rate in metadata
   - Guardrail clipping with transparency
   
   NEW: Two-Stage Pipeline (_apply_two_stage_total_calibration in schedule.py)
   - Stage 1: Mean calibration first
   - Stage 2: Variance calibration on output of Stage 1
   - Separate logging for each stage
   - Both stages optional (fail gracefully if calibrator missing)

2. VARIANCE SANITY CHECKS (TOTALS) ✓
   
   NEW: Variance Audit Module (src/calibration/variance_audit.py)
   - VarianceAuditResult: named tuple with metrics
   - compute_variance_audit(): Calculates:
     * empirical_mae vs predicted SD
     * coverage_1sd (fraction within ±1σ)
     * coverage_2sd (fraction within ±2σ)
     * outside_95ci (fraction outside 95% confidence interval)
   - log_variance_audit(): Logs concise INFO message:
     [VARIANCE AUDIT TOTAL] sd_before=..., sd_after=..., 
     coverage_1sd=...%, coverage_2sd=...%, outside_95ci=...%
   
   INTEGRATION in two-stage pipeline:
   - Captures SD before/after variance calibration
   - Computes guardrail clipping statistics
   - Logs c/tau calibration parameters

3. CALIBRATION DATA SOURCE VALIDATION ✓
   
   NEW: Validation Module (src/calibration/validation.py)
   - validate_calibration_data_source(): Checks market requirements
   - validate_no_nans_in_columns(): Strict NaN detection
   - check_outcome_range(): Bounds validation (min/max)
   - All validators fail loudly on invalid inputs
   
   INTEGRATION in MeanCalibrator:
   - Calls validate_no_nans_in_columns() at fit() entry
   - Validates SD > 0 where present
   - Rejects DataFrame if < 2 samples

4. RECENCY & WINDOW CLARITY ✓
   
   NEW: Window Logging Helper (_log_calibration_window in historical_calibration.py)
   - Parameters: market, window_type, start_date, end_date
   - Extracts date range from games DataFrame
   - Logs at INFO level:
     [calibration window] market=TOTAL mode=expanding 
     start=2025-01-01 end=2025-01-28 games=42
   
   INTEGRATION in calibrate_sport_season():
   - Called once per market before fit_calibrator_for_market()
   - Provides transparency on data used for each calibrator
   - No silent defaults

5. TESTS ✓
   
   NEW: test_phase8_calibration.py (19 comprehensive tests)
   - TestMeanCalibrator (5 tests):
     * Shifts mean, preserves SD
     * Stores metadata correctly
     * Rejects NaNs and negative SD
     * Respects delta bounds
   - TestVarianceCalibrator (3 tests):
     * Preserves mean
     * Records c, tau, clip_rate
     * Rejects SD <= 0
   - TestCalibrationSeparation (1 test):
     * Mean then variance in sequence preserves each responsibility
   - TestVarianceAudit (2 tests):
     * Metrics computed correctly
     * High coverage with small residuals
   - TestValidation (5 tests):
     * NaN detection
     * Required columns check
     * Outcome range bounds
   - TestMeanCalibratorTransform (3 tests):
     * Transform API validation
     * Missing column handling
   
   ALL TESTS PASS: 19/19

6. PHASE 7 REGRESSION VERIFICATION ✓
   
   Ran existing test suites:
   - test_phase4_heads_contract.py: 44 tests PASS
   - test_phase6_ensemble_governance.py: 25 tests PASS
   - test_calibration_bets_integration.py: 19 tests PASS
   
   Zero regression in heads, ensemble, or betting behavior


ARCHITECTURE NOTES:
===================

Mean vs Variance Separation:
- MeanCalibrator.fit() expects: pred_mean, actual_value, (optional pred_sd)
- Output: calibrated_mean = pred_mean + delta, calibrated_sd = pred_sd (unchanged)

- VarianceCalibrator.fit() expects: pred_mean, pred_sd, actual_value
- Output: calibrated_mean = pred_mean (unchanged), calibrated_sd = sqrt((c*sd)^2 + tau^2)

Why Order Matters:
1. Apply mean calibration first to fix bias
2. Then apply variance calibration to remaining residuals
3. Inverse order would inflate/deflate SD estimates

Schedule Pipeline Integration:
- _apply_two_stage_total_calibration() handles both stages
- Returns (modified_df, bool: calibration_applied)
- Appends "calibrated_total" to win_prob_source if either stage succeeds
- Logs individual stage results for diagnostics


LOGGING OUTPUT EXAMPLES:
========================

Mean Calibration:
  [TOTAL calibration] MEAN stage: delta=+4.5000, improvement=+12.3%, samples=234

Variance Calibration:
  [TOTAL calibration] VARIANCE stage: sd_before=7.452, sd_after=8.123, 
  change=+9.0%, c=1.080, tau=0.245, guardrail_clipped=2.1% (5/234), samples=234

Window Clarity:
  [calibration window] market=TOTAL mode=expanding start=2025-01-01 
  end=2025-01-28 actual_start=2025-01-02 actual_end=2025-01-27 games=42

Overall:
  [TOTAL calibration] Completed stages: mean+variance


DELIVERABLES CHECKLIST:
=======================

✓ Explicit mean + variance calibrators for TOTAL
✓ Variance audit logging (INFO level)
✓ Clear calibration window reporting
✓ Tests locking separation of responsibilities
✓ No regression in Phase 7 contracts
✓ Data source validation with NaN rejection
✓ Separate pre/post statistics logging
✓ Guardrail transparency and clipping metrics

FOLLOW-UP WORK (not in Phase 8 scope):
======================================

1. Production Deployment:
   - Register MeanCalibrator in calibration pipeline
   - Update historical_calibration.py to fit/persist both stages
   - Update CLI with --mean-only / --variance-only flags

2. Advanced Diagnostics:
   - Forecast vs realized variance tracking
   - Per-market calibration quality metrics
   - Ensemble weight audit relative to calibration status

3. Adaptive Calibration:
   - Rolling window instead of expanding (currently hardcoded)
   - Lookback period configuration per sport/market
   - Out-of-sample validation for each stage

