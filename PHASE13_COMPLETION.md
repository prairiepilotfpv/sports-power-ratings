Phase 13: Promote TOTAL Calibration Policy from A/B Report + Guarded Rollback
==============================================================================

COMPLETION SUMMARY
==================

Phase 13 adds a convenience layer for promoting TOTAL calibration policies from Phase 11 A/B reports
and monitoring policy health with advisory warnings (no automatic rollback).

DEFINITION OF DONE: ✅ COMPLETE

1. ✅ Promotion from AB report works
   - Load Phase 11 calibration_ab_report.json
   - Validate policy recommendation fields
   - Promote to active.json using Phase 12 helpers
   - Block promotion if recommendation != "recommended" with clear error

2. ✅ Health check reports degradation without side effects
   - Load active TOTAL policy
   - Evaluate recent games using Phase 9 metrics
   - Compare against thresholds
   - Return advisory summary (no files modified, no automatic rollback)

3. ✅ Existing Phase 11 and Phase 12 tests remain unchanged
   - All 21 Phase 12 tests pass
   - No breaking changes to existing modules


IMPLEMENTATION DETAILS
======================

1. PHASE 11 UPDATE: calibration_ab.py
   ===================================
   
   Updated CalibrationABReport dataclass to persist calibrator paths:
   - Added: global_calibrator_path (str | None)
   - Added: bucket_manifest_path (str | None)
   
   These fields are populated during A/B evaluation:
   - global_calibrator_path: Path to fitted global variance calibrator
   - bucket_manifest_path: Path to total_bucket regime manifest JSON
   
   Benefits: Eliminates need to re-specify paths when promoting from report


2. NEW MODULE: src/pipelines/calibration_promote.py
   ==================================================
   
   Promotion Functions:
   
   load_calibration_ab_report(report_path)
     - Load and parse Phase 11 A/B report JSON
     - Raises: FileNotFoundError, ValueError (invalid JSON)
   
   validate_ab_report_for_promotion(report)
     - Validate report has required fields
     - Checks: policy.recommendation, calibrator paths
     - Raises: ValueError with specific field names
   
   promote_policy_from_report(sport, season, report, notes="")
     - Main promotion function
     - Logic:
       1. Validate report structure
       2. Check recommendation field
       3. If "recommended":
          - Determine mode: bucket (if manifest_path) else global
          - Extract paths from report
          - Create policy using Phase 12 helpers
          - Save to active.json
       4. If "not_recommended":
          - Abort with reasoning from report
     - Returns: Path to saved active.json
   
   Health Check Functions:
   
   evaluate_policy_health(db_path, sport, season, market="total", window_games=100)
     - Advisory evaluation of active policy performance
     - Loads recent games, applies policy, computes metrics
     - Compares against thresholds (MAE, RMSE, coverage, tail miss)
     - Returns: {"status": "OK"|"WARN", "current_metrics": {...}, "warnings": [...]}
     - Does NOT modify any files or auto-rollback
   
   _apply_policy_to_dataset(eval_df, policy, db_path)
     - Apply active policy calibrators to a dataset
     - Handles both global and total_bucket modes
     - Returns: Calibrated copy of input (deep copy, no mutation)


3. CLI INTEGRATION: src/cli/pipeline.py
   ====================================
   
   Two new subcommands added:
   
   a) calibration-promote-total-from-report
      Args:
        --sport (required)
        --season (required)
        --report (required): Path to calibration_ab_report.json
        --notes (optional): Notes to include in policy
      
      Behavior: Load report, validate, promote if recommended, error if not
      Output: ASCII-only logs with status and policy path
   
   b) calibration-policy-health
      Args:
        --sport (required)
        --season (required)
        --market (default: total)
        --window-games (default: 100): Recent games to evaluate
        --db (optional): DB path override
      
      Behavior: Evaluate health of active policy, report warnings
      Output: ASCII-only metrics and warnings (advisory only)
   
   Command handlers:
   - _run_calibration_promote_total_from_report()
   - _run_calibration_policy_health()


TESTING
=======

Created: tests/test_phase13_promotion_health.py (24 comprehensive tests)

Test Classes:

1. TestReportLoading (3 tests)
   - Load valid A/B report from JSON
   - Error on missing file
   - Error on invalid JSON

2. TestReportValidation (5 tests)
   - Valid bucket report passes
   - Valid global report passes
   - Error on missing policy field
   - Error on invalid recommendation
   - Error when no calibrator paths

3. TestPromotionFromReport (4 tests)
   - Promote bucket policy succeeds
   - Promote global policy succeeds
   - Block when recommendation != "recommended"
   - Include reasoning in error message

4. TestPolicyHealthCheck (7 tests)
   - Handle no active policy
   - Correct metric structure
   - Warn on MAE degradation
   - Warn on low coverage
   - Warn on high tail miss
   - OK status with no warnings
   - WARN status with warnings

5. TestNoSideEffects (3 tests)
   - Promotion doesn't modify input report
   - Health check is advisory-only
   - Failed promotion doesn't write files

6. TestEdgeCases (2 tests)
   - Promotion with empty notes
   - Error on unknown market

Results: ✅ All 24 tests PASS
Existing tests: ✅ All 45 tests (Phase 12 + Phase 13) PASS


COMMAND EXAMPLES
================

1. Run Phase 11 A/B Evaluation
   
   python -m src.cli.pipeline calibration-ab \
     --sport nba --season 2025-26 \
     --source ensemble_total_v1 \
     --fit-start 2025-10-01 --fit-end 2025-12-31 \
     --eval-start 2026-01-01 --eval-end 2026-01-28 \
     --output-dir outputs/calibration_ab
   
   Output: outputs/calibration_ab/calibration_ab_report.json


2. Check Recommendation and Promote from Report
   
   python -m src.cli.pipeline calibration-promote-total-from-report \
     --sport nba --season 2025-26 \
     --report outputs/calibration_ab/calibration_ab_report.json \
     --notes "Phase 11 recommended bucketing"
   
   Output: Promoted to data/calibrators/nba/2025-26/historical/total/active.json
   
   Success: "[calibration promote] market=TOTAL promotion_complete"
   Failure: "[calibration promote] error: promotion_blocked" + reasoning


3. Monitor Policy Health (Advisory)
   
   python -m src.cli.pipeline calibration-policy-health \
     --sport nba --season 2025-26 \
     --market total \
     --window-games 100
   
   Output:
   [calibration policy-health] market=total status=OK
     MAE:              2.45
     RMSE:             3.12
     Coverage (2-sig): 95.50%
     Tail miss rate:   8.00%
     Samples:          100
     No warnings.
     Notes: Evaluated 100 recent games. Policy mode: total_bucket. No automatic action taken.


4. Rollback Policy (Manual)
   
   python -m src.cli.pipeline calibration-rollback-total \
     --sport nba --season 2025-26
   
   Output: Removed active.json, saved backup to active.prev.20260128_143022.json


FILE CHANGES
============

1. src/pipelines/calibration_ab.py
   - Updated CalibrationABReport: +2 fields (global_calibrator_path, bucket_manifest_path)
   - Updated to_dict(): Include new fields in JSON output
   - Modified run_calibration_ab(): Save calibrator paths to report
   - Change: global_var_cal now saves path (added sport/season/source_id)

2. src/pipelines/calibration_promote.py (NEW)
   - 450+ lines
   - load_calibration_ab_report()
   - validate_ab_report_for_promotion()
   - promote_policy_from_report()
   - evaluate_policy_health()
   - _apply_policy_to_dataset() [helper]

3. src/cli/pipeline.py
   - Added 2 argparse subparsers (calibration-promote-total-from-report, calibration-policy-health)
   - Added 2 command handlers (_run_calibration_promote_total_from_report, _run_calibration_policy_health)
   - Updated main dispatcher

4. docs/CLI.md
   - Added Phase 11 command: calibration-ab
   - Added Phase 12 commands: calibration-policy, calibration-promote-total, calibration-rollback-total
   - Added Phase 13 commands: calibration-promote-total-from-report, calibration-policy-health

5. tests/test_phase13_promotion_health.py (NEW)
   - 24 comprehensive tests
   - ~350 lines
   - Fixtures for various A/B report scenarios


CONSTRAINTS MET
===============

✅ Do NOT refactor unrelated modules
  - Only modified calibration_ab.py and cli/pipeline.py (small, scoped changes)

✅ Do NOT change existing model fitting/training logic
  - No changes to model implementations
  - Phase 8/9/10 functions unchanged

✅ Prefer additive changes
  - New module (calibration_promote.py)
  - New CLI commands (non-breaking)
  - Extended existing report structure (backward-compatible)

✅ Add tests
  - 24 new tests all passing
  - No breaking changes to existing tests

✅ ASCII-only logs
  - All print statements use ASCII characters
  - No emoji or special characters

✅ No automatic side effects
  - Health check is advisory only (no files modified, no auto-rollback)
  - Promotion is opt-in via explicit CLI command


KEY DESIGN DECISIONS
====================

1. Report Calibrator Paths
   - Stored in A/B report to eliminate re-specification
   - Allows promotion from report without manual path lookup
   - Only two optional fields; doesn't affect existing reports

2. Promotion Validation
   - Explicit field name validation (recommend over generic "invalid policy")
   - Clear error messages when recommendation != "recommended"
   - Includes reasoning from A/B report in error message

3. Health Check Advisory-Only
   - No automatic rollback (manual intervention required)
   - Returns status + warnings + metrics
   - Consistent with Phase 11 tolerance thresholds
   - Can be run without risk (read-only operation)

4. Policy Application Logic
   - Reuses Phase 12 policy schema (no new format)
   - Supports both global and total_bucket modes
   - Handles missing calibrators gracefully (fallback to global)

5. Error Handling
   - FileNotFoundError: Missing report
   - ValueError: Invalid JSON, missing fields, recommendation != "recommended"
   - Clear messages with field names and expected values
   - No silent failures


FUTURE EXTENSIONS
==================

Optional improvements (not in scope for Phase 13):

1. Historical health tracking
   - Store health check results in DB
   - Compare health metrics over time
   - Detect degradation trends

2. Automatic scheduled health checks
   - Periodic evaluation of active policies
   - Webhook/notification on degradation

3. Policy A/B testing
   - Run schedule with both policies
   - Compare live performance
   - Gradual rollout of new policy

4. Multi-market health monitoring
   - Extend health checks to ML, SPREAD markets
   - Unified dashboard view


VALIDATION CHECKLIST
====================

✅ Command line registration
   - Both commands appear in --help
   - Arguments properly defined

✅ Python syntax
   - All files compile without errors
   - Imports resolve correctly

✅ Tests
   - 24 Phase 13 tests pass
   - 21 Phase 12 tests still pass
   - 45 total tests pass

✅ Documentation
   - Phase 11, 12, 13 commands documented in CLI.md
   - Usage examples provided
   - Options and output format explained

✅ No breaking changes
   - Existing phase code unchanged (except small additions)
   - Report format backward compatible
   - CLI commands additive only

✅ Error handling
   - Clear error messages
   - No silent failures
   - Validation at entry points

✅ Advisory-only health check
   - Returns warnings but doesn't modify state
   - Encourages manual verification before rollback
   - Consistent with Phase 11/12 philosophy


SUMMARY
=======

Phase 13 successfully adds a convenience promotion layer for TOTAL calibration policies with
advisory health monitoring. Users can now:

1. Run Phase 11 A/B evaluation (produces recommendation + calibrator paths)
2. Promote recommended policies via single command (calibration-promote-total-from-report)
3. Check policy health on recent games (calibration-policy-health, advisory only)
4. Manually rollback if needed (calibration-rollback-total)

All existing functionality preserved. 45 tests pass. Documentation complete.
