Phase 7: Audited Ensemble Weight Resolution Integration
========================================================

## Summary

Phase 7 successfully integrates the Phase 6 ensemble governance (audited weight resolution) into the production schedule pipeline. The integration ensures that:

1. **Audited weight resolution** is used for ML, SPREAD, and TOTAL market ensembles
2. **CLI --strict flag** controls governance strictness (overrides ENSEMBLE_STRICT_MODE config)
3. **Audit logs and metadata** are emitted once per market and stored in resolved_ensemble_meta
4. **Graceful degradation** occurs in non-strict mode when ensembles can't be formed

## Implementation Details

### Files Modified

- **src/pipelines/schedule.py**
  - Updated `_resolve_ensemble_weights_with_audit()` to accept `strict` parameter
  - Replaced `_filter_market_weights_for_forecast()` calls with audited resolver in ML/SPREAD/TOTAL ensemble paths
  - Added `ensemble_audits` dictionary to track audits per market
  - Stored audit.to_dict() JSON in resolved_ensemble_meta under "audit" key
  - Wrapped audited resolver in try-except to gracefully handle non-strict failures

### Key Changes

1. **Strict Mode Plumbing**
   - CLI `--strict` flag is passed to `build_schedule_excel_report()`
   - Effective strict mode: `effective_strict = ENSEMBLE_STRICT_MODE if strict is None else strict`
   - RuntimeError raised before workbook writing if Neff < MIN_NEFF in strict mode

2. **Audited Resolution in ML/SPREAD/TOTAL**
   ```python
   audit = _resolve_ensemble_weights_with_audit(
       weights=weights,
       forecast_df=forecast_df,
       market=Market.{ML|SPREAD|TOTAL},
       weight_source=weight_source or "equal",
       weight_run_id=weight_run_id,
       selection_run_id=selection_run_id,
       strict=strict,  # CLI --strict flag override
   )
   ```

3. **Graceful Degradation**
   - If audit returns empty final_models in non-strict mode: log warning, set use_ensemble=False
   - If audit raises RuntimeError in strict mode: propagate (fail before workbook writing)
   - If audit raises RuntimeError in non-strict mode: log warning, skip ensemble

4. **Audit Metadata Storage**
   ```python
   resolved_ensemble_meta[market_name]["audit"] = audit.to_dict()
   ensemble_audits[market_name] = audit
   ```

## Testing

### Phase 7 Integration Tests (tests/test_phase7_audited_ensemble_integration.py)

All 11 tests pass:
- `test_small_weight_clamping_in_audit` - Verifies MIN_WEIGHT_EPS enforcement
- `test_strict_mode_raises_on_low_neff` - Verifies RuntimeError in strict mode
- `test_non_strict_mode_allows_low_neff` - Verifies graceful degradation
- `test_fallback_applied_on_weight_collapse` - Verifies fallback logic
- `test_one_audit_log_per_market` - Verifies audit logging (once per market)
- `test_audit_to_dict_serializable` - Verifies JSON serialization
- `test_strict_parameter_overrides_ensemble_strict_mode` - Verifies CLI override
- `test_multiple_markets_audit_metadata` - Verifies per-market audit storage
- `test_coverage_summary_in_audit` - Verifies coverage tracking
- `test_strict_true_with_truly_low_neff` - Verifies strict enforcement
- `test_strict_false_completes_with_warning_on_valid_scenario` - Verifies graceful completion

### Phase 6 Tests (tests/test_phase6_ensemble_governance.py)

All 25 tests still pass, verifying backward compatibility.

### Run Tests

```bash
# Phase 7 integration tests
python -m pytest tests/test_phase7_audited_ensemble_integration.py -v

# Phase 6 governance tests (backward compatibility)
python -m pytest tests/test_phase6_ensemble_governance.py -v

# Combined
python -m pytest tests/test_phase6_ensemble_governance.py tests/test_phase7_audited_ensemble_integration.py -q
```

## CLI Usage

### Default (Non-Strict) Mode
```bash
python -m src.cli.pipeline schedule \
    --sport nba \
    --season 2025-26
```
- Uses audited resolver with strict=False
- Allows fallback when ensembles can't be formed
- Logs warnings but continues processing

### Strict Mode
```bash
python -m src.cli.pipeline schedule \
    --sport nba \
    --season 2025-26 \
    --strict
```
- Uses audited resolver with strict=True
- Raises RuntimeError if Neff < MIN_NEFF after governance
- Fails before writing workbook

## Audit Output

Each market's audit is logged with INFO/WARNING level:
```
[ensemble audit][ML] final_models=['elo', 'bradley_terry'], neff=2.0, ...
[ensemble audit] Weight clamping applied for model X
[ensemble audit] Fallback applied: using uniform weights over [elo, bradley_terry, gssd]
```

Audit metadata stored in workbook metadata:
```python
resolved_ensemble_meta = {
    "ML": {
        "ensemble_id": "ensemble_ml_v1",
        "audit": {
            "market": "ML",
            "final_models": ["elo", "bradley_terry"],
            "final_weights": {...},
            "dropped_models": {...},
            "neff": 2.0,
            "neff_threshold_met": True,
            "weight_clamped": False,
            "fallback_applied": False,
            "warnings": [],
        }
    }
}
```

## Assumptions & Design

1. **One Audit Per Market**: Each market (ML/SPREAD/TOTAL) has exactly one audit object, created during ensemble weight resolution.

2. **Strict Mode Override**: CLI --strict parameter takes precedence over ENSEMBLE_STRICT_MODE config default.

3. **Fallback Semantics**: In non-strict mode, fallback to uniform weights is preferred over hard failure. Warnings are logged to explain why.

4. **Coverage Tracking**: Audits include per-model coverage stats (games_with_forecasts, required_columns, etc.) for traceability.

5. **JSON Serialization**: All audit fields are JSON-serializable via audit.to_dict() for metadata storage and logging.

## Future Work

- Write results audit to separate metadata sheet in Excel workbook (optional)
- Add per-game ensemble component tracking via audit
- Extend strictness flags to forecast_params resolution phase
- Integrate with monitoring/alerting for governance violations

## Notes

- The old `_filter_market_weights_for_forecast()` function is still present but no longer called in the main schedule pipeline
- It may be kept for backward compatibility or deprecated in a future phase
- All changes are additive; no existing model math, head logic, or calibration formulas were modified
