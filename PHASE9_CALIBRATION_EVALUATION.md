# Phase 9: Calibration Evaluation & Regime Diagnostics

## Overview

Phase 9 implements calibration quality evaluation without modifying any predictions, calibration logic, or betting formulas. This phase focuses on:

- Computing evaluation metrics for ML and SPREAD/TOTAL markets
- Slicing results by regime (favorite/underdog, buckets, home/away, season segment, market distance)
- Detecting drift and instability across time windows
- Producing auditable JSON reports and optional CSV outputs

**No predictions are changed. No calibration parameters are modified. Only metrics are computed.**

## Key Components

### 1. Calibration Evaluation Module (`src/pipelines/calibration_evaluation.py`)

#### ML Metrics
- **Brier Score**: Mean squared error between predicted and actual probabilities
- **Log Loss**: Cross-entropy loss (lower is better)
- **Calibration Error (ECE)**: Expected Calibration Error across bins
- **Reliability Curve**: Predicted vs actual probabilities in bins

#### SPREAD/TOTAL Metrics
- **MAE**: Mean Absolute Error of predictions vs actuals
- **RMSE**: Root Mean Squared Error
- **MSE**: Mean Signed Error (bias detection)
- **Empirical Coverage**: Fraction of actuals within 1σ and 2σ bands
- **Tail Miss Rate**: Fraction of actuals beyond 2σ (risk metric)

#### Slicing Framework
Composable regime slicers partition schedule data:

| Slicer | Purpose |
|--------|---------|
| `FavoriteVsUnderdogSlicer` | Separate by line sign (favorite < 0 vs underdog > 0) |
| `SpreadBucketSlicer` | Partition by absolute spread magnitude |
| `TotalBucketSlicer` | Partition by total score bucket |
| `HomeAwaySlicer` | Split by home vs away |
| `SeasonSegmentSlicer` | Split by calendar month ranges |
| `MarketDistanceSlicer` | Partition by \|line - pred_mean\| |

All slicers return `Dict[regime_name -> filtered_df]`.

#### Drift & Stability
- **DriftMetrics**: Compare fit window vs eval window; flag degradation beyond tolerance
- **Rolling Window Metrics**: Track metric values over time windows (7-day default)
- **Drift Computation**: Compare metric values across windows with percent change tracking

#### Reporting
- **MarketEvaluationReport**: Dataclass with top-level metrics, per-regime metrics, drift results, rolling window data
- **JSON Output**: Complete report persisted to `calibration_report_<market>.json`
- **CSV Output** (optional): Regime metrics and rolling window results
- **Console Summary**: High-level overview with worst-performing regimes flagged

### 2. CLI Integration (`src/cli/pipeline.py`)

#### Command: `calibration-report`

```bash
python -m src.cli.pipeline calibration-report \
  --sport nba \
  --season 2025-26 \
  --schedule-path data/processed/nba/2025-26/schedule_with_projections.csv \
  --markets "ML,spread,total" \
  --slicing-regimes '{"favorite_vs_underdog": {}, "spread_bucket": {"buckets": [0, 3, 6, 10, 20]}}' \
  --fit-start-date 2025-01-01 \
  --fit-end-date 2025-02-01 \
  --eval-start-date 2025-02-01 \
  --eval-end-date 2025-03-01 \
  --rolling-window-days 7 \
  --output-dir outputs/calibration_reports \
  --csv-output
```

#### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--sport` | Yes | - | Sport identifier (e.g., nba) |
| `--season` | Yes | - | Season identifier (e.g., 2025-26) |
| `--schedule-path` | Yes | - | Path to CSV/XLSX with post-calibration projections |
| `--markets` | No | ML,spread,total | Comma-separated markets to evaluate |
| `--slicing-regimes` | No | (none) | JSON config for regime slicing |
| `--fit-start-date` | No | (none) | Fit window start date (YYYY-MM-DD) for drift |
| `--fit-end-date` | No | (none) | Fit window end date (YYYY-MM-DD) for drift |
| `--eval-start-date` | No | (none) | Eval window start date (YYYY-MM-DD) for drift |
| `--eval-end-date` | No | (none) | Eval window end date (YYYY-MM-DD) for drift |
| `--rolling-window-days` | No | (none) | Window size for rolling metric tracking |
| `--output-dir` | No | outputs/calibration_reports | Directory for JSON/CSV outputs |
| `--csv-output` | No | False | Also export regime metrics and rolling results to CSV |

### 3. Tests (`tests/test_calibration_evaluation.py`)

#### Coverage (49 tests)
- **ML Metrics** (13 tests): Brier, log loss, calibration error, reliability curve
- **SPREAD/TOTAL Metrics** (13 tests): MAE, RMSE, MSE, bias, coverage, tail miss
- **Slicing** (12 tests): All slicer types, factory pattern, edge cases
- **Drift & Stability** (7 tests): Drift metrics, rolling windows, NaN handling
- **Reporting** (4 tests): Report creation, serialization, summary generation

All tests use synthetic data and verify:
- Correctness on known inputs
- Edge case handling (empty data, NaN, clipping)
- No mutation of input DataFrames
- Proper serialization to JSON

## Usage Examples

### Basic Evaluation: Top-Level Metrics Only

```bash
python -m src.cli.pipeline calibration-report \
  --sport nba \
  --season 2025-26 \
  --schedule-path ./schedule.csv
```

Output: `outputs/calibration_reports/calibration_report_ML.json` with Brier, log loss, ECE

### With Regime Slicing

```bash
python -m src.cli.pipeline calibration-report \
  --sport nba \
  --season 2025-26 \
  --schedule-path ./schedule.csv \
  --slicing-regimes '{"favorite_vs_underdog": {}, "spread_bucket": {}}'
```

Output includes per-regime breakdowns (e.g., "favorite_vs_underdog_favorites", "spread_bucket_0-3")

### With Drift Detection

```bash
python -m src.cli.pipeline calibration-report \
  --sport nba \
  --season 2025-26 \
  --schedule-path ./schedule.csv \
  --fit-start-date 2024-11-01 \
  --fit-end-date 2025-01-01 \
  --eval-start-date 2025-01-01 \
  --eval-end-date 2025-03-01
```

Output compares calibration fit window vs eval window, flags if metrics degrade >15%

### With Rolling Window Stability

```bash
python -m src.cli.pipeline calibration-report \
  --sport nba \
  --season 2025-26 \
  --schedule-path ./schedule.csv \
  --rolling-window-days 7 \
  --csv-output
```

Output includes `rolling_window_ML.csv` with weekly metric tracking

## Output Structure

```
outputs/calibration_reports/
├── calibration_report_ML.json          # ML metrics, regimes, drift, rolling
├── calibration_report_spread.json      # SPREAD metrics, regimes, rolling
├── calibration_report_total.json       # TOTAL metrics, regimes, rolling
├── regime_metrics_ML.csv               # (optional) per-regime breakdown
├── regime_metrics_spread.csv           # (optional) per-regime breakdown
├── regime_metrics_total.csv            # (optional) per-regime breakdown
├── rolling_window_ML.csv               # (optional) time series metrics
├── rolling_window_spread.csv           # (optional) time series metrics
└── rolling_window_total.csv            # (optional) time series metrics
```

### JSON Report Schema

```json
{
  "market": "ML",
  "num_samples": 250,
  "num_regimes": 6,
  "timestamp": "2025-01-28T14:30:00",
  "metrics": {
    "brier_score": 0.1842,
    "log_loss": 0.4123,
    "calibration_error": 0.0654
  },
  "regime_metrics": {
    "favorite_vs_underdog_favorites": {
      "brier_score": 0.1625,
      "log_loss": 0.3844,
      "calibration_error": 0.0512,
      "sample_count": 125
    },
    "favorite_vs_underdog_underdogs": {
      "brier_score": 0.2059,
      "log_loss": 0.4402,
      "calibration_error": 0.0796,
      "sample_count": 125
    }
  },
  "drift_metrics": {
    "brier_score_drift": {
      "metric_name": "brier_score",
      "fit_window_value": 0.1800,
      "eval_window_value": 0.1900,
      "absolute_change": 0.0100,
      "percent_change": 0.0556,
      "tolerance": 0.15,
      "is_degraded": false
    }
  },
  "rolling_window_results": [
    {
      "window_start": "2025-01-01",
      "window_end": "2025-01-08",
      "metric_value": 0.1823,
      "sample_count": 38
    }
  ]
}
```

## Design Principles

### No Mutation
- Evaluation metrics are **read-only**
- Input schedules and calibration data are never modified
- All slicers return filtered **copies** of DataFrames

### Composability
- Slicers can be mixed/matched via JSON config
- Metrics work independently of slicing
- Drift detection works with any metric function

### Observability
- JSON reports are self-contained and auditableAll results include timestamps, sample counts, metric names
- Warnings logged at INFO level for degraded metrics

### Isolation
- Evaluation logic is separate from prediction/calibration modules
- No dependencies on betting logic or BETS sheet formulas
- Can be run on any schedule CSV with post-calibration columns

## Calibration Quality Thresholds (Guidance)

These are **not enforced** but helpful for interpretation:

### ML Markets (Brier Score)
- **Excellent**: < 0.12
- **Good**: 0.12 - 0.18
- **Fair**: 0.18 - 0.25
- **Poor**: > 0.25

### SPREAD/TOTAL Markets (Empirical Coverage @ 1σ)
- **Excellent**: 65-70% (matches theoretical ~68%)
- **Good**: 60-75%
- **Fair**: 50-80%
- **Poor**: < 50% or > 85%

### Drift Detection
- **Alert Threshold**: 15% change from fit to eval window
- **Critical**: > 25% change suggests model/market regime shift

## Integration with Phase 8

Phase 9 **complements** Phase 8 (calibration fitting/provenance):
- Phase 8: Fits variance calibrators, appends provenance tags to `win_prob_source`
- Phase 9: Evaluates fit quality, identifies regime-specific failures

Both phases are **non-invasive**:
- No changes to predictions
- No changes to ensemble logic
- No changes to BETS formulas

## Testing

Run all evaluation tests:
```bash
pytest tests/test_calibration_evaluation.py -v
```

Run specific test class:
```bash
pytest tests/test_calibration_evaluation.py::TestMLMetrics -v
pytest tests/test_calibration_evaluation.py::TestSlicing -v
```

Run with coverage:
```bash
pytest tests/test_calibration_evaluation.py --cov=src.pipelines.calibration_evaluation
```

## Future Enhancements

1. **Visualization**: Plotting reliability curves, rolling metrics over time
2. **Threshold-Based Alerts**: Automatic warnings for critical metrics
3. **Comparison Reports**: A/B comparison of calibration across models
4. **Per-Team Slicing**: Regime slicing by specific team (NBA teams, NFL, etc.)
5. **Calibration Adjustment Recommendations**: Suggest when to refit calibrators

## References

- **Reliability Diagrams**: [Guo et al. 2017 - On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
- **Expected Calibration Error**: [Niculescu-Mizil & Caruana 2005](https://www.semanticscholar.org/paper/Predicting-good-probabilities-with-supervised-Niculescu-Mizil-Caruana/f4c3a84dc79be0a8c3e21d39df8a3f3c1a4d96f9)
- **Tail Risk in Predictions**: Standard practice in forecast verification

---

**Phase 9 Status**: ✅ Complete
- Evaluation metrics implemented and tested
- Regime slicing framework operational
- CLI integration ready
- No prediction/calibration changes
- 49 tests passing
- Phase 8 tests still passing (no regression)
