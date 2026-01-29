# PHASE 9 API REFERENCE

Quick reference for using the calibration evaluation module.

## Module: `src.pipelines.calibration_evaluation`

### ML Metrics

#### `brier_score(probs, outcomes) -> float`
Mean squared error between predicted and actual probabilities.

```python
probs = [0.7, 0.3, 0.9]
outcomes = [1, 0, 1]
score = brier_score(probs, outcomes)  # ~0.067
```

- **Range**: [0, 1]
- **Better**: Lower is better
- **Clipping**: Probs clipped to [0, 1]

---

#### `log_loss(probs, outcomes) -> float`
Cross-entropy loss (negative log likelihood).

```python
score = log_loss([0.7, 0.3], [1, 0])  # ~0.357
```

- **Range**: [0, ∞)
- **Better**: Lower is better
- **Clipping**: Probs clipped to [1e-12, 1-1e-12] to avoid log(0)

---

#### `calibration_error(probs, outcomes) -> float`
Expected Calibration Error (ECE) - average |predicted - actual| per bin.

```python
ece = calibration_error([0.3, 0.5, 0.7], [0, 0, 1])  # ~0.067
```

- **Range**: [0, 1]
- **Better**: Lower is better
- **Usage**: Measures calibration across probability bins

---

#### `reliability_curve(probs, outcomes, *, n_bins=10, min_count=5) -> pd.DataFrame`
Predicted vs actual probabilities in bins.

```python
df = reliability_curve(probs, outcomes, n_bins=5, min_count=10)
# Columns: bin_lower, bin_upper, count, avg_pred, avg_actual, calibration_gap
```

**Returns**:
```
   bin_lower  bin_upper  count  avg_pred  avg_actual  calibration_gap
0       0.0        0.2     42      0.10       0.12            -0.02
1       0.2        0.4     38      0.30       0.26             0.04
2       0.4        0.6     35      0.50       0.51            -0.01
```

---

### SPREAD/TOTAL Metrics

#### `mean_absolute_error(pred_means, actual) -> float`
MAE: mean(|pred - actual|)

```python
mae = mean_absolute_error([105, 210], [103, 208])  # 1.5
```

- **Range**: [0, ∞)
- **Better**: Lower is better

---

#### `root_mean_squared_error(pred_means, actual) -> float`
RMSE: sqrt(mean((pred - actual)^2))

```python
rmse = root_mean_squared_error([105, 210], [103, 208])  # 1.581
```

- **Range**: [0, ∞)
- **Better**: Lower is better
- **Penalizes**: Larger errors more than MAE

---

#### `mean_squared_error(pred_means, actual) -> float`
MSE: mean((pred - actual)^2)

```python
mse = mean_squared_error([105, 210], [103, 208])  # 2.5
```

- **Range**: [0, ∞)
- **Better**: Lower is better

---

#### `mean_signed_error(pred_means, actual) -> float`
Bias: mean(pred - actual). Positive = overestimate.

```python
bias = mean_signed_error([105, 210], [103, 208])  # +1.5 (overestimating)
```

- **Range**: (-∞, ∞)
- **Interpretation**: Positive bias = systematic overestimate
- **Better**: Close to 0 indicates no bias

---

#### `empirical_coverage(pred_means, pred_sds, actual, *, sigma=1.0) -> float`
Fraction of actuals within pred_mean ± sigma*pred_sd.

```python
# For 1-sigma, expect ~68% coverage on perfectly calibrated data
coverage_1sig = empirical_coverage(
    [100, 200], [5, 8], [102, 198], sigma=1.0
)  # ~0.5 in this example

# For 2-sigma, expect ~95% coverage
coverage_2sig = empirical_coverage(
    [100, 200], [5, 8], [102, 198], sigma=2.0
)  # ~1.0 in this example
```

- **Range**: [0, 1]
- **Expected**: σ=1.0 → ~68%, σ=2.0 → ~95%
- **Sigma=1.0**: "1-sigma coverage" (68% expected)
- **Sigma=2.0**: "2-sigma coverage" (95% expected)

---

#### `tail_miss_rate(pred_means, pred_sds, actual, *, sigma=2.0) -> float`
Fraction of actuals beyond sigma*pred_sd (tail risk).

```python
# Expected ~5% miss rate on perfectly calibrated data at 2-sigma
miss_rate = tail_miss_rate(
    [100, 200], [5, 8], [120, 218], sigma=2.0
)
```

- **Range**: [0, 1]
- **Expected**: σ=2.0 → ~5%
- **Interpretation**: Higher = riskier (underestimating uncertainty)

---

### Regime Slicing

#### `SlicerConfig(regime_type, name, kwargs={})`
Configuration dataclass for creating slicers.

```python
config = SlicerConfig(
    regime_type=RegimeType.SPREAD_BUCKET,
    name="spread_buckets",
    kwargs={"buckets": [0, 3, 6, 10, 20]}
)
```

---

#### `build_slicer(config) -> Slicer`
Factory function to build slicer from config.

```python
slicer = build_slicer(config)
```

---

#### Slicer Classes

All implement `slice(df) -> Dict[str, pd.DataFrame]`

**`FavoriteVsUnderdogSlicer`**
```python
slicer = FavoriteVsUnderdogSlicer(spread_col="opening_spread")
slices = slicer.slice(df)
# Returns: {"favorites": df_fav, "underdogs": df_ud}
```

**`SpreadBucketSlicer`**
```python
slicer = SpreadBucketSlicer(
    spread_col="opening_spread",
    buckets=[0, 3, 6, 10, 20]
)
slices = slicer.slice(df)
# Returns: {"0-3": df, "3-6": df, ..., "20+": df}
```

**`TotalBucketSlicer`**
```python
slicer = TotalBucketSlicer(
    total_col="opening_total",
    buckets=[0, 180, 210, 240, 500]
)
slices = slicer.slice(df)
# Returns: {"0-180": df, "180-210": df, ..., "500+": df}
```

**`HomeAwaySlicer`**
```python
slicer = HomeAwaySlicer()
slices = slicer.slice(df)  # Requires "is_home" column
# Returns: {"home": df_home, "away": df_away}
```

**`SeasonSegmentSlicer`**
```python
slicer = SeasonSegmentSlicer(
    date_col="date",
    boundaries=[(1, 3), (4, 6), (7, 9), (10, 12)]
)
slices = slicer.slice(df)
# Returns: {"01-03": df, "04-06": df, ...}
```

**`MarketDistanceSlicer`**
```python
slicer = MarketDistanceSlicer(
    line_col="opening_spread",
    mean_col="proj_margin_mean",
    buckets=[0, 1, 2, 4, 100]
)
slices = slicer.slice(df)
# Returns: {"0-1": df, "1-2": df, ..., "100+": df}
```

---

### Drift & Stability

#### `DriftMetrics` Dataclass
Represents drift comparison between fit and eval windows.

```python
drift = DriftMetrics(
    metric_name="brier_score",
    fit_window_value=0.15,
    eval_window_value=0.18,
    absolute_change=0.03,
    percent_change=0.20,  # 20% change
    tolerance=0.15,  # 15% threshold
)

print(drift.is_degraded)  # True (20% > 15% threshold)
```

**Fields**:
- `metric_name`: Name of the metric
- `fit_window_value`: Value on fit data
- `eval_window_value`: Value on eval data
- `absolute_change`: eval - fit
- `percent_change`: (eval - fit) / fit
- `tolerance`: Threshold for degradation (default 0.15)
- `is_degraded`: Bool, set by `__post_init__`

---

#### `compute_drift_metrics(fit_df, eval_df, *, metric_fn, metric_name, tolerance=0.15) -> DriftMetrics`
Compare metric across fit vs eval windows.

```python
fit_df = schedule[schedule.date < "2025-02-01"]
eval_df = schedule[schedule.date >= "2025-02-01"]

drift = compute_drift_metrics(
    fit_df,
    eval_df,
    metric_fn=lambda x: brier_score(x["prob"], x["outcome"]),
    metric_name="brier_score",
    tolerance=0.15
)

if drift.is_degraded:
    print(f"Alert: {drift.metric_name} degraded by {drift.percent_change:+.2%}")
```

---

#### `rolling_window_metrics(df, *, metric_fn, date_col="date", window_days=7) -> pd.DataFrame`
Track metric over rolling time windows.

```python
rolling = rolling_window_metrics(
    df,
    metric_fn=lambda x: mean_absolute_error(x["pred"], x["actual"]),
    date_col="date",
    window_days=7,
)

# Returns:
#   window_start  window_end  metric_value  sample_count
# 0    2025-01-01  2025-01-08          2.5             38
# 1    2025-01-08  2025-01-15          2.3             42
# ...
```

---

### Reporting

#### `MarketEvaluationReport` Dataclass
Aggregates all evaluation results for a market.

```python
report = MarketEvaluationReport(
    market="ML",
    num_samples=250,
    num_regimes=6,
    metrics={"brier_score": 0.15, "log_loss": 0.40},
    regime_metrics={
        "fav": {"brier_score": 0.12},
        "ud": {"brier_score": 0.18},
    },
)

# Write to JSON
report.to_json(Path("outputs/calibration_report_ML.json"))

# Convert to dict for custom serialization
d = report.to_dict()
```

**Attributes**:
- `market`: Market name (e.g., "ML", "SPREAD", "TOTAL")
- `num_samples`: Number of games evaluated
- `num_regimes`: Number of regime slices
- `timestamp`: When report was generated
- `metrics`: Top-level metric dict
- `regime_metrics`: Per-regime metric dicts
- `drift_metrics`: DriftMetrics objects by name
- `rolling_window_results`: DataFrame of rolling metrics

---

#### `summarize_evaluation(reports, *, max_regimes=5) -> str`
Generate human-readable summary.

```python
reports = {
    "ML": report_ml,
    "SPREAD": report_spread,
    "TOTAL": report_total,
}

summary = summarize_evaluation(reports, max_regimes=3)
print(summary)

# Output:
# ======================================================================
# CALIBRATION EVALUATION SUMMARY
# ======================================================================
# 
# ML:
#   Samples: 250
#   Regimes evaluated: 6
#   Top-level metrics:
#     brier_score: 0.1523
#     log_loss: 0.4012
#   ...
```

---

## Enum: `RegimeType`
Available regime slicing dimensions:

```python
class RegimeType(Enum):
    FAVORITE_VS_UNDERDOG = "favorite_vs_underdog"
    SPREAD_BUCKET = "spread_bucket"
    TOTAL_BUCKET = "total_bucket"
    HOME_VS_AWAY = "home_vs_away"
    SEASON_SEGMENT = "season_segment"
    MARKET_DISTANCE = "market_distance"
```

---

## CLI: `calibration-report`

```bash
python -m src.cli.pipeline calibration-report \
  --sport nba \
  --season 2025-26 \
  --schedule-path schedule.csv \
  --markets "ML,spread,total" \
  --slicing-regimes '{"favorite_vs_underdog": {}, "spread_bucket": {}}' \
  --fit-start-date 2025-01-01 \
  --fit-end-date 2025-02-01 \
  --eval-start-date 2025-02-01 \
  --eval-end-date 2025-03-01 \
  --rolling-window-days 7 \
  --output-dir outputs/calibration_reports \
  --csv-output
```

**Arguments**:
- `--sport` (required): Sport code
- `--season` (required): Season code (YYYY-YY)
- `--schedule-path` (required): Path to CSV/XLSX
- `--markets` (default: "ML,spread,total"): Comma-separated list
- `--slicing-regimes`: JSON config for regimes
- `--fit-start-date`: Fit window start (YYYY-MM-DD)
- `--fit-end-date`: Fit window end (YYYY-MM-DD)
- `--eval-start-date`: Eval window start (YYYY-MM-DD)
- `--eval-end-date`: Eval window end (YYYY-MM-DD)
- `--rolling-window-days`: Window size for rolling metrics
- `--output-dir` (default: outputs/calibration_reports): Output directory
- `--csv-output`: Also export CSV files

---

## Example: End-to-End Usage

```python
import pandas as pd
from pathlib import Path
from src.pipelines.calibration_evaluation import (
    brier_score,
    log_loss,
    reliability_curve,
    SlicerConfig,
    RegimeType,
    build_slicer,
    compute_drift_metrics,
    MarketEvaluationReport,
    summarize_evaluation,
)

# Load schedule
df = pd.read_csv("schedule.csv")

# Split into fit/eval windows
fit_df = df[df["date"] < "2025-02-01"]
eval_df = df[df["date"] >= "2025-02-01"]

# Compute top-level ML metrics
outcomes = (eval_df["home_score"] > eval_df["away_score"]).astype(float)
probs = eval_df["win_prob_home"]

metrics = {
    "brier_score": brier_score(probs, outcomes),
    "log_loss": log_loss(probs, outcomes),
    "calibration_error": calibration_error(probs, outcomes),
}

# Slice by favorite/underdog
slicer = build_slicer(SlicerConfig(
    regime_type=RegimeType.FAVORITE_VS_UNDERDOG,
    name="fav_vs_ud",
))

regime_metrics = {}
for slice_name, slice_df in slicer.slice(eval_df).items():
    slice_outcomes = (slice_df["home_score"] > slice_df["away_score"]).astype(float)
    slice_probs = slice_df["win_prob_home"]
    regime_metrics[f"fav_vs_ud_{slice_name}"] = {
        "brier_score": brier_score(slice_probs, slice_outcomes),
    }

# Compute drift
drift = compute_drift_metrics(
    fit_df,
    eval_df,
    metric_fn=lambda x: brier_score(
        x["win_prob_home"],
        (x["home_score"] > x["away_score"]).astype(float)
    ),
    metric_name="brier_score",
)

# Build report
report = MarketEvaluationReport(
    market="ML",
    num_samples=len(eval_df),
    num_regimes=len(regime_metrics),
    metrics=metrics,
    regime_metrics=regime_metrics,
    drift_metrics={"brier_score_drift": drift},
)

# Save and summarize
report.to_json(Path("calibration_report_ML.json"))
print(summarize_evaluation({"ML": report}))
```

---

## Common Patterns

### Pattern 1: Evaluate All Markets with Default Regimes
```bash
python -m src.cli.pipeline calibration-report \
  --sport nba --season 2025-26 --schedule-path schedule.csv
```

### Pattern 2: Deep Dive on Single Market
```bash
python -m src.cli.pipeline calibration-report \
  --sport nba --season 2025-26 --schedule-path schedule.csv \
  --markets "ML" \
  --slicing-regimes '{"favorite_vs_underdog": {}, "spread_bucket": {"buckets": [0, 2, 5, 10]}, "season_segment": {}}'
```

### Pattern 3: Drift Detection Only
```bash
python -m src.cli.pipeline calibration-report \
  --sport nba --season 2025-26 --schedule-path schedule.csv \
  --fit-start-date 2024-11-01 --fit-end-date 2025-01-01 \
  --eval-start-date 2025-01-01 --eval-end-date 2025-03-01
```

### Pattern 4: Rolling Window Stability
```bash
python -m src.cli.pipeline calibration-report \
  --sport nba --season 2025-26 --schedule-path schedule.csv \
  --rolling-window-days 7 --csv-output
# Generates rolling_window_ML.csv, rolling_window_SPREAD.csv, etc.
```

---

**Last Updated**: January 28, 2026
