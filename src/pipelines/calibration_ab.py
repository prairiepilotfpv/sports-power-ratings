"""Phase 11: A/B evaluation comparing global vs bucketed TOTAL calibration.

Thin orchestration wrapper around existing Phase 8/9/10 infrastructure.
Does NOT introduce new prediction, dataset-building, or calibration logic.

Usage:
    python -m src.cli.pipeline calibration-ab \
      --sport nba --season 2025-26 \
      --source ensemble_total_v1 \
      --fit-start 2025-10-01 --fit-end 2025-12-31 \
      --eval-start 2026-01-01 --eval-end 2026-01-28
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ArmMetrics:
    """Evaluation metrics for one calibration arm (global or bucketed)."""

    mae: float
    rmse: float
    bias: float
    coverage_1sd: float
    coverage_2sd: float
    tail_miss_rate: float
    sample_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BucketArmMetrics:
    """Per-bucket metrics for one calibration arm."""

    bucket: str
    metrics: ArmMetrics
    sample_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bucket": self.bucket,
            "metrics": self.metrics.to_dict(),
            "sample_count": self.sample_count,
        }


@dataclass
class PolicyDecision:
    """Automated recommendation from A/B comparison."""

    recommendation: str  # "recommended" | "not_recommended"
    reasoning: List[str]
    tail_miss_improved: bool
    coverage_2sd_improved: bool
    mae_within_tolerance: bool
    rmse_within_tolerance: bool
    tolerance_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationABReport:
    """Full A/B comparison report."""

    sport: str
    season: str
    market: str
    source: str
    fit_window: Dict[str, str]
    eval_window: Dict[str, str]
    baseline: ArmMetrics
    treatment: ArmMetrics
    deltas: Dict[str, float]
    baseline_buckets: List[BucketArmMetrics]
    treatment_buckets: List[BucketArmMetrics]
    policy: PolicyDecision
    bucket_thresholds: Tuple[float, float]
    min_samples_per_bucket: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # Calibrator paths for Phase 13 promotion
    global_calibrator_path: Optional[str] = None
    bucket_manifest_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sport": self.sport,
            "season": self.season,
            "market": self.market,
            "source": self.source,
            "fit_window": self.fit_window,
            "eval_window": self.eval_window,
            "baseline": self.baseline.to_dict(),
            "treatment": self.treatment.to_dict(),
            "deltas": self.deltas,
            "baseline_buckets": [b.to_dict() for b in self.baseline_buckets],
            "treatment_buckets": [b.to_dict() for b in self.treatment_buckets],
            "policy": self.policy.to_dict(),
            "bucket_thresholds": list(self.bucket_thresholds),
            "min_samples_per_bucket": self.min_samples_per_bucket,
            "timestamp": self.timestamp,
            "global_calibrator_path": self.global_calibrator_path,
            "bucket_manifest_path": self.bucket_manifest_path,
        }

    def to_json(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        _LOG.info("[CalibrationABReport] Wrote report to %s", path)


# ---------------------------------------------------------------------------
# Metric computation (delegates to Phase 9)
# ---------------------------------------------------------------------------


def _compute_arm_metrics(df: pd.DataFrame) -> ArmMetrics:
    """Compute all six evaluation metrics on a calibration dataset.

    Expects columns: pred_mean, pred_sd, actual_value.
    Calls Phase 9 metric functions directly.
    """
    from pipelines.calibration_evaluation import (
        mean_absolute_error,
        root_mean_squared_error,
        mean_signed_error,
        empirical_coverage,
        tail_miss_rate,
    )

    pred = df["pred_mean"]
    sd = df["pred_sd"]
    actual = df["actual_value"]

    return ArmMetrics(
        mae=mean_absolute_error(pred, actual),
        rmse=root_mean_squared_error(pred, actual),
        bias=mean_signed_error(pred, actual),
        coverage_1sd=empirical_coverage(pred, sd, actual, sigma=1.0),
        coverage_2sd=empirical_coverage(pred, sd, actual, sigma=2.0),
        tail_miss_rate=tail_miss_rate(pred, sd, actual),
        sample_count=len(df),
    )


# ---------------------------------------------------------------------------
# Calibrator application (uses existing transform methods)
# ---------------------------------------------------------------------------


def _apply_global_arm(
    eval_df: pd.DataFrame,
    mean_cal: Any | None,
    var_cal: Any | None,
) -> pd.DataFrame:
    """Apply global (Phase 8) calibration to eval dataset.

    Uses existing MeanCalibrator.transform() and VarianceCalibrator.transform().
    Returns a deep copy; input is not mutated.
    """
    out = eval_df.copy(deep=True)

    if mean_cal is not None:
        result = mean_cal.transform(out[["pred_mean", "pred_sd"]])
        out["pred_mean"] = result["calibrated_mean"].values
        if "calibrated_sd" in result.columns and result["calibrated_sd"].notna().any():
            out["pred_sd"] = result["calibrated_sd"].values

    if var_cal is not None:
        result = var_cal.transform(out[["pred_mean", "pred_sd"]])
        out["pred_sd"] = result["calibrated_sd"].values

    return out


def _apply_bucketed_arm(
    eval_df: pd.DataFrame,
    global_mean_cal: Any | None,
    global_var_cal: Any | None,
    bucket_cals: Dict[str, Any],
    thresholds: Tuple[float, float],
) -> pd.DataFrame:
    """Apply bucketed (Phase 10) calibration to eval dataset.

    Uses existing label_dataframe_with_total_bucket and per-bucket
    calibrator transforms with fallback to global.
    Returns a deep copy; input is not mutated.
    """
    from calibration.total_bucket_regimes import label_dataframe_with_total_bucket

    out = eval_df.copy(deep=True)
    out = label_dataframe_with_total_bucket(
        out, total_col="pred_mean", thresholds=thresholds
    )

    for bucket_label in ["low", "mid", "high"]:
        mask = out["total_bucket"] == bucket_label
        if not mask.any():
            continue

        bucket_df = out.loc[mask, ["pred_mean", "pred_sd"]].copy()

        # Mean calibration: bucket-specific or global fallback
        mean_key = f"{bucket_label}_mean"
        mean_cal = bucket_cals.get(mean_key, global_mean_cal)
        if mean_cal is not None:
            result = mean_cal.transform(bucket_df)
            out.loc[mask, "pred_mean"] = result["calibrated_mean"].values
            # Re-read updated pred_mean for variance stage
            bucket_df = out.loc[mask, ["pred_mean", "pred_sd"]].copy()

        # Variance calibration: bucket-specific or global fallback
        var_key = f"{bucket_label}_variance"
        var_cal = bucket_cals.get(var_key, global_var_cal)
        if var_cal is not None:
            result = var_cal.transform(bucket_df)
            out.loc[mask, "pred_sd"] = result["calibrated_sd"].values

    # Rows with no bucket label get global calibration
    null_mask = out["total_bucket"].isna()
    if null_mask.any():
        null_df = out.loc[null_mask, ["pred_mean", "pred_sd"]].copy()
        if global_mean_cal is not None:
            result = global_mean_cal.transform(null_df)
            out.loc[null_mask, "pred_mean"] = result["calibrated_mean"].values
            null_df = out.loc[null_mask, ["pred_mean", "pred_sd"]].copy()
        if global_var_cal is not None:
            result = global_var_cal.transform(null_df)
            out.loc[null_mask, "pred_sd"] = result["calibrated_sd"].values

    return out


# ---------------------------------------------------------------------------
# Per-bucket metric breakdown
# ---------------------------------------------------------------------------


def _compute_bucket_metrics(
    df: pd.DataFrame,
    thresholds: Tuple[float, float],
) -> List[BucketArmMetrics]:
    """Compute metrics per total_bucket slice."""
    from calibration.total_bucket_regimes import label_dataframe_with_total_bucket

    labeled = label_dataframe_with_total_bucket(
        df.copy(), total_col="pred_mean", thresholds=thresholds
    )

    results = []
    for bucket_label in ["low", "mid", "high"]:
        bucket_df = labeled[labeled["total_bucket"] == bucket_label]
        if len(bucket_df) < 2:
            continue
        metrics = _compute_arm_metrics(bucket_df)
        results.append(
            BucketArmMetrics(
                bucket=bucket_label,
                metrics=metrics,
                sample_count=len(bucket_df),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Policy gate
# ---------------------------------------------------------------------------


def _evaluate_policy(
    baseline: ArmMetrics,
    treatment: ArmMetrics,
    tolerance_pct: float = 0.05,
) -> PolicyDecision:
    """Evaluate whether bucketed calibration should be recommended.

    Checks:
      1. tail_miss_rate improved (lower)
      2. coverage_2sd improved (higher)
      3. MAE not worsened beyond tolerance
      4. RMSE not worsened beyond tolerance

    All four must pass for "recommended".
    """
    tail_miss_improved = treatment.tail_miss_rate < baseline.tail_miss_rate
    coverage_2sd_improved = treatment.coverage_2sd > baseline.coverage_2sd
    mae_within = treatment.mae <= baseline.mae * (1 + tolerance_pct)
    rmse_within = treatment.rmse <= baseline.rmse * (1 + tolerance_pct)

    reasoning: List[str] = []

    if tail_miss_improved:
        reasoning.append(
            f"tail_miss_rate improved: {baseline.tail_miss_rate:.4f} -> "
            f"{treatment.tail_miss_rate:.4f}"
        )
    else:
        reasoning.append(
            f"tail_miss_rate NOT improved: {baseline.tail_miss_rate:.4f} -> "
            f"{treatment.tail_miss_rate:.4f}"
        )

    if coverage_2sd_improved:
        reasoning.append(
            f"coverage_2sd improved: {baseline.coverage_2sd:.4f} -> "
            f"{treatment.coverage_2sd:.4f}"
        )
    else:
        reasoning.append(
            f"coverage_2sd NOT improved: {baseline.coverage_2sd:.4f} -> "
            f"{treatment.coverage_2sd:.4f}"
        )

    if mae_within:
        reasoning.append(
            f"MAE within tolerance ({tolerance_pct:.0%}): "
            f"{baseline.mae:.4f} -> {treatment.mae:.4f} "
            f"(limit {baseline.mae * (1 + tolerance_pct):.4f})"
        )
    else:
        reasoning.append(
            f"MAE EXCEEDED tolerance ({tolerance_pct:.0%}): "
            f"{baseline.mae:.4f} -> {treatment.mae:.4f} "
            f"(limit {baseline.mae * (1 + tolerance_pct):.4f})"
        )

    if rmse_within:
        reasoning.append(
            f"RMSE within tolerance ({tolerance_pct:.0%}): "
            f"{baseline.rmse:.4f} -> {treatment.rmse:.4f} "
            f"(limit {baseline.rmse * (1 + tolerance_pct):.4f})"
        )
    else:
        reasoning.append(
            f"RMSE EXCEEDED tolerance ({tolerance_pct:.0%}): "
            f"{baseline.rmse:.4f} -> {treatment.rmse:.4f} "
            f"(limit {baseline.rmse * (1 + tolerance_pct):.4f})"
        )

    all_pass = tail_miss_improved and coverage_2sd_improved and mae_within and rmse_within
    recommendation = "recommended" if all_pass else "not_recommended"

    return PolicyDecision(
        recommendation=recommendation,
        reasoning=reasoning,
        tail_miss_improved=tail_miss_improved,
        coverage_2sd_improved=coverage_2sd_improved,
        mae_within_tolerance=mae_within,
        rmse_within_tolerance=rmse_within,
        tolerance_pct=tolerance_pct,
    )


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run_calibration_ab(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    source: str,
    fit_start: str,
    fit_end: str,
    eval_start: str,
    eval_end: str,
    bucket_thresholds: Tuple[float, float] = (210.0, 225.0),
    min_samples_per_bucket: int = 200,
    output_dir: Path | str,
    tolerance_pct: float = 0.05,
    models: List[str] | None = None,
) -> CalibrationABReport:
    """Run A/B evaluation: global vs bucketed TOTAL calibration.

    Orchestrates existing Phase 8/9/10 functions to:
    1. Build calibration datasets for fit and eval windows
    2. Fit global and bucketed calibrators on the fit window
    3. Apply both to the eval window (on deep copies)
    4. Compute Phase 9 metrics on both
    5. Evaluate the policy gate
    6. Write JSON report

    Args:
        db_path: Path to SQLite database
        sport: Sport identifier
        season: Season identifier
        source: Prediction source id (e.g. ensemble_total_v1)
        fit_start: Fit window start date (YYYY-MM-DD)
        fit_end: Fit window end date (YYYY-MM-DD)
        eval_start: Eval window start date (YYYY-MM-DD)
        eval_end: Eval window end date (YYYY-MM-DD)
        bucket_thresholds: (low, mid) cutoffs for total buckets
        min_samples_per_bucket: Minimum samples to fit a bucket calibrator
        output_dir: Directory for report output
        tolerance_pct: MAE/RMSE degradation tolerance (default 5%)
        models: Optional list of TOTAL models; loads from ensemble config if None

    Returns:
        CalibrationABReport with metrics, deltas, and policy recommendation
    """
    from calibration.historical_calibration import (
        load_completed_games,
        generate_model_predictions,
        build_total_calibration_dataset,
        fit_calibrator_for_market,
        fit_total_bucket_calibrators,
    )
    from calibration.mean_calibrator import MeanCalibrator
    from markets.base import Market

    output_dir = Path(output_dir)

    _LOG.info(
        "[calibration_ab] Starting A/B evaluation for %s/%s "
        "(fit: %s..%s, eval: %s..%s)",
        sport, season, fit_start, fit_end, eval_start, eval_end,
    )

    # ---- Step 1: Resolve models from ensemble config if not provided ----
    if models is None:
        from ensemble.config import load_ensemble_config

        ensemble_config = load_ensemble_config(
            sport=sport, season=season, available_models=None,
        )
        market_configs = ensemble_config.get("markets", {})
        total_config = market_configs.get("TOTAL", {})
        models = list(total_config.get("models", []))
        if not models:
            raise ValueError(
                "No TOTAL models configured in ensemble config and --models not provided"
            )
    _LOG.info("[calibration_ab] Using models: %s", models)

    # ---- Step 2: Load games and generate predictions for both windows ----
    fit_games = load_completed_games(
        db_path, sport=sport, season=season,
        start_date=fit_start, end_date=fit_end,
    )
    eval_games = load_completed_games(
        db_path, sport=sport, season=season,
        start_date=eval_start, end_date=eval_end,
    )

    if fit_games.empty:
        raise ValueError(
            f"No completed games in fit window ({fit_start}..{fit_end})"
        )
    if eval_games.empty:
        raise ValueError(
            f"No completed games in eval window ({eval_start}..{eval_end})"
        )

    _LOG.info(
        "[calibration_ab] Games: fit=%d, eval=%d",
        len(fit_games), len(eval_games),
    )

    fit_preds = generate_model_predictions(
        db_path, fit_games,
        sport=sport, season=season, models=models,
        start_date=fit_start, end_date=fit_end,
    )
    eval_preds = generate_model_predictions(
        db_path, eval_games,
        sport=sport, season=season, models=models,
        start_date=eval_start, end_date=eval_end,
    )

    # ---- Step 3: Build calibration datasets ----
    fit_dataset = build_total_calibration_dataset(fit_games, fit_preds)
    eval_dataset = build_total_calibration_dataset(eval_games, eval_preds)

    if fit_dataset.empty:
        raise ValueError("Fit dataset is empty after merging predictions with outcomes")
    if eval_dataset.empty:
        raise ValueError("Eval dataset is empty after merging predictions with outcomes")

    _LOG.info(
        "[calibration_ab] Datasets: fit=%d rows, eval=%d rows",
        len(fit_dataset), len(eval_dataset),
    )

    # ---- Step 4: Fit global arm on fit dataset ----
    global_mean_cal = MeanCalibrator()
    global_mean_cal.fit(fit_dataset)
    _LOG.info(
        "[calibration_ab] Fitted global MeanCalibrator (delta=%.4f)",
        global_mean_cal.delta,
    )

    global_var_cal, global_var_path = fit_calibrator_for_market(
        fit_dataset,
        market=Market.TOTAL,
        sport=sport,
        season=season,
        source_id=source + "_ab_eval_global",
    )
    _LOG.info("[calibration_ab] Fitted global VarianceCalibrator: %s", global_var_path)

    # ---- Step 5: Fit bucketed arm on fit dataset ----
    ab_source_id = source + "_ab_eval"
    bucketed_global_var, bucketed_global_mean, bucket_cals, manifest_path = (
        fit_total_bucket_calibrators(
            fit_dataset,
            sport=sport,
            season=season,
            source_id=ab_source_id,
            low_threshold=bucket_thresholds[0],
            mid_threshold=bucket_thresholds[1],
            min_samples_per_bucket=min_samples_per_bucket,
            start_date=fit_start,
            end_date=fit_end,
        )
    )
    _LOG.info(
        "[calibration_ab] Fitted bucketed arm: %d bucket calibrators, manifest=%s",
        len(bucket_cals), manifest_path,
    )

    # ---- Step 6: Apply both arms to eval dataset (deep copies) ----
    global_eval = _apply_global_arm(eval_dataset, global_mean_cal, global_var_cal)
    bucketed_eval = _apply_bucketed_arm(
        eval_dataset,
        bucketed_global_mean,
        bucketed_global_var,
        bucket_cals,
        bucket_thresholds,
    )

    # ---- Step 7: Compute metrics ----
    baseline_metrics = _compute_arm_metrics(global_eval)
    treatment_metrics = _compute_arm_metrics(bucketed_eval)

    _LOG.info(
        "[calibration_ab] Baseline: MAE=%.4f RMSE=%.4f tail_miss=%.4f cov_2sd=%.4f",
        baseline_metrics.mae, baseline_metrics.rmse,
        baseline_metrics.tail_miss_rate, baseline_metrics.coverage_2sd,
    )
    _LOG.info(
        "[calibration_ab] Treatment: MAE=%.4f RMSE=%.4f tail_miss=%.4f cov_2sd=%.4f",
        treatment_metrics.mae, treatment_metrics.rmse,
        treatment_metrics.tail_miss_rate, treatment_metrics.coverage_2sd,
    )

    # ---- Step 8: Per-bucket metrics ----
    baseline_buckets = _compute_bucket_metrics(global_eval, bucket_thresholds)
    treatment_buckets = _compute_bucket_metrics(bucketed_eval, bucket_thresholds)

    # ---- Step 9: Deltas ----
    deltas = {
        "mae": treatment_metrics.mae - baseline_metrics.mae,
        "rmse": treatment_metrics.rmse - baseline_metrics.rmse,
        "bias": treatment_metrics.bias - baseline_metrics.bias,
        "coverage_1sd": treatment_metrics.coverage_1sd - baseline_metrics.coverage_1sd,
        "coverage_2sd": treatment_metrics.coverage_2sd - baseline_metrics.coverage_2sd,
        "tail_miss_rate": treatment_metrics.tail_miss_rate - baseline_metrics.tail_miss_rate,
    }

    # ---- Step 10: Policy gate ----
    policy = _evaluate_policy(baseline_metrics, treatment_metrics, tolerance_pct)
    _LOG.info("[calibration_ab] Policy: %s", policy.recommendation)

    # ---- Step 11: Build and save report ----
    # Determine which calibrator path to store based on mode:
    # - For global mode: store the global calibrator path
    # - For bucketed mode: store the manifest path
    global_path_to_store = str(global_var_path) if global_var_path else None
    
    report = CalibrationABReport(
        sport=sport,
        season=season,
        market="total",
        source=source,
        fit_window={"start": fit_start, "end": fit_end},
        eval_window={"start": eval_start, "end": eval_end},
        baseline=baseline_metrics,
        treatment=treatment_metrics,
        deltas=deltas,
        baseline_buckets=baseline_buckets,
        treatment_buckets=treatment_buckets,
        policy=policy,
        bucket_thresholds=bucket_thresholds,
        min_samples_per_bucket=min_samples_per_bucket,
        global_calibrator_path=global_path_to_store,
        bucket_manifest_path=str(manifest_path) if manifest_path else None,
    )

    report_path = output_dir / "calibration_ab_report.json"
    report.to_json(report_path)

    return report
