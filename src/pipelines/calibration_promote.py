"""Phase 13: Promotion and health monitoring for TOTAL calibration policies.

Thin orchestration layer that:
1. Promotes a policy from a Phase 11 A/B report
2. Evaluates policy health on recent games (advisory only)

Does NOT:
- Refit any calibrators
- Rerun evaluation
- Modify existing artifacts
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Promotion from A/B report
# ---------------------------------------------------------------------------


def load_calibration_ab_report(report_path: Path | str) -> Dict[str, Any]:
    """Load Phase 11 calibration A/B report.

    Args:
        report_path: Path to calibration_ab_report.json

    Returns:
        Parsed JSON dict

    Raises:
        FileNotFoundError: If report not found
        ValueError: If report JSON is invalid
    """
    report_path = Path(report_path)
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    try:
        with open(report_path, "r") as f:
            report = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in report {report_path}: {e}")

    return report


def validate_ab_report_for_promotion(report: Dict[str, Any]) -> None:
    """Validate that A/B report has required fields for promotion.

    Checks:
    - report.policy.recommendation exists
    - report.global_calibrator_path exists
    - report.bucket_manifest_path exists

    Args:
        report: Parsed A/B report dict

    Raises:
        ValueError: If required fields missing or malformed
    """
    if not isinstance(report, dict):
        raise ValueError("Report must be a dict")

    # Check policy section
    policy = report.get("policy")
    if not policy or not isinstance(policy, dict):
        raise ValueError("report.policy must be a dict")

    recommendation = policy.get("recommendation")
    if recommendation not in ("recommended", "not_recommended"):
        raise ValueError(
            f"report.policy.recommendation must be 'recommended' or 'not_recommended', "
            f"got '{recommendation}'"
        )

    # Check that at least one calibrator path is provided
    global_path = report.get("global_calibrator_path")
    bucket_path = report.get("bucket_manifest_path")

    if not global_path and not bucket_path:
        raise ValueError(
            "Report must contain either global_calibrator_path or bucket_manifest_path"
        )


def promote_policy_from_report(
    sport: str,
    season: str,
    report: Dict[str, Any],
    notes: str = "",
) -> Path:
    """Promote a policy from a Phase 11 A/B report to active.

    Logic:
    1. Validate the report
    2. Check report.policy.recommendation
    3. If "recommended":
       - If bucket_manifest_path exists: promote total_bucket
       - Else: promote global
    4. If "not_recommended": abort with clear message
    5. Save active.json using Phase 12 helpers

    Args:
        sport: Sport code
        season: Season identifier
        report: Parsed A/B report dict
        notes: Optional notes to include in policy

    Returns:
        Path to saved active.json

    Raises:
        ValueError: If report is invalid or recommendation is "not_recommended"
    """
    from calibration.active_policy import (
        create_global_policy,
        create_total_bucket_policy,
        save_total_active_policy,
    )

    # ---- Validation ----
    validate_ab_report_for_promotion(report)

    # ---- Check recommendation ----
    recommendation = report["policy"]["recommendation"]
    if recommendation != "recommended":
        reasoning = report["policy"].get("reasoning", [])
        reasoning_str = "\n".join(f"  - {r}" for r in reasoning)
        raise ValueError(
            f"Promotion blocked: policy.recommendation is '{recommendation}'\n"
            f"Reasoning:\n{reasoning_str}"
        )

    _LOG.info(
        "[promote_policy_from_report] Promoting policy for %s/%s "
        "(recommendation=%s)",
        sport, season, recommendation,
    )

    # ---- Determine mode and create policy ----
    bucket_manifest_path = report.get("bucket_manifest_path")
    global_calibrator_path = report.get("global_calibrator_path")

    if bucket_manifest_path:
        _LOG.info(
            "[promote_policy_from_report] Promoting total_bucket mode "
            "(manifest=%s)",
            bucket_manifest_path,
        )
        policy = create_total_bucket_policy(
            sport=sport,
            season=season,
            manifest_path=bucket_manifest_path,
            global_path=global_calibrator_path,
            notes=notes,
        )
    else:
        _LOG.info(
            "[promote_policy_from_report] Promoting global mode "
            "(path=%s)",
            global_calibrator_path,
        )
        policy = create_global_policy(
            sport=sport,
            season=season,
            global_path=global_calibrator_path,
            notes=notes,
        )

    # ---- Save policy ----
    policy_path = save_total_active_policy(sport, season, policy)
    _LOG.info("[promote_policy_from_report] Saved policy to %s", policy_path)

    return policy_path


# ---------------------------------------------------------------------------
# Policy health check
# ---------------------------------------------------------------------------


def evaluate_policy_health(
    db_path: str | Path,
    sport: str,
    season: str,
    market: str = "total",
    window_games: int = 100,
) -> Dict[str, Any]:
    """Evaluate recent performance of active TOTAL policy (advisory only).

    Logic:
    1. Load active policy
    2. Build dataset on most recent games
    3. Apply policy calibration
    4. Compute Phase 9 metrics
    5. Compare against Phase 11 tolerances
    6. Return summary (does NOT write files or modify anything)

    Args:
        db_path: Path to SQLite database
        sport: Sport code
        season: Season identifier
        market: Market to evaluate (default: total)
        window_games: Number of recent games to evaluate

    Returns:
        Health report dict with keys:
        - status: "OK" or "WARN"
        - current_metrics: dict of computed metrics
        - warnings: list of degradation warnings (empty if OK)
        - notes: additional context
    """
    from calibration.active_policy import load_total_active_policy
    from calibration.historical_calibration import (
        load_completed_games,
        generate_model_predictions,
        build_total_calibration_dataset,
    )
    from pipelines.calibration_evaluation import (
        mean_absolute_error,
        root_mean_squared_error,
        empirical_coverage,
        tail_miss_rate,
    )

    if market != "total":
        raise ValueError(f"Policy health only supports market='total', got '{market}'")

    _LOG.info(
        "[evaluate_policy_health] Checking policy health for %s/%s "
        "(window=%d games)",
        sport, season, window_games,
    )

    # ---- Step 1: Load policy ----
    policy = load_total_active_policy(sport, season)
    if not policy:
        return {
            "status": "WARN",
            "current_metrics": {},
            "warnings": ["No active policy found for this sport/season"],
            "notes": "Cannot evaluate health without active policy",
        }

    _LOG.info("[evaluate_policy_health] Loaded policy: mode=%s", policy.get("mode"))

    # ---- Step 2: Load recent games ----
    try:
        games = load_completed_games(db_path, sport=sport, season=season)
        if games.empty:
            return {
                "status": "WARN",
                "current_metrics": {},
                "warnings": ["No completed games found"],
                "notes": "Cannot evaluate on empty dataset",
            }

        # Get most recent games
        games = games.sort_values("date").tail(window_games)
        _LOG.info("[evaluate_policy_health] Loaded %d recent games", len(games))

    except Exception as e:
        _LOG.warning("[evaluate_policy_health] Failed to load games: %s", e)
        return {
            "status": "WARN",
            "current_metrics": {},
            "warnings": [f"Failed to load games: {e}"],
            "notes": "Cannot evaluate health with load error",
        }

    # ---- Step 3: Generate predictions for TOTAL market ----
    try:
        # Get ensemble models from config
        from ensemble.config import load_ensemble_config

        ensemble_config = load_ensemble_config(
            sport=sport, season=season, available_models=None
        )
        market_config = ensemble_config.get("markets", {}).get("TOTAL", {})
        models = list(market_config.get("models", []))

        if not models:
            _LOG.warning("[evaluate_policy_health] No TOTAL models in ensemble config")
            return {
                "status": "WARN",
                "current_metrics": {},
                "warnings": ["No TOTAL models configured"],
                "notes": "Cannot generate predictions without models",
            }

        preds = generate_model_predictions(
            db_path, games,
            sport=sport, season=season, models=models,
        )
        if preds.empty:
            return {
                "status": "WARN",
                "current_metrics": {},
                "warnings": ["No predictions generated"],
                "notes": "Predictions are empty",
            }

        eval_dataset = build_total_calibration_dataset(games, preds)
        if eval_dataset.empty:
            return {
                "status": "WARN",
                "current_metrics": {},
                "warnings": ["Evaluation dataset is empty"],
                "notes": "Cannot build calibration dataset",
            }

        _LOG.info("[evaluate_policy_health] Built eval dataset: %d rows", len(eval_dataset))

    except Exception as e:
        _LOG.warning("[evaluate_policy_health] Failed to build dataset: %s", e)
        return {
            "status": "WARN",
            "current_metrics": {},
            "warnings": [f"Failed to build dataset: {e}"],
            "notes": "Dataset build error",
        }

    # ---- Step 4: Apply policy calibration ----
    try:
        calibrated_df = _apply_policy_to_dataset(eval_dataset, policy, db_path)
        if calibrated_df is None:
            return {
                "status": "WARN",
                "current_metrics": {},
                "warnings": ["Failed to apply policy calibration"],
                "notes": "Policy application failed",
            }
    except Exception as e:
        _LOG.warning("[evaluate_policy_health] Failed to apply policy: %s", e)
        return {
            "status": "WARN",
            "current_metrics": {},
            "warnings": [f"Failed to apply policy: {e}"],
            "notes": "Policy application error",
        }

    # ---- Step 5: Compute metrics ----
    try:
        pred = calibrated_df["pred_mean"]
        sd = calibrated_df["pred_sd"]
        actual = calibrated_df["actual_value"]

        metrics = {
            "mae": mean_absolute_error(pred, actual),
            "rmse": root_mean_squared_error(pred, actual),
            "coverage_1sd": empirical_coverage(pred, sd, actual, sigma=1.0),
            "coverage_2sd": empirical_coverage(pred, sd, actual, sigma=2.0),
            "tail_miss_rate": tail_miss_rate(pred, sd, actual),
            "sample_count": len(calibrated_df),
        }
        _LOG.info(
            "[evaluate_policy_health] Metrics: MAE=%.4f RMSE=%.4f "
            "tail_miss=%.4f cov_2sd=%.4f",
            metrics["mae"], metrics["rmse"],
            metrics["tail_miss_rate"], metrics["coverage_2sd"],
        )

    except Exception as e:
        _LOG.warning("[evaluate_policy_health] Failed to compute metrics: %s", e)
        return {
            "status": "WARN",
            "current_metrics": {},
            "warnings": [f"Failed to compute metrics: {e}"],
            "notes": "Metric computation error",
        }

    # ---- Step 6: Assess health (advisory) ----
    warnings = []

    # Default thresholds (can be customized via config)
    mae_threshold = 5.0  # Absolute threshold in points
    rmse_threshold = 7.0  # Absolute threshold in points
    coverage_2sd_threshold = 0.94  # Should be >= 94%
    tail_miss_threshold = 0.10  # Should be <= 10%

    if metrics["mae"] > mae_threshold:
        warnings.append(
            f"MAE degradation: {metrics['mae']:.2f} (threshold: {mae_threshold})"
        )

    if metrics["rmse"] > rmse_threshold:
        warnings.append(
            f"RMSE degradation: {metrics['rmse']:.2f} (threshold: {rmse_threshold})"
        )

    if metrics["coverage_2sd"] < coverage_2sd_threshold:
        warnings.append(
            f"Coverage 2-sigma low: {metrics['coverage_2sd']:.2%} "
            f"(threshold: {coverage_2sd_threshold:.2%})"
        )

    if metrics["tail_miss_rate"] > tail_miss_threshold:
        warnings.append(
            f"Tail miss rate high: {metrics['tail_miss_rate']:.2%} "
            f"(threshold: {tail_miss_threshold:.2%})"
        )

    status = "OK" if not warnings else "WARN"

    return {
        "status": status,
        "current_metrics": metrics,
        "warnings": warnings,
        "notes": (
            f"Evaluated {len(calibrated_df)} recent games. "
            f"Policy mode: {policy.get('mode')}. "
            f"No automatic action taken."
        ),
    }


def _apply_policy_to_dataset(
    eval_df: pd.DataFrame,
    policy: Dict[str, Any],
    db_path: str | Path,
) -> Optional[pd.DataFrame]:
    """Apply active policy calibration to a dataset (deep copy).

    Args:
        eval_df: Evaluation dataset with pred_mean, pred_sd columns
        policy: Active policy dict (from load_total_active_policy)
        db_path: Path to database (for loading calibrators if needed)

    Returns:
        Calibrated copy of eval_df, or None if application failed
    """
    from calibration.io import load_calibrator
    from calibration.total_bucket_regimes import label_dataframe_with_total_bucket

    out = eval_df.copy(deep=True)
    mode = policy.get("mode")

    try:
        if mode == "global":
            # Load and apply global calibrator
            global_cfg = policy.get("global", {})
            global_path = global_cfg.get("path")
            if not global_path:
                _LOG.warning("[_apply_policy_to_dataset] No global path in policy")
                return None

            calibrator = load_calibrator(global_path)
            result = calibrator.transform(out[["pred_mean", "pred_sd"]])
            out["pred_mean"] = result.get("calibrated_mean", out["pred_mean"])
            if "calibrated_sd" in result.columns:
                out["pred_sd"] = result["calibrated_sd"]

            return out

        elif mode == "total_bucket":
            # Load manifest and apply bucket calibrators
            bucket_cfg = policy.get("total_bucket", {})
            manifest_path = bucket_cfg.get("manifest_path")
            if not manifest_path:
                _LOG.warning("[_apply_policy_to_dataset] No manifest path in policy")
                return None

            # Load manifest
            manifest_path = Path(manifest_path)
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            # Label dataset with buckets
            bucket_thresholds = manifest.get("bucket_thresholds", [210.0, 225.0])
            out = label_dataframe_with_total_bucket(
                out, total_col="pred_mean", thresholds=tuple(bucket_thresholds)
            )

            # Apply bucket calibrators with fallback to global
            bucket_calibrators = manifest.get("calibrators_bucket_mean", {})
            global_mean_path = manifest.get("calibrator_global_mean")

            for bucket_label in ["low", "mid", "high"]:
                mask = out["total_bucket"] == bucket_label
                if not mask.any():
                    continue

                mean_key = f"calibrators_bucket_mean.{bucket_label}"
                cal_path = manifest.get(mean_key, global_mean_path)

                if cal_path:
                    calibrator = load_calibrator(cal_path)
                    bucket_df = out.loc[mask, ["pred_mean", "pred_sd"]].copy()
                    result = calibrator.transform(bucket_df)
                    out.loc[mask, "pred_mean"] = result.get("calibrated_mean", out.loc[mask, "pred_mean"])

            return out

        else:
            _LOG.warning("[_apply_policy_to_dataset] Unknown mode: %s", mode)
            return None

    except Exception as e:
        _LOG.warning("[_apply_policy_to_dataset] Error applying policy: %s", e)
        return None
