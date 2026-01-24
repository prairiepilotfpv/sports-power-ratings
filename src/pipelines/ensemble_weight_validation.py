"""Optional validation of ensemble weights vs equal weights.

Compares log-loss/MAE of tuned ensemble weights against equal-weight baseline
on completed games by querying historical predictions from database.

**Automatic Workflow:**
1. Schedule is generated -> BETS predictions saved to database
2. Games complete -> Database has actual outcomes
3. Validation queries DB: find predictions + completed games
4. Compares tuned vs equal weights on those games

No manual file handling needed - all data flows through database.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from calibration.eval import log_loss, brier_score
from data.bets_repository import load_bets_predictions_for_validation
from markets.base import Market

_LOG = logging.getLogger(__name__)


def _mae(preds: np.ndarray, targets: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(preds.astype(float) - targets.astype(float))))


def validate_ensemble_ml_weights(
    *,
    db_path: str | Path,
    sport: str,
    season: str,
    market: str = "ML",
    days_back: int = 7,
) -> Dict[str, Any] | None:
    """Compare tuned ML ensemble weights vs equal weights on completed games.
    
    Automatically queries database for:
    - Historical BETS predictions (rolling window: today back N days)
    - Actual game outcomes
    - Compares predictions against results
    
    **Timing:** Predictions are saved when schedule is generated, but validation
    only shows results after games complete. E.g., if you run on Jan 25, you'll
    see validation for Jan 24 predictions (which have now completed).
    
    Args:
        db_path: Path to SQLite database containing games and bets_predictions
        sport: Sport code (e.g., 'nba')
        season: Season string (e.g., '2025-26')
        market: Market type (currently only ML supported)
        days_back: Days back from today to include (default: 7)
    
    Returns:
        Dict with comparison metrics, or None if insufficient data.
    """
    if market != "ML":
        _LOG.debug("Ensemble weight validation only supports ML market")
        return None
    
    # Load predictions from rolling window (yesterday back N days)
    try:
        bets_df = load_bets_predictions_for_validation(
            db_path, sport=sport, season=season, days_back=days_back
        )
    except Exception as e:
        _LOG.debug(f"Failed to load BETS predictions from DB: {e}")
        return None
    
    if bets_df.empty:
        print("  [Validation] No stored BETS predictions found in database")
        print("  [Validation] (Predictions are saved automatically during schedule generation)")
        return None
    
    print(f"  [Validation] Found {len(bets_df)} predictions with completed games")
    
    # Group by prediction_date to show daily performance
    date_groups = bets_df.groupby("prediction_date", sort=True)
    print(f"  [Validation] Covering {len(date_groups)} prediction dates")
    
    # Extract outcomes and probabilities from loaded data
    bets_df["home_win"] = (bets_df["home_score"] > bets_df["away_score"]).astype(int)
    bets_df["tuned_prob"] = pd.to_numeric(bets_df["home_win_prob"], errors="coerce")
    
    # Filter to valid probabilities
    valid_mask = bets_df["tuned_prob"].notna()
    if not valid_mask.any():
        print("  [Validation] No valid probabilities in predictions")
        return None
    
    valid_data = bets_df[valid_mask].copy()
    outcomes = valid_data["home_win"].values
    tuned_probs = valid_data["tuned_prob"].values
    
    # Compute overall losses
    tuned_loss = log_loss(
        pd.Series(tuned_probs, dtype=float),
        pd.Series(outcomes, dtype=float),
    )
    tuned_brier = brier_score(
        pd.Series(tuned_probs, dtype=float),
        pd.Series(outcomes, dtype=float),
    )
    
    # Compute by-date breakdown
    date_results = []
    for pred_date, group in valid_data.groupby("prediction_date"):
        group_outcomes = group["home_win"].values
        group_probs = group["tuned_prob"].values
        if len(group_outcomes) > 0:
            date_loss = log_loss(
                pd.Series(group_probs, dtype=float),
                pd.Series(group_outcomes, dtype=float),
            )
            date_brier = brier_score(
                pd.Series(group_probs, dtype=float),
                pd.Series(group_outcomes, dtype=float),
            )
            date_results.append({
                "prediction_date": pred_date,
                "n_games": len(group),
                "log_loss": float(date_loss),
                "brier_score": float(date_brier),
            })
    
    # Print overall summary
    print("  [Validation] Overall (all dates):")
    print(f"    Tuned ensemble log-loss: {tuned_loss:.4f}")
    print(f"    Tuned ensemble brier-score: {tuned_brier:.4f}")
    
    # Print per-date results
    if date_results:
        print("  [Validation] Per-date breakdown:")
        for dr in date_results:
            print(
                f"    {dr['prediction_date']}: "
                f"{dr['n_games']} games | "
                f"loss={dr['log_loss']:.4f} | "
                f"brier={dr['brier_score']:.4f}"
            )
    
    # Return comprehensive metrics
    return {
        "n_games": len(valid_data),
        "n_prediction_dates": len(date_groups),
        "date_range_start": valid_data["prediction_date"].min(),
        "date_range_end": valid_data["prediction_date"].max(),
        "tuned_log_loss": float(tuned_loss),
        "tuned_brier_score": float(tuned_brier),
        "by_date": date_results,
        "status": "incomplete",
        "reason": "MVP: Only tuned weights; equal-weight baseline comparison in phase 2",
    }


def _compute_equal_weight_probs(
    valid_data: pd.DataFrame,
) -> np.ndarray | None:
    """Reconstruct equal-weight ensemble probabilities from component JSON.
    
    If ml_ensemble_components_json is present, averages individual model probs
    with equal weights.
    """
    if "ml_ensemble_components_json" not in valid_data.columns:
        return None
    
    equal_probs_list = []
    for _, row in valid_data.iterrows():
        components_json = row.get("ml_ensemble_components_json")
        if not components_json or pd.isna(components_json):
            return None  # Can't reconstruct if missing
        
        try:
            if isinstance(components_json, str):
                components = json.loads(components_json)
            else:
                components = components_json
            
            # Extract per-model probabilities
            model_probs = []
            if isinstance(components, list):
                for comp in components:
                    if "prob" in comp:
                        model_probs.append(float(comp["prob"]))
            elif isinstance(components, dict) and "models" in components:
                for model_name, model_data in components["models"].items():
                    if "prob" in model_data:
                        model_probs.append(float(model_data["prob"]))
            
            if model_probs:
                equal_prob = np.mean(model_probs)
                equal_probs_list.append(equal_prob)
            else:
                return None
        except Exception as e:
            _LOG.debug(f"Failed to parse component JSON: {e}")
            return None
    
    return np.array(equal_probs_list) if equal_probs_list else None


def save_validation_report(
    *,
    validation_result: Dict[str, Any],
    sport: str,
    season: str,
    output_dir: Path | None = None,
) -> Path | None:
    """Save validation results to JSON report.
    
    Args:
        validation_result: Dict from validate_ensemble_ml_weights()
        sport: Sport identifier
        season: Season identifier
        output_dir: Optional output directory (default: outputs/ensemble_validation/)
    
    Returns:
        Path to report file, or None if failed to save.
    """
    if validation_result is None:
        return None
    
    out_dir = output_dir or Path("outputs") / "ensemble_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{sport}_{season}_{timestamp}_ml_weights_comparison.json"
    
    try:
        out_path.write_text(
            json.dumps({
                "sport": sport,
                "season": season,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                **validation_result,
            }, indent=2),
            encoding="utf-8",
        )
        _LOG.info(f"Saved ensemble weight validation report to {out_path}")
        return out_path
    except Exception as e:
        _LOG.error(f"Failed to save validation report: {e}")
        return None
