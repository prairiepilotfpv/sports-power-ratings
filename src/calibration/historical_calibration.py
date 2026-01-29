"""Standalone calibration engine for historical model predictions.

This module is completely independent from the betting pipeline. It:
1. Loads completed games from the database (any sport)
2. Generates predictions from models for those games
3. Computes actual outcomes
4. Fits calibrators specific to each market (ML, SPREAD, TOTAL)
5. Persists calibrators for later use

Key design:
- No references to bets_predictions, betting schema, or betting workflow
- Works with raw games table + model predictions
- Market-agnostic: supports ML, SPREAD, TOTAL
- Sport-agnostic: works with any sport in the database
- Distribution-aware: SPREAD/TOTAL use distribution calibrators
"""

from __future__ import annotations

import logging
from contextlib import closing
from pathlib import Path
from typing import Any

import pandas as pd
import sqlite3

from calibration.distribution import VarianceCalibrator
from markets.base import Market
from markets.registry import get_market_spec
from calibration.selection import select_calibrator
from ensemble.config import load_ensemble_config
from ensemble.ml_v1 import MLWeightedAverageEnsemble
from ensemble.spread_v1 import SpreadWeightedAverageEnsemble
from ensemble.total_v1 import TotalWeightedAverageEnsemble
from config import (
    MARGIN_SD_GUARDRAIL_MIN,
    MARGIN_SD_GUARDRAIL_MAX,
    TOTAL_SD_GUARDRAIL_MIN,
    TOTAL_SD_GUARDRAIL_MAX,
)

_LOG = logging.getLogger(__name__)


def _market_guardrails(market: Market) -> tuple[float | None, float | None]:
    if market == Market.SPREAD:
        return MARGIN_SD_GUARDRAIL_MIN, MARGIN_SD_GUARDRAIL_MAX
    if market == Market.TOTAL:
        return TOTAL_SD_GUARDRAIL_MIN, TOTAL_SD_GUARDRAIL_MAX
    return None, None


def _log_calibration_window(
    games_df: pd.DataFrame,
    *,
    market: str,
    window_type: str = "expanding",
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    """Log calibration window details for transparency.
    
    Args:
        games_df: DataFrame of games used for calibration
        market: Market name (ML, SPREAD, TOTAL)
        window_type: "expanding" (all data) or "rolling" (recent data)
        start_date: Optional start date cutoff (YYYY-MM-DD)
        end_date: Optional end date cutoff (YYYY-MM-DD)
    """
    game_count = len(games_df)
    
    # Extract date range from games if present
    actual_start = None
    actual_end = None
    if "date" in games_df.columns:
        try:
            dates = pd.to_datetime(games_df["date"], errors="coerce")
            actual_start = dates.min().date().isoformat() if dates.notna().any() else None
            actual_end = dates.max().date().isoformat() if dates.notna().any() else None
        except Exception:
            pass
    
    _LOG.info(
        "[calibration window] market=%s mode=%s lookback=%s start=%s end=%s "
        "actual_start=%s actual_end=%s games=%d",
        market,
        window_type,
        "N/A",  # Lookback would be computed from window_type and dates
        start_date or "none",
        end_date or "none",
        actual_start or "unknown",
        actual_end or "unknown",
        game_count,
    )


def _market_guardrails(market: Market) -> tuple[float | None, float | None]:
    if market == Market.SPREAD:
        return MARGIN_SD_GUARDRAIL_MIN, MARGIN_SD_GUARDRAIL_MAX
    if market == Market.TOTAL:
        return TOTAL_SD_GUARDRAIL_MIN, TOTAL_SD_GUARDRAIL_MAX
    return None, None


def load_completed_games(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load all completed games for calibration.

    Args:
        db_path: Path to database
        sport: Sport code (e.g., 'nba', 'nfl', 'mlb')
        season: Season identifier (e.g., '2025-26')
        start_date: Optional YYYY-MM-DD start filter
        end_date: Optional YYYY-MM-DD end filter

    Returns:
        DataFrame with game_id, date, home_team, away_team, home_score, away_score
    """
    with closing(sqlite3.connect(str(db_path))) as conn:
        query = """
            SELECT game_id, date, home_team, away_team, home_score, away_score
            FROM games
            WHERE sport = ? AND season = ?
              AND home_score IS NOT NULL AND away_score IS NOT NULL
        """
        params: list[Any] = [sport, season]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date ASC"
        games_df = pd.read_sql(query, conn, params=params)

    _LOG.info(
        f"[load_completed_games] Loaded {len(games_df)} completed games "
        f"for {sport}/{season}"
    )
    return games_df


def generate_model_predictions(
    db_path: str | Path,
    games_df: pd.DataFrame,
    *,
    sport: str,
    season: str,
    models: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Generate walk-forward backtest predictions for calibration.

    This runs the backtest engine per model to avoid data leakage and returns
    per-game predictions that can be calibrated across ML/SPREAD/TOTAL markets.

    Args:
        db_path: Path to database (for context/logging only)
        games_df: DataFrame of completed games to backtest
        sport: Sport code
        season: Season identifier
        models: List of model names to ensemble
        start_date: Optional evaluation start date (YYYY-MM-DD)
        end_date: Optional evaluation end date (YYYY-MM-DD)

    Returns:
        DataFrame with per-game predictions (one row per game, ensemble-averaged)
    """
    from backtest.runner import run_backtest
    from models.registry import get_backtest_model
    import tempfile

    if games_df.empty:
        return pd.DataFrame()

    _LOG.info(
        f"[generate_model_predictions] Running backtests for {len(models)} model(s) "
        f"on {len(games_df)} games"
    )

    # Ensure required columns are present for backtest models.
    games = games_df.copy()
    if "sport" not in games.columns:
        games["sport"] = sport
    if "season" not in games.columns:
        games["season"] = season

    if start_date is None and "date" in games.columns:
        try:
            start_date = pd.to_datetime(games["date"], errors="coerce").min().date().isoformat()
        except Exception:
            start_date = None
    if end_date is None and "date" in games.columns:
        try:
            end_date = pd.to_datetime(games["date"], errors="coerce").max().date().isoformat()
        except Exception:
            end_date = None

    all_frames: list[pd.DataFrame] = []

    for model_name in models:
        try:
            model_cls = get_backtest_model(model_name)
        except Exception as e:
            _LOG.warning(f"[generate_model_predictions] Unknown model {model_name}: {e}")
            continue

        _LOG.info(f"[generate_model_predictions] Backtesting {model_name}")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                outputs = run_backtest(
                    model_factory=lambda: model_cls(),
                    games_df=games,
                    start_date=start_date,
                    end_date=end_date,
                    output_dir=Path(tmpdir),
                    model_name=model_name,
                    db_path=None,
                    sport=sport,
                    season=season,
                    calibrate=False,
                )
            preds = outputs.predictions.copy()
        except Exception as e:
            _LOG.warning(f"[generate_model_predictions] Backtest failed for {model_name}: {e}")
            continue

        if preds.empty:
            _LOG.warning(f"[generate_model_predictions] No predictions for {model_name}")
            continue

        preds["model_name"] = model_name
        for col in (
            "p_home_win",
            "margin_mean",
            "margin_sd",
            "total_mean",
            "total_sd",
            "game_id",
        ):
            if col not in preds.columns:
                preds[col] = pd.NA
        all_frames.append(
            preds[
                [
                    "game_id",
                    "model_name",
                    "p_home_win",
                    "margin_mean",
                    "margin_sd",
                    "total_mean",
                    "total_sd",
                ]
            ]
        )

    if not all_frames:
        _LOG.warning("[generate_model_predictions] No prediction frames produced")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    _LOG.info(
        "[generate_model_predictions] Generated per-model predictions for %d rows",
        len(combined),
    )
    return combined


def _ensemble_predictions_for_market(
    preds_df: pd.DataFrame,
    *,
    market: Market,
    sport: str,
    season: str,
    ensemble_id: str,
    models: list[str],
    weights: dict[str, float] | None,
) -> pd.DataFrame:
    if preds_df.empty or not models:
        return pd.DataFrame()

    df = preds_df.copy()
    if "model_name" not in df.columns:
        return pd.DataFrame()
    df = df[df["model_name"].isin(set(models))]
    if df.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | str | None]] = []
    if market == Market.ML:
        ensemble = MLWeightedAverageEnsemble(
            sport=sport, season=season, ensemble_id=ensemble_id, weights=weights
        )
        for game_id, group in df.groupby("game_id"):
            p, _components = ensemble.combine(group[["model_name", "p_home_win"]])
            rows.append({"game_id": game_id, "p_home_win": p})
        return pd.DataFrame(rows)

    if market == Market.SPREAD:
        ensemble = SpreadWeightedAverageEnsemble(
            sport=sport, season=season, ensemble_id=ensemble_id, weights=weights
        )
        for game_id, group in df.groupby("game_id"):
            mean, sd, _components = ensemble.combine(
                group[["model_name", "margin_mean", "margin_sd"]]
            )
            rows.append({"game_id": game_id, "margin_mean": mean, "margin_sd": sd})
        return pd.DataFrame(rows)

    if market == Market.TOTAL:
        ensemble = TotalWeightedAverageEnsemble(
            sport=sport, season=season, ensemble_id=ensemble_id, weights=weights
        )
        for game_id, group in df.groupby("game_id"):
            mean, sd, _components = ensemble.combine(
                group[["model_name", "total_mean", "total_sd"]]
            )
            rows.append({"game_id": game_id, "total_mean": mean, "total_sd": sd})
        return pd.DataFrame(rows)

    return pd.DataFrame()


def build_ml_calibration_dataset(
    games_df: pd.DataFrame, predictions_df: pd.DataFrame
) -> pd.DataFrame:
    """Build calibration dataset for ML market.

    Args:
        games_df: DataFrame with game outcomes
        predictions_df: DataFrame with ML predictions (p_home_win column)

    Returns:
        DataFrame with columns: p_home_win, home_win (actual outcome)
    """
    if predictions_df.empty:
        return pd.DataFrame(columns=["p_home_win", "home_win"])

    # Merge predictions with outcomes
    merged = predictions_df.merge(
        games_df[["game_id", "home_score", "away_score"]], on="game_id", how="inner"
    )

    if merged.empty:
        return pd.DataFrame(columns=["p_home_win", "home_win"])

    # Compute outcome: 1 if home team won, 0 otherwise
    records = []
    for _, row in merged.iterrows():
        p = row.get("p_home_win")
        if p is None or (isinstance(p, float) and pd.isna(p)):
            continue
        home_score = row.get("home_score")
        away_score = row.get("away_score")
        if home_score is None or away_score is None:
            continue
        outcome = 1 if home_score > away_score else 0
        records.append({"p_home_win": p, "home_win": outcome})

    _LOG.info(f"[build_ml_calibration_dataset] Built dataset with {len(records)} records")
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["p_home_win", "home_win"])


def build_spread_calibration_dataset(
    games_df: pd.DataFrame, predictions_df: pd.DataFrame
) -> pd.DataFrame:
    """Build calibration dataset for SPREAD market.

    SPREAD predictions include distribution parameters (margin_mean, margin_sd).
    We calibrate these distributions directly to match actual margins.

    Args:
        games_df: DataFrame with game outcomes
        predictions_df: DataFrame with SPREAD predictions (margin_mean, margin_sd)

    Returns:
        DataFrame with columns: pred_mean, pred_sd, actual_value
    """
    if predictions_df.empty:
        return pd.DataFrame(columns=["pred_mean", "pred_sd", "actual_value"])

    # Merge predictions with outcomes
    merged = predictions_df.merge(
        games_df[["game_id", "home_score", "away_score"]], on="game_id", how="inner"
    )

    if merged.empty:
        return pd.DataFrame(columns=["pred_mean", "pred_sd", "actual_value"])

    # Compute actual margin
    records = []
    for _, row in merged.iterrows():
        pred_mean = row.get("margin_mean")
        pred_sd = row.get("margin_sd")
        if pred_mean is None or pred_sd is None or pred_sd <= 0:
            continue
        home_score = row.get("home_score")
        away_score = row.get("away_score")
        if home_score is None or away_score is None:
            continue
        actual_margin = home_score - away_score
        records.append({
            "pred_mean": pred_mean,
            "pred_sd": pred_sd,
            "actual_value": actual_margin,
        })

    _LOG.info(f"[build_spread_calibration_dataset] Built dataset with {len(records)} records")
    return (
        pd.DataFrame(records)
        if records
        else pd.DataFrame(columns=["pred_mean", "pred_sd", "actual_value"])
    )


def build_total_calibration_dataset(
    games_df: pd.DataFrame, predictions_df: pd.DataFrame
) -> pd.DataFrame:
    """Build calibration dataset for TOTAL market.

    TOTAL predictions include distribution parameters (total_mean, total_sd).
    We calibrate these distributions directly to match actual totals.

    Args:
        games_df: DataFrame with game outcomes
        predictions_df: DataFrame with TOTAL predictions (total_mean, total_sd)

    Returns:
        DataFrame with columns: pred_mean, pred_sd, actual_value
    """
    if predictions_df.empty:
        return pd.DataFrame(columns=["pred_mean", "pred_sd", "actual_value"])

    # Merge predictions with outcomes
    merged = predictions_df.merge(
        games_df[["game_id", "home_score", "away_score"]], on="game_id", how="inner"
    )

    if merged.empty:
        return pd.DataFrame(columns=["pred_mean", "pred_sd", "actual_value"])

    # Compute actual total
    records = []
    for _, row in merged.iterrows():
        pred_mean = row.get("total_mean")
        pred_sd = row.get("total_sd")
        if pred_mean is None or pred_sd is None or pred_sd <= 0:
            continue
        home_score = row.get("home_score")
        away_score = row.get("away_score")
        if home_score is None or away_score is None:
            continue
        actual_total = home_score + away_score
        records.append({
            "pred_mean": pred_mean,
            "pred_sd": pred_sd,
            "actual_value": actual_total,
        })

    _LOG.info(f"[build_total_calibration_dataset] Built dataset with {len(records)} records")
    return (
        pd.DataFrame(records)
        if records
        else pd.DataFrame(columns=["pred_mean", "pred_sd", "actual_value"])
    )


def fit_calibrator_for_market(
    dataset_df: pd.DataFrame,
    *,
    market: str,
    method: str = "auto",
    sport: str | None = None,
    season: str | None = None,
    source_id: str | None = None,
) -> tuple[Any, Path | None]:
    """Fit and save a calibrator for a specific market.

    Args:
        dataset_df: Calibration data (format depends on market)
        market: Market type ("ML", "spread", or "total")
        method: Calibration method ("auto", "isotonic", "platt", etc.)
        sport: Sport code (for saving artifact)
        season: Season (for saving artifact)
        source_id: Source identifier (for saving artifact)

    Returns:
        Tuple of (calibrator_instance, saved_path) or (calibrator_instance, None) if not saved
    """
    if dataset_df.empty:
        raise ValueError(f"Cannot fit calibrator for {market}: empty dataset")

    _LOG.info(
        f"[fit_calibrator_for_market] Fitting {market} calibrator "
        f"with {len(dataset_df)} records, method={method}"
    )

    # Select appropriate calibrator based on market type
    if market == Market.ML:
        # ML uses probability calibrators
        calibrator = select_calibrator(method, len(dataset_df))
        calibrator.fit(dataset_df)
    elif market in {Market.SPREAD, Market.TOTAL}:
        # SPREAD/TOTAL use variance-based distribution calibrators
        guardrail_min, guardrail_max = _market_guardrails(market)
        calibrator = VarianceCalibrator(
            guardrail_min=guardrail_min,
            guardrail_max=guardrail_max,
        )
        calibrator.fit(dataset_df)
    else:
        raise ValueError(f"Unknown market: {market}")

    _LOG.info(
        f"[fit_calibrator_for_market] Fitted {market.name} calibrator, "
        f"metadata={calibrator.metadata}"
    )

    # Save if all metadata provided
    saved_path = None
    if sport and season and source_id:
        try:
            from calibration.io import save_calibrator

            spec = get_market_spec(market)
            calib_dir = spec.calibrator_dir(sport, season, source_id)
            calib_dir.mkdir(parents=True, exist_ok=True)
            meta = dict(getattr(calibrator, "metadata", {}) or {})
            meta.update({"sport": sport, "season": season, "source_id": source_id})
            saved_path = save_calibrator(
                calibrator,
                calib_dir,
                market=market.value,
                metadata=meta,
            )
            _LOG.info(f"[fit_calibrator_for_market] Saved calibrator to {saved_path}")
        except Exception as e:
            _LOG.warning(f"[fit_calibrator_for_market] Failed to save calibrator: {e}")

    return calibrator, saved_path


def fit_total_bucket_calibrators(
    dataset_df: pd.DataFrame,
    *,
    sport: str,
    season: str,
    source_id: str = "historical",
    method: str = "auto",
    low_threshold: float = 210.0,
    mid_threshold: float = 225.0,
    min_samples_per_bucket: int = 200,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[Any, Any, dict[str, Any], Path | None]:
    """Fit global + per-bucket TOTAL calibrators with regime labeling.
    
    Phase 10 implementation: fits separate mean + variance calibrators per total_bucket,
    with fallback to global when bucket samples are insufficient.
    
    Args:
        dataset_df: Calibration dataset with pred_mean, pred_sd, actual_value
        sport: Sport code
        season: Season identifier
        source_id: Source identifier for artifacts
        method: Calibration method ("auto", "isotonic", "platt")
        low_threshold: Cutoff for low bucket (default 210)
        mid_threshold: Cutoff for mid/high boundary (default 225)
        min_samples_per_bucket: Minimum samples to fit bucket calibrator (default 200)
        start_date: Optional fit window start
        end_date: Optional fit window end
    
    Returns:
        Tuple of (global_calibrator, mean_calibrator, bucket_calibrators_dict, manifest_path)
        bucket_calibrators_dict keys: 'low_mean', 'low_variance', 'mid_mean', 'mid_variance', etc.
    """
    from calibration.total_bucket_regimes import (
        label_dataframe_with_total_bucket,
        TotalBucketManifest,
        save_total_bucket_manifest,
    )
    from calibration.mean_calibrator import MeanCalibrator
    from calibration.io import save_calibrator
    from markets.registry import get_market_spec
    
    _LOG.info(
        "[fit_total_bucket_calibrators] Starting regime-conditioned TOTAL calibration "
        "for %s/%s (thresholds: %.0f/%.0f, min_samples: %d)",
        sport, season, low_threshold, mid_threshold, min_samples_per_bucket,
    )
    
    if dataset_df.empty:
        raise ValueError("Cannot fit calibrators: empty dataset")
    
    # Label each row with its total_bucket
    df = label_dataframe_with_total_bucket(
        dataset_df.copy(),
        total_col="pred_mean",
        thresholds=(low_threshold, mid_threshold),
    )
    
    # Count samples per bucket
    bucket_counts = df["total_bucket"].value_counts().to_dict()
    sample_counts = {
        "global": len(df),
        "low": bucket_counts.get("low", 0),
        "mid": bucket_counts.get("mid", 0),
        "high": bucket_counts.get("high", 0),
    }
    _LOG.info(
        "[fit_total_bucket_calibrators] Sample distribution: "
        "global=%d, low=%d, mid=%d, high=%d",
        sample_counts["global"],
        sample_counts["low"],
        sample_counts["mid"],
        sample_counts["high"],
    )
    
    # ===== FIT GLOBAL CALIBRATORS =====
    global_mean_calibrator = None
    global_variance_calibrator = None
    
    try:
        # Mean calibrator (global)
        mean_calib, _ = fit_calibrator_for_market(
            df,
            market=Market.TOTAL,
            method=method,
            sport=None,
            season=None,
            source_id=None,
        )
        if isinstance(mean_calib, MeanCalibrator):
            global_mean_calibrator = mean_calib
            _LOG.info("[fit_total_bucket_calibrators] Fitted global mean calibrator")
    except Exception as e:
        _LOG.warning("[fit_total_bucket_calibrators] Failed to fit global mean: %s", e)
    
    try:
        # Variance calibrator (global)
        variance_calib, _ = fit_calibrator_for_market(
            df,
            market=Market.TOTAL,
            method=method,
            sport=None,
            season=None,
            source_id=None,
        )
        global_variance_calibrator = variance_calib
        _LOG.info("[fit_total_bucket_calibrators] Fitted global variance calibrator")
    except Exception as e:
        _LOG.warning("[fit_total_bucket_calibrators] Failed to fit global variance: %s", e)
    
    # ===== FIT PER-BUCKET CALIBRATORS =====
    bucket_calibrators = {}
    bucket_flags = {"mean": {}, "variance": {}}
    
    for bucket_label in ["low", "mid", "high"]:
        bucket_df = df[df["total_bucket"] == bucket_label].copy()
        bucket_sample_count = len(bucket_df)
        
        if bucket_sample_count < min_samples_per_bucket:
            _LOG.info(
                "[fit_total_bucket_calibrators] Insufficient samples for %s bucket "
                "(%d < %d); skipping",
                bucket_label, bucket_sample_count, min_samples_per_bucket,
            )
            bucket_flags["mean"][bucket_label] = False
            bucket_flags["variance"][bucket_label] = False
            continue
        
        # Fit mean calibrator for bucket
        try:
            bucket_mean_calib, _ = fit_calibrator_for_market(
                bucket_df,
                market=Market.TOTAL,
                method=method,
                sport=None,
                season=None,
                source_id=None,
            )
            if isinstance(bucket_mean_calib, MeanCalibrator):
                bucket_calibrators[f"{bucket_label}_mean"] = bucket_mean_calib
                bucket_flags["mean"][bucket_label] = True
                _LOG.info("[fit_total_bucket_calibrators] Fitted %s bucket mean calibrator", bucket_label)
        except Exception as e:
            _LOG.warning("[fit_total_bucket_calibrators] Failed to fit %s mean: %s", bucket_label, e)
            bucket_flags["mean"][bucket_label] = False
        
        # Fit variance calibrator for bucket
        try:
            bucket_var_calib, _ = fit_calibrator_for_market(
                bucket_df,
                market=Market.TOTAL,
                method=method,
                sport=None,
                season=None,
                source_id=None,
            )
            bucket_calibrators[f"{bucket_label}_variance"] = bucket_var_calib
            bucket_flags["variance"][bucket_label] = True
            _LOG.info("[fit_total_bucket_calibrators] Fitted %s bucket variance calibrator", bucket_label)
        except Exception as e:
            _LOG.warning("[fit_total_bucket_calibrators] Failed to fit %s variance: %s", bucket_label, e)
            bucket_flags["variance"][bucket_label] = False
    
    # ===== SAVE ARTIFACTS =====
    spec = get_market_spec(Market.TOTAL)
    calib_dir = spec.calibrator_dir(sport, season, source_id)
    calib_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    # Save global calibrators
    if global_mean_calibrator:
        try:
            global_mean_path = save_calibrator(
                global_mean_calibrator,
                calib_dir / "global",
                market="total_mean",
                metadata={"sport": sport, "season": season, "regime": "global"},
            )
            saved_files["calibrator_global_mean"] = str(global_mean_path.relative_to(calib_dir.parent.parent.parent))
        except Exception as e:
            _LOG.warning("[fit_total_bucket_calibrators] Failed to save global mean: %s", e)
    
    if global_variance_calibrator:
        try:
            global_var_path = save_calibrator(
                global_variance_calibrator,
                calib_dir / "global",
                market="total",
                metadata={"sport": sport, "season": season, "regime": "global"},
            )
            saved_files["calibrator_global_variance"] = str(global_var_path.relative_to(calib_dir.parent.parent.parent))
        except Exception as e:
            _LOG.warning("[fit_total_bucket_calibrators] Failed to save global variance: %s", e)
    
    # Save per-bucket calibrators
    for bucket_label in ["low", "mid", "high"]:
        for cal_type in ["mean", "variance"]:
            key = f"{bucket_label}_{cal_type}"
            if key in bucket_calibrators:
                try:
                    cal_obj = bucket_calibrators[key]
                    market_label = f"total_{cal_type}" if cal_type == "mean" else "total"
                    cal_path = save_calibrator(
                        cal_obj,
                        calib_dir / f"bucket_{bucket_label}",
                        market=market_label,
                        metadata={"sport": sport, "season": season, "regime": bucket_label},
                    )
                    saved_files[f"calibrator_bucket_{bucket_label}_{cal_type}"] = str(
                        cal_path.relative_to(calib_dir.parent.parent.parent)
                    )
                except Exception as e:
                    _LOG.warning("[fit_total_bucket_calibrators] Failed to save %s %s: %s", bucket_label, cal_type, e)
    
    # ===== CREATE AND SAVE MANIFEST =====
    manifest = TotalBucketManifest(
        sport=sport,
        season=season,
        source=source_id,
        market="total",
        low_threshold=low_threshold,
        mid_threshold=mid_threshold,
        min_samples_per_bucket=min_samples_per_bucket,
        samples_global=sample_counts["global"],
        samples_low=sample_counts["low"],
        samples_mid=sample_counts["mid"],
        samples_high=sample_counts["high"],
        has_global_mean=global_mean_calibrator is not None,
        has_global_variance=global_variance_calibrator is not None,
        has_bucket_mean=bucket_flags["mean"],
        has_bucket_variance=bucket_flags["variance"],
        fit_start_date=start_date,
        fit_end_date=end_date,
        calibrator_global_mean=saved_files.get("calibrator_global_mean"),
        calibrator_global_variance=saved_files.get("calibrator_global_variance"),
        calibrators_bucket_mean={
            b: saved_files.get(f"calibrator_bucket_{b}_mean")
            for b in ["low", "mid", "high"]
            if saved_files.get(f"calibrator_bucket_{b}_mean")
        },
        calibrators_bucket_variance={
            b: saved_files.get(f"calibrator_bucket_{b}_variance")
            for b in ["low", "mid", "high"]
            if saved_files.get(f"calibrator_bucket_{b}_variance")
        },
    )
    
    manifest_path = save_total_bucket_manifest(manifest, calib_dir)
    _LOG.info("[fit_total_bucket_calibrators] Saved manifest to %s", manifest_path)
    
    return global_variance_calibrator, global_mean_calibrator, bucket_calibrators, manifest_path


def calibrate_sport_season(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    models: list[str],
    markets: list[Market] | None = None,
    source_id: str = "historical",
    method: str = "auto",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, tuple[Any, Path | None]]:
    """Complete calibration workflow for a sport/season.

    This is the main entry point. It:
    1. Loads all completed games for the sport/season
    2. Generates predictions from specified models
    3. Builds calibration datasets per market
    4. Fits calibrators
    5. Saves artifacts

    Args:
        db_path: Path to database
        sport: Sport code (e.g., 'nba', 'nfl', 'mlb')
        season: Season identifier
        models: List of model names to use for predictions
        markets: Markets to calibrate (default: all [ML, SPREAD, TOTAL])
        source_id: Identifier for saved calibrators (default: "historical")
        method: Calibration method (default: "auto")
        start_date: Optional start filter (YYYY-MM-DD)
        end_date: Optional end filter (YYYY-MM-DD)

    Returns:
        Dictionary mapping market name to (calibrator, saved_path) tuple
    """
    if markets is None:
        markets = [Market.ML, Market.SPREAD, Market.TOTAL]

    _LOG.info(
        f"[calibrate_sport_season] Starting calibration for {sport}/{season}, "
        f"models={models}, markets={[m.name for m in markets]}"
    )

    # Resolve ensemble config (models + weights) and restrict to configured models.
    requested_models = list(models or [])
    ensemble_config = load_ensemble_config(
        sport=sport,
        season=season,
        available_models=None,
    )
    market_configs = ensemble_config.get("markets", {})
    market_model_map: dict[Market, list[str]] = {}
    market_weight_map: dict[Market, dict[str, float] | None] = {}
    market_ensemble_ids: dict[Market, str] = {}
    for market in markets:
        market_key = market.name
        entry = market_configs.get(market_key, {})
        market_models = list(entry.get("models") or [])
        market_weights = entry.get("weights")
        ensemble_id = str(entry.get("ensemble_id") or source_id).strip() or source_id
        market_model_map[market] = market_models
        market_weight_map[market] = market_weights
        market_ensemble_ids[market] = ensemble_id

    configured_models = sorted(
        {m for models_for_market in market_model_map.values() for m in models_for_market}
    )
    if requested_models:
        requested_set = {m.strip().lower() for m in requested_models if str(m).strip()}
        missing_from_request = [m for m in configured_models if m not in requested_set]
        if missing_from_request:
            _LOG.info(
                "[calibrate_sport_season] Using ensemble-config models not listed in --models: %s",
                missing_from_request,
            )
    if not configured_models:
        raise ValueError("No configured ensemble models available for calibration.")

    # Load completed games
    games_df = load_completed_games(
        db_path,
        sport=sport,
        season=season,
        start_date=start_date,
        end_date=end_date,
    )

    if games_df.empty:
        raise ValueError(f"No completed games found for {sport}/{season}")

    # Generate predictions once (walk-forward backtest) and reuse across markets.
    preds_df = generate_model_predictions(
        db_path,
        games_df,
        sport=sport,
        season=season,
        models=configured_models,
        start_date=start_date,
        end_date=end_date,
    )

    if preds_df.empty:
        _LOG.warning("[calibrate_sport_season] No predictions generated; aborting")
        return {}

    results = {}

    for market in markets:
        try:
            _LOG.info(f"[calibrate_sport_season] Processing {market.name}")

            ensemble_id = market_ensemble_ids.get(market, source_id)
            market_models = market_model_map.get(market, [])
            market_weights = market_weight_map.get(market)
            _LOG.info(
                "[calibrate_sport_season] %s ensemble_id=%s models=%s",
                market.name,
                ensemble_id,
                market_models,
            )
            market_preds_df = _ensemble_predictions_for_market(
                preds_df,
                market=market,
                sport=sport,
                season=season,
                ensemble_id=ensemble_id,
                models=market_models,
                weights=market_weights,
            )
            if market_preds_df.empty:
                _LOG.warning(f"[calibrate_sport_season] No predictions for {market.name}, skipping")
                continue

            # Build calibration dataset (format depends on market type)
            if market == Market.ML:
                dataset_df = build_ml_calibration_dataset(games_df, market_preds_df)
            elif market == Market.SPREAD:
                dataset_df = build_spread_calibration_dataset(games_df, market_preds_df)
            elif market == Market.TOTAL:
                dataset_df = build_total_calibration_dataset(games_df, market_preds_df)
            else:
                raise ValueError(f"Unknown market: {market}")

            if dataset_df.empty:
                _LOG.warning(f"[calibrate_sport_season] No valid data for {market.name}, skipping")
                continue

            # Log calibration window details for transparency
            _log_calibration_window(
                games_df,
                market=market.name,
                window_type="expanding",  # For now, always expanding (all data in season)
                start_date=start_date,
                end_date=end_date,
            )

            # Fit calibrator
            calibrator, saved_path = fit_calibrator_for_market(
                dataset_df,
                market=market,
                method=method,
                sport=sport,
                season=season,
                source_id=source_id,
            )

            results[market.name] = (calibrator, saved_path)

        except Exception as e:
            _LOG.error(f"[calibrate_sport_season] Failed for {market.name}: {e}", exc_info=True)

    _LOG.info(
        f"[calibrate_sport_season] Calibration complete. "
        f"Fitted {len(results)} markets: {list(results.keys())}"
    )

    return results
