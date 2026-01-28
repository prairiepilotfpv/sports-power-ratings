"""Apply fitted calibrators to generate calibrated predictions.

This module loads fitted calibrators and applies them to ensemble predictions,
storing the results back in the database for use by the BETS sheet generation.

Flow:
1. Load fitted calibrator for a market + source
2. Query ensemble predictions from database
3. Apply calibrator transformation
4. Save calibrated values back to database
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pandas as pd

from markets.base import Market
from markets.registry import get_market_spec
from calibration.base import BaseCalibrator
from calibration.io import load_latest_calibrator
from calibration.distribution import VarianceCalibrator

_LOG = logging.getLogger(__name__)


def get_calibrator_for_market(
    *,
    sport: str,
    season: str,
    market: Market,
    source_id: str = "historical",
) -> BaseCalibrator | None:
    """Load the latest fitted calibrator for a market.

    Args:
        sport: Sport code (e.g., 'nba')
        season: Season identifier
        market: Market type (ML, SPREAD, or TOTAL)
        source_id: Calibrator source identifier (default: 'historical')

    Returns:
        BaseCalibrator instance or None if not found
    """
    try:
        spec = get_market_spec(market)
        calib_dir = spec.calibrator_dir(sport, season, source_id)
        
        if not calib_dir.exists():
            _LOG.debug(f"Calibrator directory not found: {calib_dir}")
            return None
        
        # Load the latest calibrator from this directory
        # (get_market_spec already handles directory structure)
        calibrator = load_latest_calibrator(
            sport=sport,
            season=season,
            model=source_id,
            market=market.value,
        )
        
        if calibrator:
            _LOG.info(
                f"Loaded calibrator for {sport}/{season}/{market.name}/{source_id}: "
                f"{calibrator.metadata}"
            )
        
        return calibrator
    except Exception as e:
        _LOG.debug(f"Failed to load calibrator for {market.name}: {e}")
        return None


def apply_ml_calibrator_to_predictions(
    predictions_df: pd.DataFrame,
    calibrator: BaseCalibrator,
) -> pd.DataFrame:
    """Apply ML probability calibrator to predictions.

    Args:
        predictions_df: DataFrame with columns home_win_prob, away_win_prob
        calibrator: Fitted probability calibrator

    Returns:
        DataFrame with calibrated_home_win_prob, calibrated_away_win_prob
    """
    result = predictions_df[["home_win_prob", "away_win_prob"]].copy()
    
    # Transform home win probability
    if "home_win_prob" in result.columns and result["home_win_prob"].notna().any():
        result["calibrated_home_win_prob"] = calibrator.transform(
            result["home_win_prob"]
        )
    else:
        result["calibrated_home_win_prob"] = pd.NA
    
    # Away win prob is complement
    result["calibrated_away_win_prob"] = 1.0 - result["calibrated_home_win_prob"]
    
    return result[["calibrated_home_win_prob", "calibrated_away_win_prob"]]


def apply_distribution_calibrator_to_predictions(
    predictions_df: pd.DataFrame,
    calibrator: VarianceCalibrator,
    pred_mean_col: str,
    pred_sd_col: str,
) -> pd.DataFrame:
    """Apply distribution calibrator to SPREAD or TOTAL predictions.

    Args:
        predictions_df: DataFrame with distribution predictions
    calibrator: Fitted VarianceCalibrator
        pred_mean_col: Column name for predicted mean
        pred_sd_col: Column name for predicted SD

    Returns:
        DataFrame with calibrated_mean, calibrated_sd
    """
    # Build input for calibrator
    df = predictions_df[[pred_mean_col, pred_sd_col]].copy()
    df.columns = ["pred_mean", "pred_sd"]
    
    # Apply calibrator
    result = calibrator.transform(df)
    
    return result[["calibrated_mean", "calibrated_sd"]]


def calibrate_ensemble_predictions(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    market: Market,
    source_id: str = "historical",
) -> tuple[int, str | None]:
    """Calibrate ensemble predictions for a market and save to database.

    This function:
    1. Loads the fitted calibrator for the market/source
    2. Queries ensemble predictions from the database
    3. Applies the calibrator
    4. Stores calibrated values back in the database

    Args:
        db_path: Path to database
        sport: Sport code
        season: Season identifier
        market: Market to calibrate
        source_id: Calibrator source identifier

    Returns:
        Tuple of (rows_calibrated, saved_path_if_any)
    """
    _LOG.info(
        f"[calibrate_ensemble_predictions] Starting calibration for "
        f"{sport}/{season}/{market.name}"
    )
    
    # Load calibrator
    calibrator = get_calibrator_for_market(
        sport=sport,
        season=season,
        market=market,
        source_id=source_id,
    )
    
    if calibrator is None:
        _LOG.warning(
            f"[calibrate_ensemble_predictions] No calibrator found for {market.name}, skipping"
        )
        return 0, None
    
    # Query predictions from database
    with closing(sqlite3.connect(str(db_path))) as conn:
        if market == Market.ML:
            # ML market: query home_win_prob
            query = """
                SELECT DISTINCT
                    game_id,
                    home_win_prob,
                    (1.0 - home_win_prob) as away_win_prob
                FROM ratings
                WHERE sport = ? AND season = ?
                  AND home_win_prob IS NOT NULL
            """
            preds_df = pd.read_sql(query, conn, params=[sport, season])
            
            if preds_df.empty:
                _LOG.warning(f"[calibrate_ensemble_predictions] No ML predictions found")
                return 0, None
            
            # Apply calibrator
            calib_results = apply_ml_calibrator_to_predictions(preds_df, calibrator)
            preds_df["calibrated_home_win_prob"] = calib_results["calibrated_home_win_prob"]
            preds_df["calibrated_away_win_prob"] = calib_results["calibrated_away_win_prob"]
            
            # Update database
            cursor = conn.cursor()
            cursor.executemany(
                """
                UPDATE ratings
                SET calibrated_home_win_prob = ?,
                    calibrated_away_win_prob = ?
                WHERE game_id = ? AND sport = ? AND season = ?
                """,
                [
                    (
                        row.get("calibrated_home_win_prob"),
                        row.get("calibrated_away_win_prob"),
                        row["game_id"],
                        sport,
                        season,
                    )
                    for _, row in preds_df.iterrows()
                ],
            )
            conn.commit()
            rows_updated = len(preds_df)
            
        elif market == Market.SPREAD:
            # SPREAD market: query margin predictions
            query = """
                SELECT DISTINCT
                    game_id,
                    margin_mean,
                    margin_sd
                FROM ratings
                WHERE sport = ? AND season = ?
                  AND margin_mean IS NOT NULL
                  AND margin_sd IS NOT NULL
            """
            preds_df = pd.read_sql(query, conn, params=[sport, season])
            
            if preds_df.empty:
                _LOG.warning(f"[calibrate_ensemble_predictions] No SPREAD predictions found")
                return 0, None
            
            # Apply calibrator
            calib_results = apply_distribution_calibrator_to_predictions(
                preds_df,
                calibrator,
                "margin_mean",
                "margin_sd",
            )
            preds_df["calibrated_margin_mean"] = calib_results["calibrated_mean"]
            preds_df["calibrated_margin_sd"] = calib_results["calibrated_sd"]
            
            # Update database
            cursor = conn.cursor()
            cursor.executemany(
                """
                UPDATE ratings
                SET calibrated_margin_mean = ?,
                    calibrated_margin_sd = ?
                WHERE game_id = ? AND sport = ? AND season = ?
                """,
                [
                    (
                        row.get("calibrated_margin_mean"),
                        row.get("calibrated_margin_sd"),
                        row["game_id"],
                        sport,
                        season,
                    )
                    for _, row in preds_df.iterrows()
                ],
            )
            conn.commit()
            rows_updated = len(preds_df)
            
        elif market == Market.TOTAL:
            # TOTAL market: query total predictions
            query = """
                SELECT DISTINCT
                    game_id,
                    total_mean,
                    total_sd
                FROM ratings
                WHERE sport = ? AND season = ?
                  AND total_mean IS NOT NULL
                  AND total_sd IS NOT NULL
            """
            preds_df = pd.read_sql(query, conn, params=[sport, season])
            
            if preds_df.empty:
                _LOG.warning(f"[calibrate_ensemble_predictions] No TOTAL predictions found")
                return 0, None
            
            # Apply calibrator
            calib_results = apply_distribution_calibrator_to_predictions(
                preds_df,
                calibrator,
                "total_mean",
                "total_sd",
            )
            preds_df["calibrated_total_mean"] = calib_results["calibrated_mean"]
            preds_df["calibrated_total_sd"] = calib_results["calibrated_sd"]
            
            # Update database
            cursor = conn.cursor()
            cursor.executemany(
                """
                UPDATE ratings
                SET calibrated_total_mean = ?,
                    calibrated_total_sd = ?
                WHERE game_id = ? AND sport = ? AND season = ?
                """,
                [
                    (
                        row.get("calibrated_total_mean"),
                        row.get("calibrated_total_sd"),
                        row["game_id"],
                        sport,
                        season,
                    )
                    for _, row in preds_df.iterrows()
                ],
            )
            conn.commit()
            rows_updated = len(preds_df)
        
        else:
            raise ValueError(f"Unknown market: {market}")
    
    _LOG.info(
        f"[calibrate_ensemble_predictions] Successfully calibrated {rows_updated} "
        f"{market.name} predictions"
    )
    
    return rows_updated, None


def calibrate_all_markets(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    source_id: str = "historical",
) -> dict[str, int]:
    """Calibrate all markets and save results to database.

    Args:
        db_path: Path to database
        sport: Sport code
        season: Season identifier
        source_id: Calibrator source identifier

    Returns:
        Dictionary mapping market name to number of rows calibrated
    """
    results = {}
    
    for market in [Market.ML, Market.SPREAD, Market.TOTAL]:
        try:
            rows, _ = calibrate_ensemble_predictions(
                db_path,
                sport=sport,
                season=season,
                market=market,
                source_id=source_id,
            )
            results[market.name] = rows
        except Exception as e:
            _LOG.error(
                f"[calibrate_all_markets] Failed to calibrate {market.name}: {e}",
                exc_info=True,
            )
    
    return results
