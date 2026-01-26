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

from calibration.distribution import MarginalDistributionCalibrator
from markets.base import Market
from markets.registry import get_market_spec
from pipelines.calibration_utils import select_calibrator

_LOG = logging.getLogger(__name__)


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
    market: str,
) -> pd.DataFrame:
    """Generate ensemble predictions for games using specified models.

    This generates raw predictions (before calibration). For each game and model,
    we get the model's forecast which includes both probability and distribution info.

    Args:
        db_path: Path to database (for loading model params)
        games_df: DataFrame of games to predict for
        sport: Sport code
        season: Season identifier
        models: List of model names to ensemble
        market: Market to predict for ("ML", "spread", or "total")

    Returns:
        DataFrame with predictions, aggregated across models
    """
    from pipelines.model_params import resolve_effective_params
    from models.registry import get_model

    _LOG.info(
        f"[generate_model_predictions] Generating {market} predictions "
        f"for {len(games_df)} games from {len(models)} models"
    )

    all_predictions = []

    for model_name in models:
        _LOG.debug(f"[generate_model_predictions] Getting predictions from {model_name}")
        try:
            # Get model and resolve active parameters
            model_cls = get_model(model_name)
            model_instance = model_cls(sport, season)

            resolved = resolve_effective_params(
                db_path=db_path,
                sport=sport,
                season=season,
                model=model_name,
                market=market.name,
            )

            if resolved.params:
                model_instance.set_params(**resolved.params)

            # Generate predictions for each game
            for _, game_row in games_df.iterrows():
                try:
                    # Get model's GamePrediction for this market
                    pred = model_instance.forecast_game(
                        home_team=game_row["home_team"],
                        away_team=game_row["away_team"],
                        date=game_row["date"],
                        game_id=game_row.get("game_id"),
                    )

                    if pred is None:
                        continue

                    # Extract market-specific prediction
                    pred_dict = {
                        "game_id": game_row["game_id"],
                        "model": model_name,
                        "market": market.name,
                    }

                    if market == Market.ML:
                        pred_dict["p_home_win"] = pred.p_home_win
                    elif market == Market.SPREAD:
                        pred_dict["margin_mean"] = pred.margin_mean
                        pred_dict["margin_sd"] = pred.margin_sd
                    elif market == Market.TOTAL:
                        pred_dict["total_mean"] = pred.total_mean
                        pred_dict["total_sd"] = pred.total_sd

                    all_predictions.append(pred_dict)

                except Exception as e:
                    _LOG.debug(
                        f"[generate_model_predictions] Failed to predict {game_row['game_id']} "
                        f"with {model_name}: {e}"
                    )

        except Exception as e:
            _LOG.warning(
                f"[generate_model_predictions] Failed to load model {model_name}: {e}"
            )

    _LOG.info(f"[generate_model_predictions] Generated {len(all_predictions)} predictions")
    return pd.DataFrame(all_predictions) if all_predictions else pd.DataFrame()


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
        # SPREAD/TOTAL use distribution calibrators
        calibrator = MarginalDistributionCalibrator()
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
            spec = get_market_spec(market)
            calib_dir = spec.calibrator_dir(sport, season, source_id)
            calib_dir.mkdir(parents=True, exist_ok=True)

            from datetime import datetime, timezone

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            saved_path = calib_dir / f"{source_id}_{timestamp}.joblib"

            calibrator.save(saved_path)
            _LOG.info(f"[fit_calibrator_for_market] Saved calibrator to {saved_path}")
        except Exception as e:
            _LOG.warning(f"[fit_calibrator_for_market] Failed to save calibrator: {e}")

    return calibrator, saved_path


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

    # Generate predictions for each market
    results = {}

    for market in markets:
        try:
            _LOG.info(f"[calibrate_sport_season] Processing {market.name}")

            # Generate predictions for this market
            preds_df = generate_model_predictions(
                db_path,
                games_df,
                sport=sport,
                season=season,
                models=models,
                market=market,
            )

            if preds_df.empty:
                _LOG.warning(f"[calibrate_sport_season] No predictions for {market.name}, skipping")
                continue

            # Build calibration dataset (format depends on market type)
            if market == Market.ML:
                dataset_df = build_ml_calibration_dataset(games_df, preds_df)
            elif market == Market.SPREAD:
                dataset_df = build_spread_calibration_dataset(games_df, preds_df)
            elif market == Market.TOTAL:
                dataset_df = build_total_calibration_dataset(games_df, preds_df)
            else:
                raise ValueError(f"Unknown market: {market}")

            if dataset_df.empty:
                _LOG.warning(f"[calibrate_sport_season] No valid data for {market.name}, skipping")
                continue

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
