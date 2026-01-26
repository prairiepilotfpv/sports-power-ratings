"""Core calibration engine for ensemble predictions.

This module provides a standalone calibration system that:
1. Generates ensemble predictions for historical games
2. Computes actual outcomes
3. Fits calibrators to align predictions with outcomes
4. Persists calibrators for use in production

Completely independent from the betting pipeline.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from markets.base import Market
from markets.registry import get_market_spec
from pipelines.calibration_utils import select_calibrator

_LOG = logging.getLogger(__name__)


def generate_predictions_for_games(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    models: list[str],
    game_ids: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate ensemble predictions for specified games.
    
    Args:
        db_path: Path to database
        sport: Sport identifier (e.g., 'nba')
        season: Season identifier (e.g., '2025-26')
        models: List of model names to ensemble
        game_ids: Optional list of game IDs to predict for (if None, use date range)
        start_date: Optional start date (YYYY-MM-DD) to filter games
        end_date: Optional end date (YYYY-MM-DD) to filter games
    
    Returns:
        Dictionary with keys 'ML', 'SPREAD', 'TOTAL' containing DataFrames of predictions
    """
    _LOG.info(f"[generate_predictions] Starting for {sport}/{season}, models={models}")
    
    # Query games to predict for
    from src.data.repository import get_schedule
    
    with closing(sqlite3.connect(str(db_path))) as conn:
        query = "SELECT * FROM games WHERE sport = ? AND season = ?"
        params = [sport, season]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        if game_ids:
            placeholders = ",".join("?" * len(game_ids))
            query += f" AND game_id IN ({placeholders})"
            params.extend(game_ids)
        
        games_df = pd.read_sql(query, conn, params=params)
    
    _LOG.info(f"[generate_predictions] Found {len(games_df)} games to predict for")
    
    if games_df.empty:
        return {
            Market.ML.name: pd.DataFrame(),
            Market.SPREAD.name: pd.DataFrame(),
            Market.TOTAL.name: pd.DataFrame(),
        }
    
    # Generate predictions for each market
    predictions = {Market.ML.name: [], Market.SPREAD.name: [], Market.TOTAL.name: []}
    
    for market in [Market.ML, Market.SPREAD, Market.TOTAL]:
        _LOG.info(f"[generate_predictions] Generating {market.name} predictions for {len(games_df)} games")
        
        market_preds = _generate_market_predictions(
            db_path=db_path,
            games_df=games_df,
            sport=sport,
            season=season,
            market=market,
            models=models,
        )
        
        predictions[market.name] = market_preds
        _LOG.info(f"[generate_predictions] Generated {len(market_preds)} {market.name} predictions")
    
    return predictions


def _generate_market_predictions(
    db_path: str | Path,
    games_df: pd.DataFrame,
    *,
    sport: str,
    season: str,
    market: Market,
    models: list[str],
) -> list[dict[str, Any]]:
    """Generate predictions for a specific market across games."""
    from src.data.repository import resolve_effective_params
    from src.models.registry import get_model
    
    records = []
    
    for model_name in models:
        _LOG.debug(f"[_generate_market_predictions] Getting {market.name} predictions from {model_name}")
        
        try:
            # Get model and active params
            model_cls = get_model(model_name)
            model = model_cls(sport, season)
            
            resolved = resolve_effective_params(
                db_path=db_path,
                sport=sport,
                season=season,
                model=model_name,
                market=market.name,
            )
            
            if resolved.params:
                model.set_params(**resolved.params)
            
            # Generate predictions for this market
            for _, game in games_df.iterrows():
                pred = model.forecast_market(game, market=market)
                if pred is not None:
                    records.append({
                        "game_id": game.get("game_id"),
                        "model_name": model_name,
                        "market": market.name,
                        **pred,  # Contains: home_win_prob, margin_mean, margin_sd, total, total_sd, etc.
                    })
        except Exception as e:
            _LOG.warning(f"[_generate_market_predictions] Failed to get {market.name} from {model_name}: {e}")
    
    return records


def compute_outcomes(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    game_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Compute actual outcomes for completed games.
    
    Returns DataFrame with columns: game_id, home_team, away_team, home_score, away_score
    Only includes games that have been completed (scores recorded).
    """
    _LOG.info(f"[compute_outcomes] Loading outcomes for {sport}/{season}")
    
    with closing(sqlite3.connect(str(db_path))) as conn:
        query = """
            SELECT game_id, home_team, away_team, home_score, away_score, date
            FROM games
            WHERE sport = ? AND season = ?
              AND home_score IS NOT NULL
              AND away_score IS NOT NULL
        """
        params = [sport, season]
        
        if game_ids:
            placeholders = ",".join("?" * len(game_ids))
            query += f" AND game_id IN ({placeholders})"
            params.extend(game_ids)
        
        outcomes_df = pd.read_sql(query, conn, params=params)
    
    _LOG.info(f"[compute_outcomes] Found {len(outcomes_df)} completed games")
    return outcomes_df


def build_calibration_dataset(
    predictions: dict[str, list[dict[str, Any]]],
    outcomes_df: pd.DataFrame,
    *,
    market: Market,
    selection: str | None = None,
) -> pd.DataFrame:
    """Build calibration dataset by matching predictions to outcomes.
    
    Returns DataFrame with columns: p_home_win, home_win (actual outcome)
    """
    _LOG.info(f"[build_calibration_dataset] Building {market.name} dataset with {len(outcomes_df)} outcomes")
    
    market_preds = predictions.get(market.name, [])
    if not market_preds:
        _LOG.warning(f"[build_calibration_dataset] No {market.name} predictions found")
        return pd.DataFrame(columns=["p_home_win", "home_win"])
    
    preds_df = pd.DataFrame(market_preds)
    _LOG.info(f"[build_calibration_dataset] Have {len(preds_df)} {market.name} predictions")
    
    # Merge predictions with outcomes
    merged = preds_df.merge(
        outcomes_df,
        on="game_id",
        how="inner",
    )
    
    _LOG.info(f"[build_calibration_dataset] After merge: {len(merged)} rows")
    
    if merged.empty:
        _LOG.warning(f"[build_calibration_dataset] No matching games between predictions and outcomes")
        return pd.DataFrame(columns=["p_home_win", "home_win"])
    
    # Convert predictions and outcomes to calibration format
    records = []
    
    for _, row in merged.iterrows():
        if market == Market.ML:
            # ML: extract home win probability
            p = row.get("home_win_prob")
            if p is None or (isinstance(p, float) and pd.isna(p)):
                continue
            
            # Determine actual outcome
            home_score = row.get("home_score")
            away_score = row.get("away_score")
            if home_score is None or away_score is None:
                continue
            
            outcome = 1 if home_score > away_score else 0
            records.append({"p_home_win": p, "home_win": outcome})
        
        elif market == Market.SPREAD:
            # SPREAD: compute cover probability
            margin_mean = row.get("margin_mean")
            margin_sd = row.get("margin_sd")
            if margin_mean is None or margin_sd is None or margin_sd <= 0:
                continue
            
            # Compute actual spread outcome
            home_score = row.get("home_score")
            away_score = row.get("away_score")
            if home_score is None or away_score is None:
                continue
            
            # For now, assume line is 0 (can be enhanced to use actual market lines)
            line = 0
            actual_margin = home_score - away_score
            outcome = 1 if actual_margin > line else 0
            
            # Compute cover probability (normal CDF)
            z = (line - margin_mean) / margin_sd
            p = 0.5 * (1.0 + _erf(z / 1.4142135623730951))
            records.append({"p_home_win": p, "home_win": outcome})
        
        elif market == Market.TOTAL:
            # TOTAL: compute over/under probability
            total = row.get("total")
            total_sd = row.get("total_sd")
            if total is None or total_sd is None or total_sd <= 0:
                continue
            
            # Compute actual total
            home_score = row.get("home_score")
            away_score = row.get("away_score")
            if home_score is None or away_score is None:
                continue
            
            actual_total = home_score + away_score
            # Assume line is the predicted total (can be enhanced)
            line = total
            outcome = 1 if actual_total > line else 0
            
            # Compute over probability (normal CDF)
            z = (line - total) / total_sd
            p = 0.5 * (1.0 + _erf(z / 1.4142135623730951))
            records.append({"p_home_win": p, "home_win": outcome})
    
    _LOG.info(f"[build_calibration_dataset] Built calibration dataset with {len(records)} records")
    return pd.DataFrame(records)


def fit_and_save_calibrator(
    dataset_df: pd.DataFrame,
    *,
    sport: str,
    season: str,
    market: Market,
    ensemble_id: str = "ensemble_v1",
    method: str = "auto",
) -> Path:
    """Fit a calibrator to the dataset and save it.
    
    Returns: Path to saved calibrator file
    """
    if dataset_df.empty:
        raise ValueError(f"Cannot fit calibrator for {market.name}: empty dataset")
    
    _LOG.info(
        f"[fit_and_save_calibrator] Fitting {method} calibrator for {market.name} "
        f"with {len(dataset_df)} records"
    )
    
    calibrator = select_calibrator(method, len(dataset_df))
    calibrator.fit(dataset_df)
    
    # Save to disk
    spec = get_market_spec(market)
    calib_dir = spec.calibrator_dir(sport, season, ensemble_id)
    calib_dir.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = calib_dir / f"{ensemble_id}_{timestamp}.joblib"
    
    calibrator.save(out_path)
    _LOG.info(f"[fit_and_save_calibrator] Saved calibrator to {out_path}")
    
    return out_path


def calibrate_ensemble(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    models: list[str],
    ensemble_id: str = "ensemble_v1",
    start_date: str | None = None,
    end_date: str | None = None,
    method: str = "auto",
) -> dict[str, Path]:
    """Complete calibration workflow: generate predictions, fit calibrators.
    
    Returns: Dictionary mapping market name to calibrator file path
    """
    _LOG.info(f"[calibrate_ensemble] Starting ensemble calibration for {sport}/{season}")
    
    # Step 1: Generate predictions for historical games
    predictions = generate_predictions_for_games(
        db_path,
        sport=sport,
        season=season,
        models=models,
        start_date=start_date,
        end_date=end_date,
    )
    
    # Step 2: Load outcomes
    outcomes_df = compute_outcomes(db_path, sport=sport, season=season)
    
    if outcomes_df.empty:
        raise ValueError(f"No completed games found for {sport}/{season}")
    
    # Step 3: Fit calibrators for each market
    calibrators = {}
    
    for market in [Market.ML, Market.SPREAD, Market.TOTAL]:
        try:
            dataset = build_calibration_dataset(
                predictions,
                outcomes_df,
                market=market,
            )
            
            if dataset.empty:
                _LOG.warning(f"[calibrate_ensemble] Skipping {market.name} - no data")
                continue
            
            calib_path = fit_and_save_calibrator(
                dataset,
                sport=sport,
                season=season,
                market=market,
                ensemble_id=ensemble_id,
                method=method,
            )
            calibrators[market.name] = calib_path
        except Exception as e:
            _LOG.error(f"[calibrate_ensemble] Failed to calibrate {market.name}: {e}")
    
    return calibrators


def _erf(x: float) -> float:
    """Error function (approximation matching Excel ERF)."""
    import math
    return math.erf(x)
