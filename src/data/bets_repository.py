"""Store and retrieve BETS predictions from database.

This module persists BETS sheet predictions (made on a specific date) alongside
actual game outcomes. Enables automatic validation of ensemble weights by
comparing historical predictions against actual results.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _calculate_missing_model_prob(bets_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate model_prob for rows where it's missing or contains Excel formulas.
    
    This is necessary because when reading Excel with formulas via pandas,
    the formulas are returned as text strings (starting with '='), not evaluated values.
    
    Calculation logic matches the Excel formulas:
    - ML: uses home_win_prob or away_win_prob based on selection
    - SPREAD: normal CDF of cover probability
    - TOTAL: normal CDF of over/under probability
    """
    if bets_df.empty or "market_type" not in bets_df.columns:
        return bets_df
    
    out_df = bets_df.copy()
    
    # Helper: Normal CDF approximation (matches Excel ERF formula)
    def _normal_cdf(x: float, mean: float, sd: float) -> float | None:
        if sd <= 0 or not math.isfinite(sd):
            return None
        z = (x - mean) / (sd * math.sqrt(2.0))
        return 0.5 * (1.0 + math.erf(z))
    
    # Helper: Check if value is missing or a formula string
    def _is_formula_or_missing(val: Any) -> bool:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return True
        if isinstance(val, str):
            return val.strip().startswith("=") or val.strip() == ""
        return False
    
    # ML market: use home_win_prob or away_win_prob based on selection
    if "market_type" in out_df.columns and "home_team" in out_df.columns and "selection" in out_df.columns:
        ml_mask = (out_df["market_type"] == "ML") & (out_df["model_prob"].apply(_is_formula_or_missing))
        if ml_mask.any():
            def _ml_prob(row):
                sel = str(row.get("selection", "")).strip()
                home = str(row.get("home_team", "")).strip()
                if sel == home:
                    return row.get("home_win_prob")
                return row.get("away_win_prob")
            out_df.loc[ml_mask, "model_prob"] = out_df[ml_mask].apply(_ml_prob, axis=1)
    
    # SPREAD market: calculate CDF probability for cover
    if "market_type" in out_df.columns:
        spread_mask = (out_df["market_type"] == "spread") & (out_df["model_prob"].apply(_is_formula_or_missing))
        if spread_mask.any():
            def _spread_prob(row):
                line = row.get("line")
                mean = row.get("margin_mean")
                sd = row.get("margin_sd")
                if line is None or mean is None or sd is None:
                    return None
                try:
                    line = float(line)
                    mean = float(mean)
                    sd = float(sd)
                except (TypeError, ValueError):
                    return None
                
                selection = str(row.get("selection", "")).strip()
                home_team = str(row.get("home_team", "")).strip()
                away_team = str(row.get("away_team", "")).strip()
                
                if selection == home_team:
                    # Home team cover: margin > -line
                    cdf_val = _normal_cdf(-line, mean=mean, sd=sd)
                    if cdf_val is None:
                        return None
                    return 1.0 - cdf_val
                elif selection == away_team:
                    # Away team cover: margin < line
                    cdf_val = _normal_cdf(line, mean=mean, sd=sd)
                    if cdf_val is None:
                        return None
                    return cdf_val
                return None
            
            out_df.loc[spread_mask, "model_prob"] = out_df[spread_mask].apply(_spread_prob, axis=1)
    
    # TOTAL market: calculate CDF probability for over/under
    if "market_type" in out_df.columns:
        total_mask = (out_df["market_type"] == "total") & (out_df["model_prob"].apply(_is_formula_or_missing))
        if total_mask.any():
            def _total_prob(row):
                line = row.get("line")
                mean = row.get("total")
                sd = row.get("total_sd")
                if line is None or mean is None or sd is None:
                    return None
                try:
                    line = float(line)
                    mean = float(mean)
                    sd = float(sd)
                except (TypeError, ValueError):
                    return None
                
                selection = str(row.get("selection", "")).strip().lower()
                over_cdf = _normal_cdf(line, mean=mean, sd=sd)
                if over_cdf is None:
                    return None
                
                if selection == "over":
                    return 1.0 - over_cdf
                elif selection == "under":
                    return over_cdf
                return None
            
            out_df.loc[total_mask, "model_prob"] = out_df[total_mask].apply(_total_prob, axis=1)
    
    return out_df


def save_bets_predictions(
    db_path: str | Path,
    *,
    bets_df: pd.DataFrame,
    sport: str,
    season: str,
    prediction_date: str | None = None,
) -> int:
    """Save BETS sheet predictions to database.
    
    Args:
        db_path: Path to SQLite database
        bets_df: BETS sheet DataFrame with game_id, home_win_prob, etc.
        sport: Sport code (e.g., 'nba')
        season: Season string (e.g., '2025-26')
        prediction_date: ISO date string (defaults to today UTC)
    
    Returns:
        Number of rows inserted/updated
    """
    if prediction_date is None:
        prediction_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    from data.repository import init_db

    init_db(db_path)
    
    if bets_df.empty:
        return 0
    
    # Calculate model_prob for rows where it's missing or contains formula strings
    # This is necessary because Excel formulas aren't evaluated when reading via pandas
    bets_df = _calculate_missing_model_prob(bets_df)
    
    # Log what we read from Excel
    from_log = logging.getLogger(__name__)
    if not bets_df.empty and "market_type" in bets_df.columns:
        from_log.info(f"[save_bets_predictions] Read {len(bets_df)} rows from Excel for {sport}/{season}")
        if "spread_source" in bets_df.columns:
            spread_df = bets_df[bets_df["market_type"] == "spread"]
            if len(spread_df) > 0:
                spread_sources = spread_df["spread_source"].unique()
                from_log.info(f"[save_bets_predictions] SPREAD rows from Excel: count={len(spread_df)}, sources={spread_sources}")
        if "total_source" in bets_df.columns:
            total_df = bets_df[bets_df["market_type"] == "total"]
            if len(total_df) > 0:
                total_sources = total_df["total_source"].unique()
                from_log.info(f"[save_bets_predictions] TOTAL rows from Excel: count={len(total_df)}, sources={total_sources}")
    
    # Only keep relevant columns
    required_cols = ["game_id"]
    optional_cols = [
        "home_win_prob",
        "model_prob",
        "edge",
        "ev",
        "market_type",
        "selection",
        "line",
        "market_forecast_source",
        "spread_source",
        "total_source",
        "ml_ensemble_components_json",
        "spread_ensemble_components_json",
        "total_ensemble_components_json",
    ]
    
    cols_to_keep = required_cols + [c for c in optional_cols if c in bets_df.columns]
    save_df = bets_df[cols_to_keep].copy()
    
    # Filter to rows that have either home_win_prob (ML) or model_prob (SPREAD/TOTAL)
    # This keeps all market types with predictions
    if "home_win_prob" in save_df.columns and "model_prob" in save_df.columns:
        save_df = save_df[
            (save_df["home_win_prob"].notna()) | (save_df["model_prob"].notna())
        ].copy()
    elif "home_win_prob" in save_df.columns:
        save_df = save_df[save_df["home_win_prob"].notna()].copy()
    
    if save_df.empty:
        return 0
    
    # Add metadata
    save_df["sport"] = sport
    save_df["season"] = season
    save_df["prediction_date"] = prediction_date
    
    # For SPREAD/TOTAL rows (which don't have home_win_prob), fill with a neutral value
    # This satisfies the NOT NULL constraint while keeping the row in the database
    if "home_win_prob" in save_df.columns:
        save_df["home_win_prob"] = save_df["home_win_prob"].fillna(0.5)
    
    # Fill missing optional columns with None
    for col in optional_cols:
        if col not in save_df.columns:
            save_df[col] = None

    # Normalize ensemble components JSON columns
    def _normalize_components(value: Any) -> str | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if isinstance(value, str):
            text = value.strip()
            return text if text else None
        try:
            return json.dumps(value)
        except Exception:
            return str(value)

    # Apply normalization to all ensemble components columns
    for col in ["ml_ensemble_components_json", "spread_ensemble_components_json", "total_ensemble_components_json"]:
        if col in save_df.columns:
            save_df[col] = save_df[col].apply(_normalize_components)
    
    # Map market_type to correct source column
    # ML uses market_forecast_source, SPREAD uses spread_source, TOTAL uses total_source
    def _get_source_for_market(row):
        market_type = row.get("market_type")
        if market_type == "spread":
            source = row.get("spread_source", row.get("market_forecast_source"))
        elif market_type == "total":
            source = row.get("total_source", row.get("market_forecast_source"))
        else:
            # ML or default
            source = row.get("market_forecast_source")
        return source
    
    save_df["market_forecast_source"] = save_df.apply(_get_source_for_market, axis=1)
    
    # Log what we're about to save
    from_log = logging.getLogger(__name__)
    source_counts = save_df.groupby(["market_type", "market_forecast_source"]).size()
    for (market_type, source), count in source_counts.items():
        from_log.info(f"[save_bets_predictions] SAVING: market_type={market_type}, source={source}, count={count}")
    
    # Detailed logging: show sample rows for each market type
    for market_type in ["ML", "spread", "total"]:
        market_rows = save_df[save_df["market_type"] == market_type]
        if len(market_rows) > 0:
            sample = market_rows.iloc[0]
            from_log.debug(f"[save_bets_predictions] SAMPLE {market_type}: game_id={sample.get('game_id')}, source={sample.get('market_forecast_source')}")
    
    # Insert via upsert (replace on conflict)
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        
        # Log what we're about to insert
        from_log.info(f"[save_bets_predictions] Inserting {len(save_df)} rows into database")
        
        # Insert or replace
        cursor.executemany(
            """
            INSERT OR REPLACE INTO bets_predictions
            (game_id, sport, season, prediction_date, home_win_prob,
             model_prob, edge, ev, market_type, selection, line,
             market_forecast_source, ml_ensemble_components_json,
             spread_ensemble_components_json, total_ensemble_components_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            save_df[
                [
                    "game_id",
                    "sport",
                    "season",
                    "prediction_date",
                    "home_win_prob",
                    "model_prob",
                    "edge",
                    "ev",
                    "market_type",
                    "selection",
                    "line",
                    "market_forecast_source",
                    "ml_ensemble_components_json",
                    "spread_ensemble_components_json",
                    "total_ensemble_components_json",
                ]
            ].values,
        )
        conn.commit()
        rowcount = cursor.rowcount
        from_log.info(f"[save_bets_predictions] Successfully inserted {rowcount} rows")
        return rowcount
    finally:
        conn.close()


def load_bets_predictions_for_validation(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    prediction_date: str | None = None,
    days_back: int = 7,
) -> pd.DataFrame:
    """Load BETS predictions that have completed games from a rolling window.
    
    Returns a DataFrame with predictions joined to actual outcomes.
    
    Args:
        db_path: Path to SQLite database
        sport: Sport code (e.g., 'nba')
        season: Season string (e.g., '2025-26')
        prediction_date: End date for window (defaults to yesterday)
        days_back: Number of days back from prediction_date to include (default: 7)
    
    Returns:
        DataFrame with columns: game_id, prediction_date, home_win_prob,
        home_score, away_score
    """
    if prediction_date is None:
        # Default to yesterday
        from datetime import timedelta
        prediction_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Calculate start date (days_back days before prediction_date)
    pred_dt = datetime.fromisoformat(prediction_date).date()
    from datetime import timedelta
    start_date = (pred_dt - timedelta(days=days_back - 1)).strftime("%Y-%m-%d")
    
    conn = sqlite3.connect(db_path)
    try:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(bets_predictions)")]
        has_components = "ml_ensemble_components_json" in cols
        select_cols = [
            "bp.game_id",
            "bp.prediction_date",
            "bp.home_win_prob",
            "g.home_score",
            "g.away_score",
        ]
        if has_components:
            select_cols.append("bp.ml_ensemble_components_json")
        query = f"""
        SELECT
            {", ".join(select_cols)}
        FROM bets_predictions bp
        JOIN games g ON bp.game_id = g.game_id
        WHERE bp.sport = ? AND bp.season = ?
            AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
            AND bp.prediction_date >= ? AND bp.prediction_date <= ?
        ORDER BY bp.prediction_date DESC, bp.game_id
        """
        params: list[Any] = [sport, season, start_date, prediction_date]
        df = pd.read_sql(query, conn, params=params)
        if not has_components:
            df["ml_ensemble_components_json"] = None
        return df
    finally:
        conn.close()


def get_available_prediction_dates(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
) -> list[str]:
    """Get all dates for which we have BETS predictions stored.
    
    Args:
        db_path: Path to SQLite database
        sport: Sport code (e.g., 'nba')
        season: Season string (e.g., '2025-26')
    
    Returns:
        List of ISO date strings, most recent first
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        rows = cursor.execute(
            """
            SELECT DISTINCT prediction_date
            FROM bets_predictions
            WHERE sport = ? AND season = ?
            ORDER BY prediction_date DESC
            """,
            (sport, season),
        ).fetchall()
        return [row[0] for row in rows]
    finally:
        conn.close()
