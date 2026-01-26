"""Store and retrieve BETS predictions from database.

This module persists BETS sheet predictions (made on a specific date) alongside
actual game outcomes. Enables automatic validation of ensemble weights by
comparing historical predictions against actual results.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


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
        "ml_ensemble_components_json",
    ]
    
    cols_to_keep = required_cols + [c for c in optional_cols if c in bets_df.columns]
    save_df = bets_df[cols_to_keep].copy()
    
    # Filter to rows that have home_win_prob (predictions with estimated probability)
    if "home_win_prob" in save_df.columns:
        save_df = save_df[save_df["home_win_prob"].notna()].copy()
    
    if save_df.empty:
        return 0
    
    # Add metadata
    save_df["sport"] = sport
    save_df["season"] = season
    save_df["prediction_date"] = prediction_date
    
    # Fill missing optional columns with None
    for col in optional_cols:
        if col not in save_df.columns:
            save_df[col] = None
    if "ml_ensemble_components_json" in save_df.columns:
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

        save_df["ml_ensemble_components_json"] = save_df["ml_ensemble_components_json"].apply(
            _normalize_components
        )
    
    # Insert via upsert (replace on conflict)
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        
        # Insert or replace
        cursor.executemany(
            """
            INSERT OR REPLACE INTO bets_predictions
            (game_id, sport, season, prediction_date, home_win_prob,
             model_prob, edge, ev, market_type, selection, line,
             ml_ensemble_components_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    "ml_ensemble_components_json",
                ]
            ].values,
        )
        conn.commit()
        return cursor.rowcount
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
