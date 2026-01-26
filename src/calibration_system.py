"""
Independent Ensemble Calibration System

Generates calibrated predictions for ML/SPREAD/TOTAL markets by:
1. Running ensemble predictions on historical game data
2. Fitting calibrators to prediction vs outcome pairs
3. Outputting calibrated predictions for use in BETS sheet

Completely standalone - does not depend on betting pipeline.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_LOG = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result from fitting a calibrator to historical data."""
    market: str
    ensemble_id: str
    start_date: date
    end_date: date
    n_samples: int
    calibrator_type: str
    params: dict[str, float]
    metric_before: float
    metric_after: float


class IsotonicCalibrator:
    """Simple isotonic regression calibrator."""
    
    def __init__(self):
        self.bins = []
        self.calibrated_probs = []
    
    def fit(self, probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10):
        """Fit isotonic calibrator using binned approach."""
        probs = np.asarray(probs)
        outcomes = np.asarray(outcomes)
        
        # Create bins
        self.bins = np.linspace(0, 1, n_bins + 1)
        self.calibrated_probs = []
        
        for i in range(len(self.bins) - 1):
            mask = (probs >= self.bins[i]) & (probs < self.bins[i + 1])
            if mask.sum() > 0:
                self.calibrated_probs.append(outcomes[mask].mean())
            else:
                self.calibrated_probs.append(0.5)  # Default to 0.5 if no samples
        
        return self
    
    def calibrate(self, prob: float) -> float:
        """Apply calibration to a probability."""
        if prob < 0 or prob > 1:
            return prob
        
        # Find bin
        for i in range(len(self.bins) - 1):
            if self.bins[i] <= prob < self.bins[i + 1]:
                return self.calibrated_probs[i]
        
        return self.calibrated_probs[-1] if self.calibrated_probs else prob


class TemperatureScalingCalibrator:
    """Temperature scaling calibrator (Platt scaling variant)."""
    
    def __init__(self):
        self.temperature = 1.0
    
    def fit(self, probs: np.ndarray, outcomes: np.ndarray):
        """Fit temperature scaling using negative log-loss."""
        probs = np.asarray(probs)
        outcomes = np.asarray(outcomes)
        
        def nll(temp):
            if temp <= 0:
                return 1e10
            scaled = 1.0 / (1.0 + np.exp(-np.log(probs / (1 - probs + 1e-10)) / temp))
            return -np.mean(outcomes * np.log(scaled + 1e-10) + (1 - outcomes) * np.log(1 - scaled + 1e-10))
        
        result = minimize(nll, [1.0], bounds=[(0.1, 10.0)], method='L-BFGS-B')
        self.temperature = float(result.x[0])
        return self
    
    def calibrate(self, prob: float) -> float:
        """Apply temperature scaling to a probability."""
        if prob < 1e-10:
            prob = 1e-10
        if prob > 1 - 1e-10:
            prob = 1 - 1e-10
        
        logit = np.log(prob / (1 - prob))
        scaled_logit = logit / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled_logit))


class EnsembleCalibrator:
    """Fits calibrators to ensemble predictions against actual outcomes."""
    
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
    
    def get_historical_predictions(
        self,
        sport: str,
        season: str,
        market: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Load ensemble predictions with actual outcomes from database.
        
        Returns DataFrame with columns:
        - prediction: ensemble prediction probability
        - outcome: actual outcome (0 or 1)
        - game_id: for reference
        - game_date: date game was played
        """
        conn = sqlite3.connect(str(self.db_path))
        
        # Build query for historical games with ensemble predictions and outcomes
        query = """
        SELECT
            bp.model_prob as prediction,
            CASE 
                WHEN bp.market_type = 'ML' THEN
                    CASE WHEN g.home_team = bp.selection AND g.home_score > g.away_score THEN 1
                         WHEN g.away_team = bp.selection AND g.away_score > g.home_score THEN 1
                         ELSE 0 END
                WHEN bp.market_type = 'spread' THEN
                    CASE WHEN g.home_score - g.away_score >= bp.line THEN 1
                         ELSE 0 END
                WHEN bp.market_type = 'total' THEN
                    CASE WHEN g.home_score + g.away_score >= bp.line THEN 1
                         ELSE 0 END
            END as outcome,
            bp.game_id,
            g.date as game_date,
            bp.selection,
            bp.line
        FROM bets_predictions bp
        JOIN games g ON bp.game_id = g.game_id
        WHERE bp.sport = ?
          AND bp.season = ?
          AND bp.market_type = ?
          AND g.home_score IS NOT NULL
          AND g.away_score IS NOT NULL
          AND bp.model_prob IS NOT NULL
          AND bp.model_prob > 0
          AND bp.model_prob < 1
        """
        
        params = [sport, season, market.lower() if market != 'ML' else 'ML']
        
        if start_date:
            query += " AND DATE(g.date) >= DATE(?)"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND DATE(g.date) <= DATE(?)"
            params.append(end_date.isoformat())
        
        query += " ORDER BY g.date"
        
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        
        return df
    
    def fit_calibrator(
        self,
        sport: str,
        season: str,
        market: str,
        ensemble_id: str,
        calibrator_type: str = "temperature",
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> CalibrationResult:
        """Fit a calibrator for an ensemble on historical data.
        
        Args:
            sport: Sport code (e.g., 'nba')
            season: Season code (e.g., '2025-26')
            market: Market type ('ML', 'SPREAD', 'TOTAL')
            ensemble_id: Ensemble identifier (e.g., 'ensemble_ml_v1')
            calibrator_type: 'temperature' or 'isotonic'
            start_date: Optional start date for historical data
            end_date: Optional end date for historical data
        
        Returns:
            CalibrationResult with fitted parameters
        """
        _LOG.info(f"Loading historical predictions for {sport}/{season} {market}")
        
        df = self.get_historical_predictions(
            sport, season, market, start_date, end_date
        )
        
        if df.empty:
            raise ValueError(f"No historical predictions found for {market}")
        
        _LOG.info(f"Loaded {len(df)} prediction/outcome pairs for {market}")
        
        probs = df['prediction'].values
        outcomes = df['outcome'].values
        
        # Calculate metrics before calibration
        log_loss_before = -np.mean(
            outcomes * np.log(probs + 1e-10) + 
            (1 - outcomes) * np.log(1 - probs + 1e-10)
        )
        
        # Fit calibrator
        if calibrator_type == "temperature":
            calibrator = TemperatureScalingCalibrator()
            calibrator.fit(probs, outcomes)
            params = {"temperature": calibrator.temperature}
        else:
            calibrator = IsotonicCalibrator()
            calibrator.fit(probs, outcomes)
            params = {"bins": calibrator.bins.tolist(), "calibrated_probs": calibrator.calibrated_probs}
        
        # Calculate metrics after calibration
        calibrated_probs = np.array([calibrator.calibrate(p) for p in probs])
        calibrated_probs = np.clip(calibrated_probs, 1e-10, 1 - 1e-10)
        log_loss_after = -np.mean(
            outcomes * np.log(calibrated_probs) + 
            (1 - outcomes) * np.log(1 - calibrated_probs)
        )
        
        result = CalibrationResult(
            market=market,
            ensemble_id=ensemble_id,
            start_date=df['game_date'].min() if not df.empty else start_date,
            end_date=df['game_date'].max() if not df.empty else end_date,
            n_samples=len(df),
            calibrator_type=calibrator_type,
            params=params,
            metric_before=log_loss_before,
            metric_after=log_loss_after,
        )
        
        _LOG.info(
            f"Calibration complete for {market}: "
            f"log_loss {log_loss_before:.4f} → {log_loss_after:.4f}"
        )
        
        return result, calibrator
    
    def apply_calibrator(
        self,
        calibrator: TemperatureScalingCalibrator | IsotonicCalibrator,
        probabilities: np.ndarray | list,
    ) -> np.ndarray:
        """Apply a fitted calibrator to probabilities."""
        return np.array([calibrator.calibrate(p) for p in probabilities])


def save_calibration_result(
    result: CalibrationResult,
    output_dir: str | Path = "data/calibrators"
) -> Path:
    """Save calibration result to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{result.market}_{result.ensemble_id}_{result.calibrator_type}.json"
    filepath = output_dir / filename
    
    data = {
        "market": result.market,
        "ensemble_id": result.ensemble_id,
        "start_date": result.start_date.isoformat(),
        "end_date": result.end_date.isoformat(),
        "n_samples": result.n_samples,
        "calibrator_type": result.calibrator_type,
        "params": result.params,
        "metric_before": float(result.metric_before),
        "metric_after": float(result.metric_after),
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    _LOG.info(f"Saved calibration result to {filepath}")
    return filepath


def load_calibration(filepath: str | Path) -> tuple[str, Any]:
    """Load a calibration result and reconstruct the calibrator.
    
    Returns (calibrator_type, calibrator_object)
    """
    with open(filepath) as f:
        data = json.load(f)
    
    calibrator_type = data['calibrator_type']
    params = data['params']
    
    if calibrator_type == "temperature":
        calibrator = TemperatureScalingCalibrator()
        calibrator.temperature = params['temperature']
    else:  # isotonic
        calibrator = IsotonicCalibrator()
        calibrator.bins = np.array(params['bins'])
        calibrator.calibrated_probs = params['calibrated_probs']
    
    return calibrator_type, calibrator
