from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from data.repository import (
    get_active_ensemble_market_weights,
    get_active_ensemble_market_weights_source,
    get_active_model_market_params,
    get_active_model_market_params_source,
    init_db,
    set_active_ensemble_market_weights,
    set_active_model_market_params,
)


def test_init_db_creates_market_tuning_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "market.db"
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        metric_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(model_metrics)").fetchall()
        }
    assert "model_market_tuning_runs" in tables
    assert "model_market_active_params" in tables
    assert "ensemble_market_tuning_runs" in tables
    assert "ensemble_market_active_weights" in tables
    assert "backtest_mae_total" in metric_cols


def test_round_trip_model_market_params(tmp_path: Path) -> None:
    db_path = tmp_path / "params.db"
    init_db(db_path)
    params = {"k_factor": 20.0}
    set_active_model_market_params(
        db_path,
        sport="nba",
        season="2024-25",
        model="elo",
        market="ML",
        params=params,
        source_run_id="run-123",
    )
    loaded = get_active_model_market_params(
        db_path,
        sport="nba",
        season="2024-25",
        model="elo",
        market="ML",
    )
    source = get_active_model_market_params_source(
        db_path,
        sport="nba",
        season="2024-25",
        model="elo",
        market="ML",
    )
    assert loaded == params
    assert source == "run-123"


def test_round_trip_ensemble_market_weights(tmp_path: Path) -> None:
    db_path = tmp_path / "weights.db"
    init_db(db_path)
    weights = {"elo": 0.6, "gssd": 0.4}
    models = ["elo", "gssd"]
    set_active_ensemble_market_weights(
        db_path,
        sport="nba",
        season="2024-25",
        market="ML",
        ensemble_id="ensemble_ml_v1",
        weights_json=json.dumps(weights),
        models_json=json.dumps(models),
        source_run_id="run-456",
    )
    loaded = get_active_ensemble_market_weights(
        db_path,
        sport="nba",
        season="2024-25",
        market="ML",
        ensemble_id="ensemble_ml_v1",
    )
    source = get_active_ensemble_market_weights_source(
        db_path,
        sport="nba",
        season="2024-25",
        market="ML",
        ensemble_id="ensemble_ml_v1",
    )
    assert loaded == weights
    assert source == "run-456"
