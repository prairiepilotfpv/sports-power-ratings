import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.repository import (
    init_db,
    save_ensemble_market_selection_run,
    set_active_ensemble_market_selection,
    get_active_ensemble_market_tuning_run,
)
from pipelines.ensemble_tuning import (
    EnsembleDataset,
    GamesDatasetMetadata,
    MarketPredictionBundle,
    EnsembleTuningResult,
    run_market_ensemble_selection,
    resolve_games_df,
)
from pipelines.market_tuning import run_ensemble_market_tuning
from pipelines.schedule import _resolve_ensemble_weights


def _create_game_row(date, home, away, home_score, away_score, game_id):
    return (date, home, away, home_score, away_score, 0, 0, None, game_id, "nba", "2024-25", None, None, None)


def _make_predictions():
    rows = {
        "elo": pd.DataFrame(
            [
                {"game_key": "g1", "game_id": "g1", "model_name": "elo", "pred_value": 0.2, "target_value": 0.0},
                {"game_key": "g2", "game_id": "g2", "model_name": "elo", "pred_value": 0.9, "target_value": 1.0},
            ]
        ),
        "bradley-terry": pd.DataFrame(
            [
                {"game_key": "g1", "game_id": "g1", "model_name": "bradley-terry", "pred_value": 0.9, "target_value": 0.0},
                {"game_key": "g2", "game_id": "g2", "model_name": "bradley-terry", "pred_value": 0.1, "target_value": 1.0},
            ]
        ),
    }
    metadata = GamesDatasetMetadata(
        scorable_games=2,
        date_min="2024-01-01",
        date_max="2024-01-02",
        asof="2024-01-02",
        data_source="db",
        db_path="db",
        csv_path=None,
    )
    return MarketPredictionBundle(models=["elo", "bradley-terry"], predictions=rows, metadata=metadata)


def test_resolve_games_df_filters_completed_games(tmp_path):
    db_path = tmp_path / "games.db"
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    rows = [
        _create_game_row("2024-01-01", "A", "B", 10, 5, "g1"),
        _create_game_row("2024-01-02", "C", "D", None, 7, "g2"),
        _create_game_row("2024-01-03", "E", "F", 8, 9, "g3"),
    ]
    conn.executemany(
        """
        INSERT INTO games (
            date, home_team, away_team, home_score, away_score,
            neutral, overtime, decision_type, game_id, sport, season, division, conference, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()

    df, metadata = resolve_games_df(
        sport="nba",
        season="2024-25",
        start_date="2024-01-01",
        end_date="2024-01-03",
        as_of_date="2024-01-02",
        db_path=str(db_path),
    )
    assert len(df) == 1
    assert metadata.scorable_games == 1
    assert metadata.date_min == "2024-01-01"
    assert metadata.date_max == "2024-01-01"


def test_selection_activation_only_on_improvement(monkeypatch, tmp_path):
    db_path = tmp_path / "selection.db"
    init_db(db_path)

    def fake_collect(*args, **kwargs):
        return _make_predictions()

    monkeypatch.setattr("pipelines.ensemble_tuning._collect_market_predictions", fake_collect)
    selection1, activated1 = run_market_ensemble_selection(
        sport="nba",
        season="2024-25",
        market="ML",
        ensemble_id="ensemble_ml_v1",
        start_date=None,
        end_date=None,
        as_of_date=None,
        candidates=["elo", "bradley-terry"],
        min_coverage=0.5,
        max_members=2,
        epsilon=1e-4,
        activate=True,
        notes=None,
        csv_path=None,
        db_path=str(db_path),
    )
    assert activated1

    selection2, activated2 = run_market_ensemble_selection(
        sport="nba",
        season="2024-25",
        market="ML",
        ensemble_id="ensemble_ml_v1",
        start_date=None,
        end_date=None,
        as_of_date=None,
        candidates=["elo", "bradley-terry"],
        min_coverage=0.5,
        max_members=2,
        epsilon=10.0,
        activate=True,
        notes=None,
        csv_path=None,
        db_path=str(db_path),
    )
    assert not activated2


def test_tuning_prefers_active_selection(monkeypatch, tmp_path):
    db_path = tmp_path / "tuning.db"
    init_db(db_path)
    selection_run_id = "sel-run"
    save_ensemble_market_selection_run(
        db_path=str(db_path),
        sport="nba",
        season="2024-25",
        market="ML",
        ensemble_id="ensemble_ml_v1",
        run_id=selection_run_id,
        window_start="2024-01-01",
        window_end="2024-01-02",
        asof="2024-01-02",
        data_source="db",
        dataset_db_path=str(db_path),
        csv_path=None,
        scorable_games=2,
        date_min="2024-01-01",
        date_max="2024-01-02",
        candidates=["elo", "bradley-terry"],
        selected=["elo", "bradley-terry"],
        objective_metric="log_loss",
        summary={"final_score": 0.2},
        baseline_score=0.4,
        notes=None,
    )
    set_active_ensemble_market_selection(
        db_path=str(db_path),
        sport="nba",
        season="2024-25",
        market="ML",
        ensemble_id="ensemble_ml_v1",
        active_run_id=selection_run_id,
    )

    captured = {}

    def fake_tune(**kwargs):
        captured["models"] = list(kwargs.get("models"))
        dataset = EnsembleDataset(
            pred_matrix=np.array([[0.2, 0.8]]),
            targets=np.array([1.0]),
            models=list(kwargs.get("models")),
            game_keys=["g1"],
        )
        metadata = GamesDatasetMetadata(
            scorable_games=1,
            date_min="2024-01-01",
            date_max="2024-01-01",
            asof="2024-01-01",
            data_source="db",
            db_path=str(db_path),
            csv_path=None,
        )
        return EnsembleTuningResult(
            weights={"elo": 0.5, "bradley-terry": 0.5},
            artifact_path=Path("dummy"),
            games=1,
            models=list(kwargs.get("models")),
            best_score=0.0,
            metric_optimized="log_loss",
            summary_metrics={"log_loss": 0.0},
            metadata=metadata,
            dataset=dataset,
        )

    monkeypatch.setattr("pipelines.market_tuning.tune_market_ensemble", fake_tune)

    result = run_ensemble_market_tuning(
        sport="nba",
        season="2024-25",
        market="ML",
        ensemble_id="ensemble_ml_v1",
        start_date="2024-01-01",
        end_date="2024-01-02",
        as_of_date=None,
        selection_models=["elo", "bradley-terry"],
        selection_run_id=selection_run_id,
        csv_path=None,
        db_path=str(db_path),
    )
    assert captured["models"] == ["elo", "bradley-terry"]
    assert result.selection_run_id == selection_run_id
    assert result.activated is True
    assert get_active_ensemble_market_tuning_run(
        str(db_path),
        sport="nba",
        season="2024-25",
        market="ML",
        ensemble_id="ensemble_ml_v1",
    )["active_run_id"] == result.run_id


def test_schedule_mismatch_ignores_tuning():
    selection_context = {"run_id": "selection-1", "models": ["elo", "bradley-terry"]}
    tuning_context = {
        "run_id": "tuning-1",
        "weights": {"elo": 0.5, "bradley-terry": 0.5},
        "models": ["elo", "bradley-terry", "gamma"],
        "selection_run_id": "selection-2",
        "selection_models": ["elo", "gamma"],
    }
    warnings = []
    weights, models, source, run_id, selection_run_id = _resolve_ensemble_weights(
        db_path=None,
        sport="nba",
        season="2024-25",
        market="ML",
        ensemble_id="ensemble_ml_v1",
        config_weights=None,
        selection_context=selection_context,
        tuning_context=tuning_context,
        config_warnings=warnings,
    )
    assert math.isclose(weights["elo"] + weights["bradley-terry"], 1.0)
    assert warnings
    assert "selection mismatch" in warnings[0]


def test_tuning_enforces_weight_normalization(monkeypatch, tmp_path):
    db_path = tmp_path / "tuning.db"
    init_db(db_path)
    selection_run_id = "sel-run"
    save_ensemble_market_selection_run(
        db_path=str(db_path),
        sport="nba",
        season="2024-25",
        market="ML",
        ensemble_id="ensemble_ml_v1",
        run_id=selection_run_id,
        window_start="2024-01-01",
        window_end="2024-01-02",
        asof="2024-01-02",
        data_source="db",
        dataset_db_path=str(db_path),
        csv_path=None,
        scorable_games=2,
        date_min="2024-01-01",
        date_max="2024-01-02",
        candidates=["elo", "bradley-terry"],
        selected=["elo", "bradley-terry"],
        objective_metric="log_loss",
        summary={"final_score": 0.2},
        baseline_score=0.4,
        notes=None,
    )
    set_active_ensemble_market_selection(
        db_path=str(db_path),
        sport="nba",
        season="2024-25",
        market="ML",
        ensemble_id="ensemble_ml_v1",
        active_run_id=selection_run_id,
    )

    def fake_tune(**kwargs):
        dataset = EnsembleDataset(
            pred_matrix=np.array([[0.2, 0.8]]),
            targets=np.array([1.0]),
            models=list(kwargs.get("models")),
            game_keys=["g1"],
        )
        metadata = GamesDatasetMetadata(
            scorable_games=1,
            date_min="2024-01-01",
            date_max="2024-01-01",
            asof="2024-01-01",
            data_source="db",
            db_path=str(db_path),
            csv_path=None,
        )
        return EnsembleTuningResult(
            weights={"elo": 0.3, "bradley-terry": 0.3},
            artifact_path=Path("dummy"),
            games=1,
            models=list(kwargs.get("models")),
            best_score=0.0,
            metric_optimized="log_loss",
            summary_metrics={"log_loss": 0.0},
            metadata=metadata,
            dataset=dataset,
        )

    monkeypatch.setattr("pipelines.market_tuning.tune_market_ensemble", fake_tune)

    with pytest.raises(ValueError):
        run_ensemble_market_tuning(
            sport="nba",
            season="2024-25",
            market="ML",
            ensemble_id="ensemble_ml_v1",
            start_date="2024-01-01",
            end_date="2024-01-02",
            as_of_date=None,
            selection_models=None,
            selection_run_id=None,
            csv_path=None,
            db_path=str(db_path),
        )
