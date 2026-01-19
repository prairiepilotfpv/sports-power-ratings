from __future__ import annotations

import json

import pandas as pd
import pytest

from data.repository import (
    save_model_market_tuning_run,
    set_active_model_market_params,
    upsert_model_tuned_params,
)
from pipelines.model_params import activate_best_params, resolve_effective_params
from pipelines.schedule import _project_row


def test_activate_best_params_sets_tuned_active(tmp_path) -> None:
    db_path = tmp_path / "active.db"
    sport = "nba"
    season = "2025-26"
    model = "elo"
    market = "ML"
    params = {"k_factor": 33.0}
    run_id = "run-abc"

    save_model_market_tuning_run(
        db_path,
        sport=sport,
        season=season,
        model=model,
        market=market,
        metric_optimized="backtest_log_loss",
        run_id=run_id,
        best_score=0.12,
        best_params_json=json.dumps(params, sort_keys=True),
        summary_metrics_json=None,
        started_at=None,
        finished_at=None,
        notes=None,
    )

    activate_best_params(
        db_path=db_path,
        sport=sport,
        season=season,
        model=model,
        market=market,
        run_id=run_id,
        best_params=params,
        best_score=0.12,
        metric_optimized="backtest_log_loss",
    )

    resolved = resolve_effective_params(
        db_path=db_path,
        sport=sport,
        season=season,
        model=model,
        market=market,
    )
    assert resolved.params == params
    assert resolved.params_source_label == "tuned_active"
    assert resolved.metric_optimized == "backtest_log_loss"
    assert resolved.best_score == pytest.approx(0.12)
    assert resolved.params_nonempty is True
    assert resolved.params_fingerprint


def test_effective_params_classification_default_and_legacy(tmp_path, caplog) -> None:
    db_path = tmp_path / "classify.db"
    sport = "nba"
    season = "2025-26"
    model = "bradley-terry"

    set_active_model_market_params(
        db_path,
        sport=sport,
        season=season,
        model=model,
        market="ML",
        params={},
        source_run_id="default",
        params_source="default",
        metric_optimized="backtest_log_loss",
    )
    res_default = resolve_effective_params(
        db_path=db_path,
        sport=sport,
        season=season,
        model=model,
        market="ML",
    )
    assert res_default.params_source_label == "default_active"
    assert res_default.params_nonempty is False

    legacy_params = {"temp": 2.0}
    upsert_model_tuned_params(
        db_path,
        sport=sport,
        season=season,
        model=model,
        metric="mae_margin",
        run_id="legacy-run",
        params=legacy_params,
        best_score=1.23,
    )
    set_active_model_market_params(
        db_path,
        sport=sport,
        season=season,
        model=model,
        market="SPREAD",
        params={},
        source_run_id="model_tuned_params:mae_margin",
        params_source="legacy",
        metric_optimized="backtest_mae_margin",
    )
    res_legacy = resolve_effective_params(
        db_path=db_path,
        sport=sport,
        season=season,
        model=model,
        market="SPREAD",
    )
    assert res_legacy.params_source_label == "legacy_active"
    assert res_legacy.params == legacy_params

    save_model_market_tuning_run(
        db_path,
        sport=sport,
        season=season,
        model=model,
        market="TOTAL",
        metric_optimized="backtest_mae_total",
        run_id="run-missing",
        best_score=0.5,
        best_params_json=json.dumps({"lambda": 1.0}, sort_keys=True),
        summary_metrics_json=None,
        started_at=None,
        finished_at=None,
        notes=None,
    )
    with caplog.at_level("WARNING"):
        res_missing = resolve_effective_params(
            db_path=db_path,
            sport=sport,
            season=season,
            model=model,
            market="TOTAL",
        )
    assert res_missing.params_source_label == "db_market_best_run"
    assert any(
        "Tuned params exist but are not active" in record.message for record in caplog.records
    )


def test_project_row_sets_provenance_columns() -> None:
    row = pd.Series(
        {
            "date": "2024-01-01",
            "home_team": "Home",
            "away_team": "Away",
            "home_score": None,
            "away_score": None,
            "neutral": False,
            "game_id": "gid-1",
        }
    )
    ratings = {"Home": 1.0, "Away": 0.0}

    def _projection_engine(home, away, model_instance, context):
        return {
            "model_p_home_win": 0.6,
            "margin_mean": 2.5,
            "total_mean": 200.0,
        }

    projected = _project_row(
        row,
        ratings=ratings,
        status="scheduled",
        home_advantage=0.0,
        params_source="tuned_active",
        params_source_label="tuned_active",
        params_source_run_id="run-xyz",
        tuned_metric_used="log_loss",
        params_metric_optimized="backtest_log_loss",
        params_best_score=0.2,
        params_fingerprint="abc123",
        params_nonempty=True,
        params_run_id="run-xyz",
        params_market="ML",
        model_instance=None,
        projection_engine=_projection_engine,
        projection_context={
            "ratings": ratings,
            "base_total": 0.0,
            "scoring_averages": {},
            "total_intercept": 0.0,
            "total_slope": 0.0,
            "margin_std": None,
            "total_std": None,
            "conditional_sd_intercept": None,
            "conditional_sd_slope": None,
            "win_prob_k": 1.0,
            "rating_units": "points",
        },
    )

    assert projected["params_source_label"] == "tuned_active"
    assert projected["params_source_run_id"] == "run-xyz"
    assert projected["params_metric_optimized"] == "backtest_log_loss"
    assert projected["params_best_score"] == 0.2
    assert projected["params_fingerprint"] == "abc123"
    assert projected["params_nonempty"] is True
