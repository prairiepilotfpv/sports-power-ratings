from __future__ import annotations

import json
from pathlib import Path

from data.repository import (
    get_active_model_market_params,
    get_active_model_market_params_source,
    init_db,
    save_model_market_tuning_run,
    save_tuned_params,
    set_active_model_market_params,
)
from pipelines.market_tuning import _resolve_market_metric
from pipelines.model_params import bootstrap_market_active_params


def test_bootstrap_market_active_params_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "params.db"
    init_db(db_path)

    set_active_model_market_params(
        db_path,
        sport="nba",
        season="2024-25",
        model="elo",
        market="SPREAD",
        params={"k_factor": 11},
        source_run_id="manual",
    )

    _, metric_optimized = _resolve_market_metric("TOTAL", None)
    save_model_market_tuning_run(
        db_path,
        sport="nba",
        season="2024-25",
        model="elo",
        market="TOTAL",
        metric_optimized=metric_optimized,
        run_id="run-total-1",
        best_score=1.0,
        best_params_json=json.dumps({"a": 1}),
        summary_metrics_json=None,
        started_at=None,
        finished_at=None,
        notes=None,
    )
    save_tuned_params(
        db_path,
        sport="nba",
        season="2024-25",
        model="elo",
        metric="mae_margin",
        run_id="run-margin-1",
        params_json=json.dumps({"b": 2}),
        best_score=0.1,
    )
    save_tuned_params(
        db_path,
        sport="nba",
        season="2024-25",
        model="elo",
        metric="mae_total",
        run_id="run-total-2",
        params_json=json.dumps({"c": 3}),
        best_score=0.2,
    )

    summary = bootstrap_market_active_params(
        db_path=db_path,
        sport="nba",
        season="2024-25",
        models=["elo"],
        include_ml=True,
    )

    assert get_active_model_market_params(
        db_path,
        sport="nba",
        season="2024-25",
        model="elo",
        market="SPREAD",
    ) == {"k_factor": 11}
    assert (
        get_active_model_market_params_source(
            db_path,
            sport="nba",
            season="2024-25",
            model="elo",
            market="SPREAD",
        )
        == "manual"
    )
    assert get_active_model_market_params(
        db_path,
        sport="nba",
        season="2024-25",
        model="elo",
        market="TOTAL",
    ) == {"a": 1}
    assert (
        get_active_model_market_params_source(
            db_path,
            sport="nba",
            season="2024-25",
            model="elo",
            market="TOTAL",
        )
        == "run-total-1"
    )
    assert get_active_model_market_params(
        db_path,
        sport="nba",
        season="2024-25",
        model="elo",
        market="ML",
    ) == {}
    assert (
        get_active_model_market_params_source(
            db_path,
            sport="nba",
            season="2024-25",
            model="elo",
            market="ML",
        )
        == "default"
    )

    assert summary["counts"]["created_from_best_run"] == 1
    assert summary["counts"]["created_from_model_metric"] == 0
    assert summary["counts"]["created_default"] == 1
    assert summary["counts"]["skipped_existing"] == 1

    summary_repeat = bootstrap_market_active_params(
        db_path=db_path,
        sport="nba",
        season="2024-25",
        models=["elo"],
        include_ml=True,
    )
    assert summary_repeat["counts"]["created_from_best_run"] == 0
    assert summary_repeat["counts"]["created_from_model_metric"] == 0
    assert summary_repeat["counts"]["created_default"] == 0
    assert summary_repeat["counts"]["skipped_existing"] == 3


def test_bootstrap_market_active_params_metric_fallback(tmp_path: Path) -> None:
    db_path = tmp_path / "params.db"
    init_db(db_path)

    save_tuned_params(
        db_path,
        sport="nba",
        season="2024-25",
        model="elo",
        metric="mae_margin",
        run_id="run-margin-1",
        params_json=json.dumps({"k_factor": 22}),
        best_score=0.1,
    )

    bootstrap_market_active_params(
        db_path=db_path,
        sport="nba",
        season="2024-25",
        models=["elo"],
        include_ml=False,
    )

    assert get_active_model_market_params(
        db_path,
        sport="nba",
        season="2024-25",
        model="elo",
        market="SPREAD",
    ) == {"k_factor": 22}
    assert (
        get_active_model_market_params_source(
            db_path,
            sport="nba",
            season="2024-25",
            model="elo",
            market="SPREAD",
        )
        == "model_tuned_params:mae_margin"
    )
