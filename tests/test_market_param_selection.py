from __future__ import annotations

import json
from pathlib import Path

from data.repository import (
    init_db,
    save_model_market_tuning_run,
    set_active_model_market_params,
)
from pipelines.market_tuning import _resolve_market_metric
from pipelines.schedule import _collect_market_param_sources


def test_spread_uses_best_run_when_no_market_params(tmp_path: Path) -> None:
    db_path = tmp_path / "params.db"
    init_db(db_path)
    # record the best run that optimizes the spread market
    _, metric_optimized = _resolve_market_metric("SPREAD", None)
    save_model_market_tuning_run(
        db_path,
        sport="nba",
        season="2024-25",
        model="elo",
        market="SPREAD",
        metric_optimized=metric_optimized,
        run_id="run-xx",
        best_score=0.1,
        best_params_json=json.dumps({"k_factor": 42}),
        summary_metrics_json=None,
        started_at=None,
        finished_at=None,
        notes=None,
    )

    sources = _collect_market_param_sources(
        db_path=db_path, sport="nba", season="2024-25", models=["elo"]
    )

    assert "SPREAD" in sources
    assert sources["SPREAD"]["elo"] == "db_market_best_run/mae_margin"


def test_db_market_overrides_best_run(tmp_path: Path) -> None:
    db_path = tmp_path / "params.db"
    init_db(db_path)
    # save a best run entry for the SPREAD metric
    _, metric_optimized = _resolve_market_metric("SPREAD", None)
    save_model_market_tuning_run(
        db_path,
        sport="nba",
        season="2024-25",
        model="elo",
        market="SPREAD",
        metric_optimized=metric_optimized,
        run_id="run-xx",
        best_score=0.1,
        best_params_json=json.dumps({"k_factor": 42}),
        summary_metrics_json=None,
        started_at=None,
        finished_at=None,
        notes=None,
    )
    # set an explicit market-specific active params (should take precedence)
    set_active_model_market_params(
        db_path,
        sport="nba",
        season="2024-25",
        model="elo",
        market="SPREAD",
        params={"k_factor": 99},
        source_run_id="run-xx",
    )

    sources = _collect_market_param_sources(
        db_path=db_path, sport="nba", season="2024-25", models=["elo"]
    )

    assert sources["SPREAD"]["elo"] == "db_market_active/mae_margin"
