import json
from pathlib import Path

from data.repository import (
    init_db,
    save_model_market_tuning_run,
)
from pipelines.model_params import resolve_active_model_market_params
from pipelines.schedule import _collect_market_param_sources


def test_resolver_auto_selects_best_when_no_active(tmp_path):
    db = tmp_path / "test.db"
    db_path = str(db)
    # initialize DB
    init_db(db_path)
    # insert a tuning run for model+market
    run_id = "run-123"
    params = {"k_factor": 42.0}
    save_model_market_tuning_run(
        db_path,
        sport="nhl",
        season="2025-26",
        model="elo",
        market="ML",
        metric_optimized="backtest_log_loss",
        run_id=run_id,
        best_score=0.5,
        best_params_json=json.dumps(params),
        summary_metrics_json=None,
        started_at=None,
        finished_at=None,
    )

    resolved = resolve_active_model_market_params(
        db_path=db_path, sport="nhl", season="2025-26", model="elo", market="ML"
    )
    assert resolved.params is not None
    assert resolved.params.get("k_factor") == 42.0
    assert resolved.source_run_id == run_id


def test_collect_market_param_sources_includes_tuned_run(tmp_path):
    db = tmp_path / "test2.db"
    db_path = str(db)
    init_db(db_path)
    run_id = "run-456"
    params = {"k_factor": 11.0}
    save_model_market_tuning_run(
        db_path,
        sport="nhl",
        season="2025-26",
        model="elo",
        market="ML",
        metric_optimized="backtest_log_loss",
        run_id=run_id,
        best_score=0.4,
        best_params_json=json.dumps(params),
        summary_metrics_json=None,
        started_at=None,
        finished_at=None,
    )

    sources = _collect_market_param_sources(db_path=db_path, sport="nhl", season="2025-26", models=["elo"])  # type: ignore[arg-type]
    # Expect a source entry for ML->elo that references the db selection
    assert "ML" in sources
    assert "elo" in sources["ML"]
    assert sources["ML"]["elo"] is not None