from __future__ import annotations

import json
from pathlib import Path

from data.repository import init_db, save_tuned_params, set_active_model_market_params
from pipelines.schedule import _collect_market_param_sources


def test_spread_uses_metric_tuned_params_when_no_market_params(tmp_path: Path) -> None:
    db_path = tmp_path / "params.db"
    init_db(db_path)
    # save tuned params for mae_margin (SPREAD)
    save_tuned_params(
        db_path,
        sport="nba",
        season="2024-25",
        model="elo",
        metric="mae_margin",
        run_id="run-xx",
        params_json=json.dumps({"k_factor": 42}),
        best_score=0.1,
    )

    sources = _collect_market_param_sources(
        db_path=db_path, sport="nba", season="2024-25", models=["elo"]
    )

    assert "SPREAD" in sources
    assert sources["SPREAD"]["elo"] is None


def test_db_market_overrides_db_metric(tmp_path: Path) -> None:
    db_path = tmp_path / "params.db"
    init_db(db_path)
    # save tuned params for mae_margin (SPREAD)
    save_tuned_params(
        db_path,
        sport="nba",
        season="2024-25",
        model="elo",
        metric="mae_margin",
        run_id="run-xx",
        params_json=json.dumps({"k_factor": 42}),
        best_score=0.1,
    )
    # set an explicit market-specific active params (should take precedence)
    set_active_model_market_params(
        db_path,
        sport="nba",
        season="2024-25",
        model="elo",
        market="SPREAD",
        params={"k_factor": 99},
        source_run_id="run-market-123",
    )

    sources = _collect_market_param_sources(
        db_path=db_path, sport="nba", season="2024-25", models=["elo"]
    )

    assert sources["SPREAD"]["elo"] == "db_market_active"
