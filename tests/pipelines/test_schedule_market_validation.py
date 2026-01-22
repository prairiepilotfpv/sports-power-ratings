from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from data.repository import (
    init_db,
    save_games,
    save_model_market_tuning_run,
    set_active_model_market_params,
)
from ingest.schema import GameResult
from pipelines.market_config import (
    resolve_market_params,
    get_market_ensemble_spec,
)
from pipelines.market_tuning import _resolve_market_metric
from pipelines.schedule import _validate_market_tuning_inputs


def test_validate_market_tuning_inputs_strict_fails(tmp_path: Path) -> None:
    db_path = tmp_path / "params.db"
    init_db(db_path)

    with pytest.raises(ValueError) as excinfo:
        _validate_market_tuning_inputs(
            db_path=db_path,
            sport="nba",
            season="2024-25",
            models=["elo"],
            ensemble_ids={},
            strict=True,
        )

    message = str(excinfo.value)
    assert "bootstrap-market-actives" in message


def test_validate_market_tuning_inputs_non_strict_warns(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "params.db"
    init_db(db_path)

    _validate_market_tuning_inputs(
        db_path=db_path,
        sport="nba",
        season="2024-25",
        models=["elo"],
        ensemble_ids={},
        strict=False,
    )

    output = capsys.readouterr().out
    assert "Missing active params for model=elo market=ML" in output
    assert "bootstrap-market-actives" in output


class TestScheduleMarketIsolationContract:
    """Integration tests verifying schedule output uses correct market params."""

    @pytest.fixture
    def db_with_market_params(self, tmp_path: Path) -> Path:
        """Create a DB with market-specific params for elo model."""
        db_path = tmp_path / "games.db"
        init_db(db_path)

        # Create some games
        games = [
            GameResult(
                date=date(2024, 1, 1),
                home_team="Team A",
                away_team="Team B",
                home_score=100,
                away_score=90,
                sport="nba",
                season="2024-25",
            ),
            GameResult(
                date=date(2024, 1, 2),
                home_team="Team B",
                away_team="Team C",
                home_score=95,
                away_score=110,
                sport="nba",
                season="2024-25",
            ),
        ]
        save_games(db_path, games)

        # Create distinct params for each market to trace provenance
        market_params = {
            "ML": {"k_factor": 10.0, "_market_tag": "ML"},
            "SPREAD": {"k_factor": 20.0, "_market_tag": "SPREAD"},
            "TOTAL": {"k_factor": 30.0, "_market_tag": "TOTAL"},
        }

        for market, params in market_params.items():
            _, metric_optimized = _resolve_market_metric(market, None)
            run_id = f"run-{market.lower()}-elo"
            save_model_market_tuning_run(
                db_path,
                sport="nba",
                season="2024-25",
                model="elo",
                market=market,
                metric_optimized=metric_optimized,
                run_id=run_id,
                best_score=1.0,
                best_params_json=json.dumps(params),
                summary_metrics_json=None,
                started_at=None,
                finished_at=None,
                notes=None,
            )
            set_active_model_market_params(
                db_path,
                sport="nba",
                season="2024-25",
                model="elo",
                market=market,
                params=params,
                source_run_id=run_id,
                params_source="tuned",
                metric_optimized=metric_optimized,
            )

        return db_path

    def test_market_params_resolve_correctly(self, db_with_market_params: Path) -> None:
        """Verify each market resolves to its own params."""
        db_path = db_with_market_params

        # ML should get ML params
        ml_res = resolve_market_params(
            db_path=db_path,
            sport="nba",
            season="2024-25",
            model="elo",
            market="ML",
        )
        assert ml_res.params is not None
        assert ml_res.params.get("_market_tag") == "ML"
        assert ml_res.params.get("k_factor") == 10.0

        # SPREAD should get SPREAD params
        spread_res = resolve_market_params(
            db_path=db_path,
            sport="nba",
            season="2024-25",
            model="elo",
            market="SPREAD",
        )
        assert spread_res.params is not None
        assert spread_res.params.get("_market_tag") == "SPREAD"
        assert spread_res.params.get("k_factor") == 20.0

        # TOTAL should get TOTAL params
        total_res = resolve_market_params(
            db_path=db_path,
            sport="nba",
            season="2024-25",
            model="elo",
            market="TOTAL",
        )
        assert total_res.params is not None
        assert total_res.params.get("_market_tag") == "TOTAL"
        assert total_res.params.get("k_factor") == 30.0

    def test_ensemble_specs_are_market_specific(self, db_with_market_params: Path) -> None:
        """Verify ensemble specs are configured independently per market."""
        db_path = db_with_market_params

        ml_spec = get_market_ensemble_spec(
            db_path=db_path,
            sport="nba",
            season="2024-25",
            market="ML",
        )
        spread_spec = get_market_ensemble_spec(
            db_path=db_path,
            sport="nba",
            season="2024-25",
            market="SPREAD",
        )
        total_spec = get_market_ensemble_spec(
            db_path=db_path,
            sport="nba",
            season="2024-25",
            market="TOTAL",
        )

        # Each should have the correct metric
        assert ml_spec.metric_slot == "log_loss"
        assert spread_spec.metric_slot == "mae_margin"
        assert total_spec.metric_slot == "mae_total"

        # Each should have distinct ensemble IDs
        assert ml_spec.ensemble_id == "ensemble_ml_v1"
        assert spread_spec.ensemble_id == "ensemble_spread_v1"
        assert total_spec.ensemble_id == "ensemble_total_v1"

    def test_market_metric_never_cross_pollinated(self, db_with_market_params: Path) -> None:
        """The metric_optimized field should always match the market."""
        db_path = db_with_market_params

        for market, expected_metric_suffix in [
            ("ML", "log_loss"),
            ("SPREAD", "mae_margin"),
            ("TOTAL", "mae_total"),
        ]:
            res = resolve_market_params(
                db_path=db_path,
                sport="nba",
                season="2024-25",
                model="elo",
                market=market,
            )
            # The metric_optimized should contain the market's metric
            assert res.metric_optimized is not None, f"No metric for {market}"
            assert expected_metric_suffix in res.metric_optimized, (
                f"Market {market} has wrong metric: {res.metric_optimized}"
            )
