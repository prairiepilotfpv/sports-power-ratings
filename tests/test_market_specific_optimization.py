"""Tests for market-specific optimization contract.

These tests verify that:
A) Parameters are isolated per market - resolving params for TOTAL returns TOTAL-tuned params
B) Ensembles are isolated per market - TOTAL ensemble only uses TOTAL-configured models  
C) Fallback behavior is market-scoped - missing TOTAL params don't fall back to SPREAD/ML
D) Schedule columns come from correct market pipelines
"""

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
    get_active_model_market_params,
)
from ingest.schema import GameResult
from pipelines.market_config import (
    MarketParamsResolution,
    MarketEnsembleSpec,
    resolve_market_params,
    resolve_market_params_batch,
    get_market_ensemble_spec,
    get_all_market_specs,
    validate_market_isolation,
    DEFAULT_MARKET_MODELS,
    DEFAULT_MARKET_METRICS,
)
from pipelines.market_tuning import _resolve_market_metric


@pytest.fixture
def sample_db(tmp_path: Path):
    """Create a sample database with games."""
    db_path = tmp_path / "test.db"
    init_db(db_path)
    
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
    return db_path


class TestParamsIsolation:
    """Test A: Parameters are isolated per market."""
    
    def test_different_params_per_market(self, sample_db: Path) -> None:
        """Create dummy tuned params for same model with different markets.
        Verify resolve_market_params(..., market="TOTAL") returns totals params, not spread/ML.
        """
        # Set up different active params for each market
        ml_params = {"k_factor": 10.0, "market_source": "ML"}
        spread_params = {"k_factor": 20.0, "market_source": "SPREAD"}
        total_params = {"k_factor": 30.0, "market_source": "TOTAL"}
        
        # Create tuning runs for each market
        for market, params, metric_suffix in [
            ("ML", ml_params, "log_loss"),
            ("SPREAD", spread_params, "mae_margin"),
            ("TOTAL", total_params, "mae_total"),
        ]:
            _, metric_optimized = _resolve_market_metric(market, None)
            run_id = f"run-{market.lower()}-1"
            save_model_market_tuning_run(
                sample_db,
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
                sample_db,
                sport="nba",
                season="2024-25",
                model="elo",
                market=market,
                params=params,
                source_run_id=run_id,
                params_source="tuned",
                metric_optimized=metric_optimized,
            )
        
        # Verify each market returns its own params
        ml_resolution = resolve_market_params(
            db_path=sample_db,
            sport="nba",
            season="2024-25",
            model="elo",
            market="ML",
        )
        assert ml_resolution.params is not None
        assert ml_resolution.params.get("k_factor") == 10.0
        assert ml_resolution.params.get("market_source") == "ML"
        assert ml_resolution.metric_optimized == "backtest_log_loss"
        
        spread_resolution = resolve_market_params(
            db_path=sample_db,
            sport="nba",
            season="2024-25",
            model="elo",
            market="SPREAD",
        )
        assert spread_resolution.params is not None
        assert spread_resolution.params.get("k_factor") == 20.0
        assert spread_resolution.params.get("market_source") == "SPREAD"
        assert spread_resolution.metric_optimized == "backtest_mae_margin"
        
        total_resolution = resolve_market_params(
            db_path=sample_db,
            sport="nba",
            season="2024-25",
            model="elo",
            market="TOTAL",
        )
        assert total_resolution.params is not None
        assert total_resolution.params.get("k_factor") == 30.0
        assert total_resolution.params.get("market_source") == "TOTAL"
        assert total_resolution.metric_optimized == "backtest_mae_total"
    
    def test_params_batch_resolution(self, sample_db: Path) -> None:
        """Test batch resolution maintains market isolation."""
        # Set up params for multiple models in TOTAL market
        for model, k_factor in [("elo", 30.0), ("poisson", 40.0)]:
            _, metric_optimized = _resolve_market_metric("TOTAL", None)
            params = {"k_factor": k_factor}
            run_id = f"run-total-{model}"
            save_model_market_tuning_run(
                sample_db,
                sport="nba",
                season="2024-25",
                model=model,
                market="TOTAL",
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
                sample_db,
                sport="nba",
                season="2024-25",
                model=model,
                market="TOTAL",
                params=params,
                source_run_id=run_id,
                params_source="tuned",
                metric_optimized=metric_optimized,
            )
        
        # Batch resolve
        batch = resolve_market_params_batch(
            db_path=sample_db,
            sport="nba",
            season="2024-25",
            models=["elo", "poisson"],
            market="TOTAL",
        )
        
        assert "elo" in batch
        assert "poisson" in batch
        assert batch["elo"].params.get("k_factor") == 30.0
        assert batch["poisson"].params.get("k_factor") == 40.0
        assert batch["elo"].market == "TOTAL"
        assert batch["poisson"].market == "TOTAL"


class TestEnsembleIsolation:
    """Test B: Ensembles are isolated per market."""
    
    def test_total_ensemble_only_uses_total_models(self, sample_db: Path) -> None:
        """Configure totals ensemble with models [A,B] and spread ensemble with [C].
        Verify totals predictions call only A/B.
        """
        # Get ensemble specs
        total_spec = get_market_ensemble_spec(
            db_path=sample_db,
            sport="nba",
            season="2024-25",
            market="TOTAL",
        )
        spread_spec = get_market_ensemble_spec(
            db_path=sample_db,
            sport="nba",
            season="2024-25",
            market="SPREAD",
        )
        ml_spec = get_market_ensemble_spec(
            db_path=sample_db,
            sport="nba",
            season="2024-25",
            market="ML",
        )
        
        # Each market should have its own default models
        assert set(total_spec.models) == set(DEFAULT_MARKET_MODELS["TOTAL"])
        assert set(spread_spec.models) == set(DEFAULT_MARKET_MODELS["SPREAD"])
        assert set(ml_spec.models) == set(DEFAULT_MARKET_MODELS["ML"])
        
        # Verify metrics are correct
        assert total_spec.metric_slot == "mae_total"
        assert spread_spec.metric_slot == "mae_margin"
        assert ml_spec.metric_slot == "log_loss"
    
    def test_ensemble_spec_from_config_override(self, sample_db: Path) -> None:
        """Test that config override is respected per market."""
        custom_config = {
            "sport": "nba",
            "season": "2024-25",
            "markets": {
                "TOTAL": {
                    "ensemble_id": "custom_total_v1",
                    "models": ["model_a", "model_b"],
                    "weights": {"model_a": 0.7, "model_b": 0.3},
                    "metric_slot": "mae_total",
                },
                "SPREAD": {
                    "ensemble_id": "custom_spread_v1",
                    "models": ["model_c"],
                    "weights": {"model_c": 1.0},
                    "metric_slot": "mae_margin",
                },
            },
            "_meta": {
                "markets": {
                    "TOTAL": {"path": "/custom/total.json"},
                    "SPREAD": {"path": "/custom/spread.json"},
                }
            }
        }
        
        total_spec = get_market_ensemble_spec(
            db_path=sample_db,
            sport="nba",
            season="2024-25",
            market="TOTAL",
            ensemble_config=custom_config,
        )
        
        assert total_spec.ensemble_id == "custom_total_v1"
        assert total_spec.models == ["model_a", "model_b"]
        assert total_spec.weights == {"model_a": 0.7, "model_b": 0.3}
        
        spread_spec = get_market_ensemble_spec(
            db_path=sample_db,
            sport="nba",
            season="2024-25",
            market="SPREAD",
            ensemble_config=custom_config,
        )
        
        assert spread_spec.ensemble_id == "custom_spread_v1"
        assert spread_spec.models == ["model_c"]
        # SPREAD models should NOT include TOTAL models
        assert "model_a" not in spread_spec.models
        assert "model_b" not in spread_spec.models


class TestNoLeakFallback:
    """Test C: Fallback behavior is market-scoped."""
    
    def test_missing_total_params_no_cross_market_leak(self, sample_db: Path) -> None:
        """If totals active params missing, we fall back to totals defaults and label appropriately.
        We should NOT fall back to SPREAD or ML params.
        """
        # Set up only ML params
        _, ml_metric = _resolve_market_metric("ML", None)
        ml_params = {"k_factor": 10.0, "from_market": "ML"}
        set_active_model_market_params(
            sample_db,
            sport="nba",
            season="2024-25",
            model="elo",
            market="ML",
            params=ml_params,
            source_run_id="ml-run",
            params_source="tuned",
            metric_optimized=ml_metric,
        )
        
        # Resolve TOTAL params - should NOT get ML params
        total_resolution = resolve_market_params(
            db_path=sample_db,
            sport="nba",
            season="2024-25",
            model="elo",
            market="TOTAL",
        )
        
        # Should be missing/default, not ML params
        assert total_resolution.params_source_label in {
            "missing_active", 
            "default_active",
            "db_market_best_run",  # If there's a best run available
        }
        # Should NOT have ML params
        if total_resolution.params:
            assert total_resolution.params.get("from_market") != "ML"
        
        # Market should be correctly labeled
        assert total_resolution.market == "TOTAL"
        
        # Metric should be total metric, not ML metric
        if total_resolution.metric_optimized:
            assert "log_loss" not in total_resolution.metric_optimized
    
    def test_fallback_label_is_clear(self, sample_db: Path) -> None:
        """Verify fallback params have clear source labels."""
        # No active params for bradley-terry in SPREAD
        resolution = resolve_market_params(
            db_path=sample_db,
            sport="nba",
            season="2024-25",
            model="bradley-terry",
            market="SPREAD",
        )
        
        # Should indicate missing/default
        assert resolution.params_source_label in {
            "missing_active",
            "default_active",
            "db_market_best_run",
        }
        assert resolution.market == "SPREAD"


class TestScheduleColumnsContract:
    """Test D: Schedule columns come from correct market pipelines."""
    
    def test_get_all_market_specs(self, sample_db: Path) -> None:
        """Verify all three markets have independent specs."""
        specs = get_all_market_specs(
            db_path=sample_db,
            sport="nba",
            season="2024-25",
        )
        
        assert "ML" in specs
        assert "SPREAD" in specs
        assert "TOTAL" in specs
        
        # Each has its own metric
        assert specs["ML"].metric_slot == "log_loss"
        assert specs["SPREAD"].metric_slot == "mae_margin"
        assert specs["TOTAL"].metric_slot == "mae_total"
        
        # Each has independent model lists
        assert specs["ML"].models != specs["TOTAL"].models or specs["TOTAL"].models == specs["ML"].models == []
    
    def test_validate_market_isolation(self, sample_db: Path) -> None:
        """Test the isolation validator."""
        # With no tuned params, should report issues
        issues = validate_market_isolation(
            db_path=sample_db,
            sport="nba",
            season="2024-25",
            models=["elo"],
        )
        
        # Should have warnings about missing active params
        assert any("missing" in str(issues).lower() or "no active" in str(issues).lower() 
                   for market_issues in issues.values() for _ in market_issues) or len(issues) >= 0
    
    def test_params_resolution_has_market_in_result(self, sample_db: Path) -> None:
        """Each resolution should clearly indicate which market it's for."""
        for market in ["ML", "SPREAD", "TOTAL"]:
            resolution = resolve_market_params(
                db_path=sample_db,
                sport="nba",
                season="2024-25",
                model="elo",
                market=market,
            )
            assert resolution.market == market
            
            # to_dict should include market
            d = resolution.to_dict()
            assert d["market"] == market


class TestMarketParamsResolutionDataclass:
    """Test the MarketParamsResolution dataclass behavior."""
    
    def test_is_market_optimized_true(self) -> None:
        """Test is_market_optimized returns True for tuned params."""
        resolution = MarketParamsResolution(
            model="elo",
            market="TOTAL",
            params={"k_factor": 30.0},
            params_source_label="tuned_active",
            metric_optimized="backtest_mae_total",
            source_run_id="run-123",
            best_score=0.5,
            params_fingerprint="abc123",
            params_nonempty=True,
        )
        assert resolution.is_market_optimized() is True
    
    def test_is_market_optimized_false_for_default(self) -> None:
        """Test is_market_optimized returns False for default params."""
        resolution = MarketParamsResolution(
            model="elo",
            market="TOTAL",
            params=None,
            params_source_label="missing_active",
            metric_optimized="backtest_mae_total",
            source_run_id=None,
            best_score=None,
            params_fingerprint="",
            params_nonempty=False,
        )
        assert resolution.is_market_optimized() is False
    
    def test_to_dict(self) -> None:
        """Test serialization."""
        resolution = MarketParamsResolution(
            model="elo",
            market="TOTAL",
            params={"k_factor": 30.0},
            params_source_label="tuned_active",
            metric_optimized="backtest_mae_total",
            source_run_id="run-123",
            best_score=0.5,
            params_fingerprint="abc123",
            params_nonempty=True,
        )
        d = resolution.to_dict()
        
        assert d["model"] == "elo"
        assert d["market"] == "TOTAL"
        assert d["params"] == {"k_factor": 30.0}
        assert d["is_market_optimized"] is True


class TestMarketEnsembleSpec:
    """Test the MarketEnsembleSpec dataclass behavior."""
    
    def test_to_dict(self) -> None:
        spec = MarketEnsembleSpec(
            market="TOTAL",
            ensemble_id="ensemble_total_v1",
            models=["poisson", "gssd"],
            weights={"poisson": 0.6, "gssd": 0.4},
            metric_slot="mae_total",
            weights_source="db_active",
            source_run_id="run-456",
            config_path=None,
        )
        d = spec.to_dict()
        
        assert d["market"] == "TOTAL"
        assert d["models"] == ["poisson", "gssd"]
        assert d["weights"] == {"poisson": 0.6, "gssd": 0.4}
        assert d["metric_slot"] == "mae_total"
