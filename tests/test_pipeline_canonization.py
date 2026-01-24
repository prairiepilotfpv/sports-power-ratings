"""
Integration tests for pipeline canonization - protect key system invariants.

These tests ensure that core workflows remain stable as models/sports/ensembles change:
1. Import → Rank → Schedule → Backtest flow
2. Game ID consistency across all import paths
3. Ensemble application for all three markets (ML, SPREAD, TOTAL)
4. Market parameter isolation (ML params don't leak into SPREAD/TOTAL)
5. Source labeling correctness
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from data.repository import (
    init_db,
    save_games,
    load_games,
    save_model_market_tuning_run,
    set_active_model_market_params,
)
from ingest.schema import GameResult
from ingest.sources import SportsReferenceSource
from pipelines.ingest import ingest_games
from pipelines.run_rankings import run_rankings
from pipelines import schedule
from pipelines.market_utils import _resolve_market_metric
from pipelines.schedule import build_schedule_excel_report
from pipelines.backtest import run_backtest_pipeline


class TestPipelineCanonization:
    """End-to-end invariant tests for the complete pipeline."""

    def test_full_pipeline_flow_nba(self, tmp_path: Path) -> None:
        """Test complete flow: import → rank → schedule → backtest."""
        db_path = tmp_path / "nba.db"
        init_db(db_path)
        
        # Step 1: Import games
        games = [
            GameResult(
                date=date(2025, 1, 1),
                home_team="Lakers",
                away_team="Celtics",
                home_score=110,
                away_score=105,
                sport="nba",
                season="2024-25",
            ),
            GameResult(
                date=date(2025, 1, 2),
                home_team="Warriors",
                away_team="Lakers",
                home_score=98,
                away_score=102,
                sport="nba",
                season="2024-25",
            ),
            GameResult(
                date=date(2025, 1, 10),  # Future game
                home_team="Celtics",
                away_team="Warriors",
                home_score=None,
                away_score=None,
                sport="nba",
                season="2024-25",
            ),
        ]
        save_games(db_path, games)
        
        # Verify games imported with consistent IDs
        loaded = load_games(db_path, sport="nba", season="2024-25")
        assert len(loaded) == 3
        assert all(g.game_id.startswith("nba:2024-25:") for g in loaded)
        
        # Step 2: Build rankings
        rankings_path = run_rankings(
            db_path,
            sport="nba",
            season="2024-25",
            model="elo",
            output_path=tmp_path / "rankings.csv",
        )
        assert rankings_path.exists()
        rankings = pd.read_csv(rankings_path)
        assert {"Lakers", "Celtics", "Warriors"}.issubset(set(rankings["team"]))
        
        # Step 3: Build schedule with projections
        schedule_path = build_schedule_excel_report(
            db_path,
            sport="nba",
            season="2024-25",
            model="elo",
            output_path=tmp_path / "schedule.xlsx",
            as_of_date=date(2025, 1, 5),
        )
        assert schedule_path.exists()
        
        # Verify schedule has projections
        schedule_df = pd.read_excel(
            schedule_path,
            sheet_name="elo",
            skiprows=schedule.MODEL_METADATA_DATA_START_ROW - 1,
        )
        future_games = schedule_df[
            schedule_df["status"].isin(["scheduled", "upcoming"])
        ]
        assert len(future_games) > 0
        assert future_games["projected_win_prob"].notna().all()
        
        # Step 4: Run backtest
        csv_path = tmp_path / "games.csv"
        games_df = pd.DataFrame([
            {
                "date": g.date,
                "home_team": g.home_team,
                "away_team": g.away_team,
                "home_score": g.home_score,
                "away_score": g.away_score,
            }
            for g in games if g.home_score is not None
        ])
        games_df.to_csv(csv_path, index=False)
        
        outputs = run_backtest_pipeline(
            csv_path=csv_path,
            model="elo",
            start_date="2025-01-01",
            end_date="2025-01-02",
            output_dir=tmp_path / "backtest",
            db_path=db_path,
            sport="nba",
            season="2024-25",
        )
        assert not outputs.predictions.empty
        assert outputs.predictions["p_home_win"].notna().all()

    def test_game_id_consistency_across_sources(self, tmp_path: Path) -> None:
        """Game IDs must be identical regardless of import source."""
        db_path = tmp_path / "test.db"
        init_db(db_path)
        
        # Import via GameResult directly
        game1 = GameResult(
            date=date(2025, 1, 15),
            home_team="Team A",
            away_team="Team B",
            home_score=100,
            away_score=95,
            sport="nba",
            season="2024-25",
        )
        save_games(db_path, [game1])
        
        loaded1 = load_games(db_path, sport="nba", season="2024-25")
        assert len(loaded1) == 1
        game_id_format1 = loaded1[0].game_id
        
        # Verify hash-based format
        assert game_id_format1.startswith("nba:2024-25:2025-01-15:")
        assert len(game_id_format1.split(":")[-1]) == 12  # 12-char hash
        
        # Import same game via CSV (simulating different path)
        csv_path = tmp_path / "games.csv"
        df = pd.DataFrame([{
            "date": "2025-01-15",
            "home_team": "Team A",
            "away_team": "Team B",
            "home_score": 100,
            "away_score": 95,
        }])
        df.to_csv(csv_path, index=False)
        
        # Import again - should match existing game_id
        from contracts import ensure_game_id
        df_with_ids = ensure_game_id(df, sport="nba", season="2024-25")
        assert df_with_ids["game_id"].iloc[0] == game_id_format1

    def test_ensemble_application_all_markets(self, tmp_path: Path) -> None:
        """Ensembles must apply to ML, SPREAD, and TOTAL when multiple models exist."""
        db_path = tmp_path / "test.db"
        init_db(db_path)
        
        # Seed games
        games = [
            GameResult(
                date=date(2025, 1, 1),
                home_team="A",
                away_team="B",
                home_score=100,
                away_score=90,
                sport="nba",
                season="2024-25",
            ),
            GameResult(
                date=date(2025, 1, 10),
                home_team="A",
                away_team="B",
                home_score=None,
                away_score=None,
                sport="nba",
                season="2024-25",
            ),
        ]
        save_games(db_path, games)
        
        # Run rankings for multiple models
        for model in ["elo", "bradley-terry", "poisson"]:
            run_rankings(
                db_path,
                sport="nba",
                season="2024-25",
                model=model,
                output_path=tmp_path / f"{model}.csv",
            )
        
        # Build schedule with all models
        schedule_path = build_schedule_excel_report(
            db_path,
            sport="nba",
            season="2024-25",
            model=["elo", "bradley-terry", "poisson"],
            output_path=tmp_path / "schedule.xlsx",
            as_of_date=date(2025, 1, 5),
        )
        
        # Check BETS sheet for ensemble sources
        bets_df = pd.read_excel(schedule_path, sheet_name="BETS")
        ml_rows = bets_df[bets_df["market_type"] == "ML"]
        spread_rows = bets_df[bets_df["market_type"] == "spread"]
        total_rows = bets_df[bets_df["market_type"] == "total"]
        
        # With 3 models, all markets should use ensembles
        assert ml_rows["win_prob_source"].str.contains("ensemble").all(), \
            "ML should use ensemble with multiple models"
        assert spread_rows["spread_source"].str.contains("ensemble").all(), \
            "SPREAD should use ensemble with multiple models"
        assert total_rows["total_source"].str.contains("ensemble").all(), \
            "TOTAL should use ensemble with multiple models"
        
        # Verify no "direct+ensemble" concatenation
        assert not ml_rows["win_prob_source"].str.contains(r"\+").any(), \
            "ML source should not have concatenation (direct+ensemble)"

    def test_market_parameter_isolation(self, tmp_path: Path) -> None:
        """Market params must be isolated - ML params don't affect SPREAD/TOTAL."""
        db_path = tmp_path / "test.db"
        init_db(db_path)
        
        # Save different params for each market
        _, ml_metric_optimized = _resolve_market_metric("ML", None)
        ml_params = {"k_factor": 20.0, "_market_tag": "ML"}
        save_model_market_tuning_run(
            db_path,
            sport="nba",
            season="2024-25",
            model="elo",
            market="ML",
            metric_optimized=ml_metric_optimized,
            run_id="test_ml",
            best_score=0.5,
            best_params_json=json.dumps(ml_params),
            summary_metrics_json=None,
            started_at="2025-01-01",
            finished_at="2025-01-10",
            notes=None,
        )
        set_active_model_market_params(
            db_path,
            sport="nba",
            season="2024-25",
            model="elo",
            market="ML",
            params=ml_params,
            source_run_id="test_ml",
            params_source="tuned",
            metric_optimized=ml_metric_optimized,
            best_score=0.5,
        )
        
        _, spread_metric_optimized = _resolve_market_metric("SPREAD", None)
        spread_params = {"k_factor": 30.0, "_market_tag": "SPREAD"}
        save_model_market_tuning_run(
            db_path,
            sport="nba",
            season="2024-25",
            model="elo",
            market="SPREAD",
            metric_optimized=spread_metric_optimized,
            run_id="test_spread",
            best_score=5.0,
            best_params_json=json.dumps(spread_params),
            summary_metrics_json=None,
            started_at="2025-01-01",
            finished_at="2025-01-10",
            notes=None,
        )
        set_active_model_market_params(
            db_path,
            sport="nba",
            season="2024-25",
            model="elo",
            market="SPREAD",
            params=spread_params,
            source_run_id="test_spread",
            params_source="tuned",
            metric_optimized=spread_metric_optimized,
            best_score=5.0,
        )
        
        # Query and verify isolation
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT market, params_json
            FROM model_market_active_params
            WHERE sport = 'nba' AND season = '2024-25' AND model = 'elo'
        """)
        results = cursor.fetchall()
        conn.close()
        
        assert len(results) == 2
        params_by_market = {row[0]: row[1] for row in results}
        
        ml_params = json.loads(params_by_market["ML"])
        spread_params = json.loads(params_by_market["SPREAD"])
        
        assert ml_params["_market_tag"] == "ML"
        assert ml_params["k_factor"] == 20.0
        assert spread_params["_market_tag"] == "SPREAD"
        assert spread_params["k_factor"] == 30.0

    def test_source_labeling_correctness(self, tmp_path: Path) -> None:
        """Source labels must accurately reflect the data source."""
        db_path = tmp_path / "test.db"
        init_db(db_path)
        
        games = [
            GameResult(
                date=date(2025, 1, 1),
                home_team="A",
                away_team="B",
                home_score=100,
                away_score=95,
                sport="nba",
                season="2024-25",
            ),
            GameResult(
                date=date(2025, 1, 10),
                home_team="A",
                away_team="B",
                home_score=None,
                away_score=None,
                sport="nba",
                season="2024-25",
            ),
        ]
        save_games(db_path, games)
        
        # Single model - should show model name, not "ensemble"
        run_rankings(
            db_path,
            sport="nba",
            season="2024-25",
            model="elo",
            output_path=tmp_path / "elo.csv",
        )
        
        schedule_path = build_schedule_excel_report(
            db_path,
            sport="nba",
            season="2024-25",
            model="elo",
            output_path=tmp_path / "schedule.xlsx",
            as_of_date=date(2025, 1, 5),
        )
        
        bets_df = pd.read_excel(schedule_path, sheet_name="BETS")
        ml_rows = bets_df[bets_df["market_type"] == "ML"]
        
        # Single model should NOT use ensemble
        # (unless ensemble config forces it, which isn't the default)
        # At minimum, should not show "direct+ensemble" concatenation
        assert not ml_rows["win_prob_source"].str.contains(r"\+").any()

    def test_schedule_column_contract(self, tmp_path: Path) -> None:
        """Schedule output must maintain stable column schema."""
        from pipelines.schedule import SCHEDULE_EXPORT_COLUMNS
        
        db_path = tmp_path / "test.db"
        init_db(db_path)
        
        games = [
            GameResult(
                date=date(2025, 1, 5),
                home_team="A",
                away_team="B",
                home_score=101,
                away_score=99,
                sport="nba",
                season="2024-25",
            ),
            GameResult(
                date=date(2025, 1, 10),
                home_team="A",
                away_team="B",
                home_score=None,
                away_score=None,
                sport="nba",
                season="2024-25",
            ),
        ]
        save_games(db_path, games)
        run_rankings(
            db_path,
            sport="nba",
            season="2024-25",
            model="elo",
            output_path=tmp_path / "elo.csv",
        )
        
        schedule_path = build_schedule_excel_report(
            db_path,
            sport="nba",
            season="2024-25",
            model="elo",
            output_path=tmp_path / "schedule.xlsx",
            as_of_date=date(2025, 1, 5),
        )
        
        df = pd.read_excel(
            schedule_path,
            sheet_name="elo",
            skiprows=schedule.MODEL_METADATA_DATA_START_ROW - 1,
        )
        assert list(df.columns) == SCHEDULE_EXPORT_COLUMNS, \
            "Schedule columns must match SCHEDULE_EXPORT_COLUMNS contract"

    def test_backtest_prediction_contract(self, tmp_path: Path) -> None:
        """Backtest predictions must have p_home_win, pred_margin, pred_total populated."""
        csv_path = tmp_path / "games.csv"
        df = pd.DataFrame([
            {
                "date": "2025-01-01",
                "home_team": "A",
                "away_team": "B",
                "home_score": 100,
                "away_score": 95,
            },
            {
                "date": "2025-01-02",
                "home_team": "B",
                "away_team": "A",
                "home_score": 88,
                "away_score": 92,
            },
        ])
        df.to_csv(csv_path, index=False)
        
        outputs = run_backtest_pipeline(
            csv_path=csv_path,
            model="poisson",
            start_date="2025-01-01",
            end_date="2025-01-02",
            output_dir=tmp_path / "backtest",
        )
        
        preds = outputs.predictions
        assert preds["p_home_win"].notna().all(), "p_home_win must be populated"
        assert preds["pred_margin"].notna().all(), "pred_margin must be populated for MAE"
        assert preds["pred_total"].notna().all(), "pred_total must be populated for MAE"
        
        # Verify metrics computed successfully
        assert not outputs.metrics_overall.empty
        assert "log_loss" in outputs.metrics_overall.columns
        assert "mae_margin" in outputs.metrics_overall.columns
        assert "mae_total" in outputs.metrics_overall.columns


class TestSystemInvariants:
    """Test core system invariants that must never break."""

    def test_ensemble_imports_exist(self) -> None:
        """Verify all ensemble classes are importable (prevent missing import bugs)."""
        from ensemble.ml_v1 import MLWeightedAverageEnsemble
        from ensemble.spread_v1 import SpreadWeightedAverageEnsemble
        from ensemble.total_v1 import TotalWeightedAverageEnsemble
        
        # Verify classes are callable
        assert callable(MLWeightedAverageEnsemble)
        assert callable(SpreadWeightedAverageEnsemble)
        assert callable(TotalWeightedAverageEnsemble)

    def test_model_registry_stable(self) -> None:
        """Model registry must list all expected models."""
        from models.registry import list_backtest_models, list_models
        
        backtest_models = set(list_backtest_models())
        all_models = set(list_models())
        
        # Core models that must always exist
        required_models = {"elo", "bradley-terry", "poisson"}
        assert required_models.issubset(backtest_models)
        assert required_models.issubset(all_models)

    def test_contracts_module_validates(self) -> None:
        """Contracts module must be importable and functional."""
        from contracts import (
            validate_schedule_export_frame,
            ensure_game_id,
            SCHEDULE_EXPORT_COLUMNS,
        )
        
        # Verify constants exist
        assert isinstance(SCHEDULE_EXPORT_COLUMNS, list)
        assert len(SCHEDULE_EXPORT_COLUMNS) > 0
        
        # Verify validation works
        df = pd.DataFrame([{col: None for col in SCHEDULE_EXPORT_COLUMNS}])
        result = validate_schedule_export_frame(df)
        assert not result.empty

    def test_make_game_id_deterministic(self) -> None:
        """Game ID generation must be deterministic and stable."""
        from src.utils.game_id import make_game_id
        
        id1 = make_game_id("nba", "2024-25", date(2025, 1, 15), "Lakers", "Celtics")
        id2 = make_game_id("nba", "2024-25", date(2025, 1, 15), "Lakers", "Celtics")
        
        assert id1 == id2, "make_game_id must be deterministic"
        assert id1.startswith("nba:2024-25:2025-01-15:")
        assert len(id1.split(":")[-1]) == 12, "Hash must be 12 characters"

    def test_db_schema_stable(self, tmp_path: Path) -> None:
        """Database schema must have expected tables and columns."""
        db_path = tmp_path / "test.db"
        init_db(db_path)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check core tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        
        required_tables = {
            "games",
            "model_metrics",
            "model_market_tuning_runs",
            "model_market_active_params",
            "ensemble_market_tuning_runs",
            "ensemble_market_active_weights",
        }
        assert required_tables.issubset(tables), f"Missing tables: {required_tables - tables}"
        
        # Check games table columns
        cursor.execute("PRAGMA table_info(games)")
        columns = {row[1] for row in cursor.fetchall()}
        required_columns = {
            "game_id",
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "sport",
            "season",
        }
        assert required_columns.issubset(columns), f"Missing columns: {required_columns - columns}"
        
        conn.close()
