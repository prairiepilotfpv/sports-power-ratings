from __future__ import annotations

from datetime import date
from pathlib import Path
import logging

import pandas as pd
import pytest

from calibration.distribution import MarginalDistributionCalibrator
from data.repository import save_games
from ingest.schema import GameResult
from markets.base import Market
from pipelines.schedule import build_schedule_excel_report, build_schedule_with_projections
import pipelines.schedule as schedule_module


def _seed_minimal_schedule_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "games.db"
    games = [
        GameResult(
            date=date(2025, 10, 25),
            home_team="Home",
            away_team="Away",
            home_score=110,
            away_score=105,
            sport="nba",
            season="2025-26",
        ),
        GameResult(
            date=date(2025, 10, 27),
            home_team="Away",
            away_team="Home",
            home_score=None,
            away_score=None,
            sport="nba",
            season="2025-26",
        ),
    ]
    save_games(db_path, games)
    return db_path


class _IdentityMLCalibrator:
    def transform(self, values):
        return pd.Series(values, dtype=float)


def _build_distribution_calibrator() -> MarginalDistributionCalibrator:
    calibrator = MarginalDistributionCalibrator()
    data = pd.DataFrame(
        {
            "pred_mean": [1.0, 2.0, 3.0],
            "pred_sd": [0.5, 0.7, 0.9],
            "actual_value": [0.8, 2.2, 3.1],
        }
    )
    calibrator.fit(data)
    return calibrator


def test_calibration_applied_once_per_export_path(monkeypatch):
    schedule_row = pd.DataFrame(
        [
            {
                "game_id": "G-1",
                "home_team": "Home",
                "away_team": "Away",
                "projected_winner": "Home",
                "home_win_prob": 0.7,
                "away_win_prob": 0.3,
                "margin_mean": 5.0,
                "margin_sd": 2.0,
                "total_mean": 210.0,
                "total_sd": 4.0,
                "win_prob_source": "seed_model",
            }
        ]
    )

    class _CountingProbabilityCalibrator:
        def __init__(self) -> None:
            self.calls = 0

        def transform(self, values):
            self.calls += 1
            return pd.Series(values, dtype=float)

    class _CountingDistributionCalibrator:
        def __init__(self) -> None:
            self.calls = 0
            self.metadata = {"method": "marginal_distribution"}

        def transform(self, df: pd.DataFrame) -> pd.DataFrame:
            self.calls += 1
            return pd.DataFrame(
                {
                    "calibrated_mean": df["pred_mean"] + 0.5,
                    "calibrated_sd": df["pred_sd"] + 0.5,
                }
            )

    ml_calibrator = _CountingProbabilityCalibrator()
    spread_calibrator = _CountingDistributionCalibrator()
    total_calibrator = _CountingDistributionCalibrator()

    def _mock_calibrator(*, market, **kwargs):
        key = market.name if isinstance(market, Market) else str(market)
        key = key.upper()
        if key == Market.ML.name:
            return ml_calibrator
        if key == Market.SPREAD.name:
            return spread_calibrator
        if key == Market.TOTAL.name:
            return total_calibrator
        return None

    monkeypatch.setattr(
        schedule_module,
        "load_latest_calibrator",
        _mock_calibrator,
    )

    first = schedule_module._apply_calibration_to_schedule_df(
        schedule_row,
        sport="nba",
        season="2025-26",
        model="bradley-terry",
    )
    second = schedule_module._apply_calibration_to_schedule_df(
        first,
        sport="nba",
        season="2025-26",
        model="bradley-terry",
    )

    assert ml_calibrator.calls == 1
    assert spread_calibrator.calls == 1
    assert total_calibrator.calls == 1
    guardrail_cols = ["margin_sd_pre_guardrail", "total_sd_pre_guardrail"]
    def _prune(df: pd.DataFrame) -> pd.DataFrame:
        drop = [col for col in guardrail_cols if col in df.columns]
        return df.drop(columns=drop)

    pd.testing.assert_frame_equal(_prune(first), _prune(second))

    tags = second["win_prob_source"].iloc[0]
    assert tags.count("calibrated_ml") == 1
    assert tags.count("calibrated_spread") == 1
    assert tags.count("calibrated_total") == 1


def test_spread_weights_filter_out_models_without_spread_outputs(caplog):
    caplog.set_level(logging.INFO)
    caplog.clear()
    forecast_df = pd.DataFrame(
        [
            {
                "game_id": "G1",
                "model_name": "gssd",
                "margin_mean": 1.0,
                "margin_sd": 0.5,
            },
            {
                "game_id": "G1",
                "model_name": "poisson",
                "margin_mean": None,
                "margin_sd": None,
            },
            {
                "game_id": "G1",
                "model_name": "elo",
                "margin_mean": 2.0,
                "margin_sd": 1.0,
            },
        ]
    )
    weights = {"gssd": 0.1, "poisson": 0.9, "elo": 0.0}

    filtered_weights, final_models = schedule_module._filter_market_weights_for_forecast(
        weights=weights,
        forecast_df=forecast_df,
        market=Market.SPREAD.name,
    )

    assert set(filtered_weights.keys()) == {"gssd"}
    assert filtered_weights["gssd"] == pytest.approx(1.0)
    assert final_models == {"gssd"}

    log_text = " ".join(record.message for record in caplog.records)
    assert "poisson" in log_text and "margin_mean" in log_text
    assert "elo" in log_text
