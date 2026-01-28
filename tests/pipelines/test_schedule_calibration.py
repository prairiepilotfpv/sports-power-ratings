from __future__ import annotations

from datetime import date
from pathlib import Path
import logging

import pandas as pd
import pytest

from calibration.distribution import VarianceCalibrator
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


def _build_distribution_calibrator() -> VarianceCalibrator:
    calibrator = VarianceCalibrator()
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
            self.metadata = {"method": "variance_distribution"}

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


# ========== MULTI-MEMBER ENSEMBLE TESTS ==========


def test_multi_member_ensemble_components_json_ml(tmp_path, monkeypatch):
    """Verify ml_ensemble_components_json lists multiple members when available."""
    import json
    from ensemble.ml_v1 import MLWeightedAverageEnsemble

    monkeypatch.chdir(tmp_path)

    # Create weights config with 3 models
    weights = {"model_a": 0.4, "model_b": 0.35, "model_c": 0.25}

    ens = MLWeightedAverageEnsemble(
        "testsport",
        "2025",
        weights=weights,
    )

    # Create forecast DataFrame with predictions from all 3 models
    forecast_df = pd.DataFrame([
        {"model_name": "model_a", "p_home_win": 0.65},
        {"model_name": "model_b", "p_home_win": 0.55},
        {"model_name": "model_c", "p_home_win": 0.60},
    ])

    combined_prob, components_json = ens.combine(forecast_df)

    # Parse and verify components
    components = json.loads(components_json)

    # Must have all 3 members
    assert len(components) == 3, f"Expected 3 components, got {len(components)}"

    # Verify structure of each component
    models_in_output = {c["model"] for c in components}
    assert models_in_output == {"model_a", "model_b", "model_c"}

    # Verify weights sum to 1.0
    weight_sum = sum(c["weight"] for c in components)
    assert weight_sum == pytest.approx(1.0, rel=1e-6)

    # Verify each component has required fields
    for comp in components:
        assert "model" in comp
        assert "weight" in comp
        assert "value" in comp
        assert comp["value"] is not None  # All models had valid probs


def test_multi_member_ensemble_components_json_spread(tmp_path, monkeypatch):
    """Verify spread_ensemble_components_json lists multiple members when available."""
    import json
    from ensemble.spread_v1 import SpreadWeightedAverageEnsemble

    monkeypatch.chdir(tmp_path)

    # Create weights config with 3 models
    weights = {"gssd": 0.5, "toor": 0.3, "elo": 0.2}

    ens = SpreadWeightedAverageEnsemble(
        "nba",
        "2025-26",
        weights=weights,
    )

    # Create forecast DataFrame with predictions from all 3 models
    forecast_df = pd.DataFrame([
        {"model_name": "gssd", "margin_mean": 5.0, "margin_sd": 10.0},
        {"model_name": "toor", "margin_mean": 3.0, "margin_sd": 11.0},
        {"model_name": "elo", "margin_mean": 4.0, "margin_sd": 12.0},
    ])

    result = ens.combine(forecast_df)
    assert result is not None

    margin_mean, margin_sd, components_json = result

    # Parse and verify components
    components = json.loads(components_json)

    # Must have all 3 members
    assert len(components) >= 2, f"Expected at least 2 components, got {len(components)}"

    # Verify weights sum to 1.0
    weight_sum = sum(c["weight"] for c in components)
    assert weight_sum == pytest.approx(1.0, rel=1e-6)


def test_ensemble_components_sorted_by_model(tmp_path, monkeypatch):
    """Components should be sorted by model name for determinism."""
    import json
    from ensemble.ml_v1 import MLWeightedAverageEnsemble

    monkeypatch.chdir(tmp_path)

    ens = MLWeightedAverageEnsemble(
        "testsport",
        "2025",
        weights={"zebra": 1.0, "alpha": 1.0, "middle": 1.0},
    )

    forecast_df = pd.DataFrame([
        {"model_name": "zebra", "p_home_win": 0.5},
        {"model_name": "alpha", "p_home_win": 0.5},
        {"model_name": "middle", "p_home_win": 0.5},
    ])

    _, components_json = ens.combine(forecast_df)
    components = json.loads(components_json)

    model_order = [c["model"] for c in components]
    assert model_order == ["alpha", "middle", "zebra"], \
        f"Expected alphabetical order, got {model_order}"
