import pytest
from datetime import date
from pathlib import Path

import pandas as pd

from data.repository import (
    save_games,
    save_model_market_tuning_run,
    set_active_model_market_params,
)
from forecasting.forecast_service import (
    ForecastParams,
    load_latest_forecast_params,
    refresh_forecast_params,
)
from ingest.schema import GameResult
from pipelines.schedule import build_schedule_excel_report


def _sample_games():
    teams = [
        ("TeamA", "TeamB"),
        ("TeamC", "TeamD"),
        ("TeamE", "TeamF"),
        ("TeamG", "TeamH"),
        ("TeamI", "TeamJ"),
    ]
    games = []
    current = date(2025, 1, 1)
    score = 100
    for home, away in teams:
        games.append(
            GameResult(
                date=current,
                start_time=None,
                home_team=home,
                away_team=away,
                home_score=score,
                away_score=score - 5,
                neutral=False,
                overtime=False,
                decision_type=None,
                game_id=None,
                sport="nba",
                season="2025-26",
                division=None,
                conference=None,
                notes=None,
            )
        )
        current = date(current.year, current.month, current.day + 1)
        score += 1
    return games


def _setup_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "games.db"
    save_games(db_path, _sample_games())
    return db_path


def _load_bets_total(workbook_path: Path) -> pd.Series:
    bets = pd.read_excel(workbook_path, sheet_name="BETS")
    return bets["total"].dropna()


def test_schedule_production_mode_no_fit(tmp_path, monkeypatch):
    db_path = _setup_db(tmp_path)
    output_path = tmp_path / "schedule.xlsx"

    with pytest.raises(RuntimeError, match="Production schedule requires persisted forecast params"):
        build_schedule_excel_report(
            db_path=db_path,
            sport="nba",
            season="2025-26",
            model="bradley-terry",
            output_path=output_path,
            mode="production",
        )

    refresh_forecast_params(
        db_path=db_path,
        sport="nba",
        season="2025-26",
        model="bradley-terry",
        as_of_date="2025-01-31",
    )

    for target in [
        "pipelines.projections.fit_total_model",
        "pipelines.projections.fit_win_prob_scale",
        "models.calibration.fit_conditional_sd",
        "pipelines.matchups.team_home_advantages",
    ]:
        monkeypatch.setattr(target, lambda *args, **kwargs: pytest.fail(f"{target} called"))

    path = build_schedule_excel_report(
        db_path=db_path,
        sport="nba",
        season="2025-26",
        model="bradley-terry",
        output_path=output_path,
        mode="production",
    )
    assert path.exists()


def test_refresh_writes_forecast_params(tmp_path):
    db_path = _setup_db(tmp_path)
    forecast_params = refresh_forecast_params(
        db_path=db_path,
        sport="nba",
        season="2025-26",
        model="bradley-terry",
        as_of_date="2025-01-31",
    )
    record = load_latest_forecast_params(
        db_path, sport="nba", season="2025-26", model="bradley-terry"
    )
    assert record is not None
    loaded = ForecastParams.from_dict(record["params"])
    assert loaded.ratings == forecast_params.ratings
    assert loaded.as_of_date == forecast_params.as_of_date


def test_total_recency_adjustment_only_from_artifact(tmp_path, monkeypatch):
    """Test that total recency adjustments are loaded from artifacts, not computed inline.
    
    NOTE: This test references load_latest_total_recency_adjustment() and
    _calculate_total_recency_adjustment() which are NOT currently implemented.
    The test will fail with AttributeError when trying to mock these functions.
    
    Purpose: Total recency adjustment is a feature that modifies total predictions
    based on temporal patterns (e.g., recent scoring trends). When implemented:
    - The refresh pipeline should compute and save recency adjustments
    - The schedule pipeline should load pre-computed adjustments from artifacts
    - This test verifies the schedule pipeline doesn't try to compute them inline
    
    TODO: Implement load_latest_total_recency_adjustment() in src/pipelines/schedule.py
    and _calculate_total_recency_adjustment() in src/forecasting/
    """
    db_path = _setup_db(tmp_path)
    refresh_forecast_params(
        db_path=db_path,
        sport="nba",
        season="2025-26",
        model="bradley-terry",
        as_of_date="2025-02-01",
    )

    monkeypatch.setattr(
        "forecasting.forecast_service._calculate_total_recency_adjustment",
        lambda *args, **kwargs: pytest.fail("Internal recency computation invoked"),
    )
    called = {"loaded": False}

    def _mock_load_latest(*_, **__):
        called["loaded"] = True
        return {
            "delta": 5.0,
            "sample_size": 20,
            "lookback_games": 100,
            "as_of_date": "2025-02-01",
        }

    monkeypatch.setattr(
        "forecasting.forecast_service.load_latest_total_recency_adjustment",
        _mock_load_latest,
    )
    monkeypatch.setattr(
        "pipelines.schedule.load_latest_total_recency_adjustment",
        _mock_load_latest,
    )

    base_path = build_schedule_excel_report(
        db_path=db_path,
        sport="nba",
        season="2025-26",
        model="bradley-terry",
        output_path=tmp_path / "base.xlsx",
        mode="production",
    )
    base_totals = _load_bets_total(base_path)

    delta_path = build_schedule_excel_report(
        db_path=db_path,
        sport="nba",
        season="2025-26",
        model="bradley-terry",
        output_path=tmp_path / "delta.xlsx",
        mode="production",
        apply_total_recency_adjustment=True,
    )
    delta_totals = _load_bets_total(delta_path)

    assert called["loaded"]
    assert len(base_totals) == len(delta_totals)
    diffs = delta_totals.reset_index(drop=True) - base_totals.reset_index(drop=True)
    assert (diffs == pytest.approx(5.0)).all()


def test_artifact_precedence_production(tmp_path, monkeypatch):
    db_path = _setup_db(tmp_path)

    from data.repository import set_active_model_market_params

    set_active_model_market_params(
        db_path=db_path,
        sport="nba",
        season="2025-26",
        model="bradley-terry",
        market="ML",
        params={"home_advantage": 3.0},
        source_run_id="active",
        params_source="cli",
        metric_optimized=None,
        best_score=None,
    )
    save_model_market_tuning_run(
        db_path=db_path,
        sport="nba",
        season="2025-26",
        model="bradley-terry",
        market="ML",
        metric_optimized="log_loss",
        run_id="best-run",
        best_score=1.0,
        best_params_json="{}",
        summary_metrics_json="{}",
        started_at="",
        finished_at="",
    )

    refresh_forecast_params(
        db_path=db_path,
        sport="nba",
        season="2025-26",
        model="bradley-terry",
        as_of_date="2025-01-31",
    )

    monkeypatch.setattr(
        "pipelines.model_params.load_best_model_market_tuning_run",
        lambda *args, **kwargs: pytest.fail("Best-run fallback invoked in production"),
    )

    build_schedule_excel_report(
        db_path=db_path,
        sport="nba",
        season="2025-26",
        model="bradley-terry",
        output_path=tmp_path / "precedence.xlsx",
        mode="production",
    )
