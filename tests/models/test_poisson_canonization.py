from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.poisson import PoissonModel
from pipelines.market_tuning import run_model_markets_tuning
from pipelines.projection_engines import get_projection_engine


def _training_games_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": date(2024, 1, 1),
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_score": 3,
                "away_score": 1,
                "neutral": False,
            },
            {
                "date": date(2024, 1, 2),
                "home_team": "Beta",
                "away_team": "Gamma",
                "home_score": 2,
                "away_score": 2,
                "neutral": False,
            },
            {
                "date": date(2024, 1, 3),
                "home_team": "Gamma",
                "away_team": "Alpha",
                "home_score": 1,
                "away_score": 4,
                "neutral": True,
            },
        ]
    )


def test_poisson_predictions_populate_all_heads() -> None:
    games = _training_games_df()
    model = PoissonModel(n_simulations=300, random_seed=5)
    model.fit(games)

    upcoming = pd.DataFrame(
        [
            {
                "date": date(2024, 1, 4),
                "home_team": "Alpha",
                "away_team": "Gamma",
                "neutral": False,
            }
        ]
    )
    predictions = model.predict(upcoming)

    assert predictions
    pred = predictions[0]
    assert 0.0 < pred.p_home_win < 1.0
    assert pred.pred_margin is not None
    assert pred.margin_mean is not None
    assert pred.pred_total is not None
    assert pred.total_mean is not None
    assert pytest.approx(pred.pred_margin) == pred.margin_mean
    assert pytest.approx(pred.pred_total) == pred.total_mean
    assert pred.margin_sd is not None and pred.margin_sd > 0
    assert pred.total_sd is not None and pred.total_sd > 0
    assert pred.margin_sd_raw is not None and pred.margin_sd_raw >= pred.margin_sd_used
    assert pred.total_sd_raw is not None and pred.total_sd_raw >= pred.total_sd_used
    assert pred.win_prob_source == "poisson_skellam"
    assert pred.sigma_source == "poisson_skellam"
    assert pred.margin_dist_assumption == "skellam"


def test_poisson_projection_engine_matches_prediction() -> None:
    games = _training_games_df()
    model = PoissonModel(n_simulations=4, random_seed=11)
    model.fit(games)

    upcoming = pd.DataFrame(
        [
            {
                "date": date(2024, 1, 4),
                "home_team": "Alpha",
                "away_team": "Gamma",
                "neutral": False,
            }
        ]
    )
    pred = model.predict(upcoming)[0]

    engine = get_projection_engine(model._rating_model)
    projection = engine(
        "Alpha",
        "Gamma",
        model._rating_model,
        {"neutral": False, "n_simulations": 4},
    )

    assert projection["model_p_home_win"] == pytest.approx(pred.p_home_win)
    assert projection["projected_win_prob"] == pytest.approx(pred.p_home_win)
    assert projection["win_prob_source"] == pred.win_prob_source
    assert projection["normal_p_home_win"] is None


def test_poisson_projection_engine_falls_back_to_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    games = _training_games_df()
    model = PoissonModel(n_simulations=4, random_seed=11)
    model.fit(games)

    def _expected_none(*_args: object, **_kwargs: object):
        return None

    monkeypatch.setattr(model._rating_model, "expected_goals", _expected_none)
    samples = (
        np.array([2.0, 2.0, 3.0, 1.0], dtype=float),
        np.array([1.0, 1.0, 0.0, 2.0], dtype=float),
    )

    def _fixed_samples(*_args: object, **_kwargs: object):
        return samples

    monkeypatch.setattr(model._rating_model, "simulate_matchup", _fixed_samples)

    upcoming = pd.DataFrame(
        [
            {
                "date": date(2024, 1, 4),
                "home_team": "Alpha",
                "away_team": "Gamma",
                "neutral": False,
            }
        ]
    )
    pred = model.predict(upcoming)[0]

    engine = get_projection_engine(model._rating_model)
    projection = engine(
        "Alpha",
        "Gamma",
        model._rating_model,
        {"neutral": False, "n_simulations": 4},
    )

    assert pred.win_prob_source == "sample"
    assert projection["win_prob_source"] == "sample"


def test_poisson_mc_fallback_sets_sigma_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force MC path by blocking analytical path, verify sigma_source == 'poisson_mc'.

    This test ensures that when the analytical (Skellam) path is unavailable,
    the model correctly falls back to Monte Carlo simulation and sets the
    appropriate sigma_source indicator.
    """
    games = _training_games_df()
    model = PoissonModel(n_simulations=100, random_seed=42)
    model.fit(games)

    # Block analytical path by making expected_goals return None
    def _expected_none(*_args, **_kwargs):
        return None

    monkeypatch.setattr(model._rating_model, "expected_goals", _expected_none)

    # Use fixed deterministic samples for reproducibility
    fixed_home_samples = np.array([3, 2, 4, 3, 2, 3, 3, 2, 4, 3] * 10, dtype=float)
    fixed_away_samples = np.array([1, 2, 1, 2, 3, 1, 2, 2, 2, 1] * 10, dtype=float)

    def _fixed_samples(*_args, **_kwargs):
        return (fixed_home_samples, fixed_away_samples)

    monkeypatch.setattr(model._rating_model, "simulate_matchup", _fixed_samples)

    upcoming = pd.DataFrame([{
        "date": date(2024, 1, 4),
        "home_team": "Alpha",
        "away_team": "Gamma",
        "neutral": False,
    }])

    predictions = model.predict(upcoming)
    assert len(predictions) == 1
    pred = predictions[0]

    # Key assertion: sigma_source must be poisson_mc when using MC fallback
    assert pred.sigma_source == "poisson_mc", \
        f"Expected sigma_source='poisson_mc', got '{pred.sigma_source}'"

    # Also verify win_prob_source indicates sample-based path
    assert pred.win_prob_source == "sample", \
        f"Expected win_prob_source='sample', got '{pred.win_prob_source}'"

    # Verify margin_dist_assumption is empirical
    assert pred.margin_dist_assumption == "empirical", \
        f"Expected margin_dist_assumption='empirical', got '{pred.margin_dist_assumption}'"

    # Verify SD values are computed from samples
    assert pred.margin_sd is not None and pred.margin_sd > 0
    assert pred.total_sd is not None and pred.total_sd > 0

    # Verify raw vs used SD tracking
    assert pred.margin_sd_raw is not None
    assert pred.total_sd_raw is not None


def _create_fixture_csv(path: Path) -> None:
    path.write_text(
        "date,home_team,away_team,home_score,away_score\n"
        "2024-11-01,A,B,100,90\n"
        "2024-11-02,B,C,95,98\n"
        "2024-11-03,C,A,88,92\n",
        encoding="utf-8",
    )


def test_poisson_tuning_records_markets(tmp_path: Path) -> None:
    csv_path = tmp_path / "games.csv"
    _create_fixture_csv(csv_path)
    db_path = tmp_path / "season.db"
    output_dir = tmp_path / "outputs"

    outcomes = run_model_markets_tuning(
        sport="nba",
        season="2025-26",
        model="poisson",
        markets=None,
        start_date="2024-11-01",
        end_date="2024-11-04",
        window="expanding",
        rolling_days=None,
        rolling_games=None,
        csv_path=csv_path,
        output_dir=output_dir,
        grid_override={"n_simulations": [300]},
        db_path=db_path,
        metric_overrides=None,
        allow_worse=True,
        jobs=1,
        activate_best=True,
    )

    assert [outcome.market for outcome in outcomes] == ["ML", "SPREAD", "TOTAL"]
    market_run_ids: dict[str, str] = {}
    for outcome in outcomes:
        assert outcome.result is not None
        market_run_ids[outcome.market] = outcome.result.run_id
    assert set(market_run_ids) == {"ML", "SPREAD", "TOTAL"}

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT market, run_id FROM model_market_tuning_runs "
            "WHERE sport=? AND season=? AND model=?",
            ("nba", "2025-26", "poisson"),
        ).fetchall()
        assert {row[0] for row in rows} == {"ML", "SPREAD", "TOTAL"}
        for market, run_id in rows:
            assert market_run_ids[market] == run_id

        active_rows = conn.execute(
            "SELECT market, source_run_id FROM model_market_active_params "
            "WHERE sport=? AND season=? AND model=?",
            ("nba", "2025-26", "poisson"),
        ).fetchall()
        assert len(active_rows) == 3
        for market, source_run_id in active_rows:
            assert market_run_ids[market] == source_run_id
