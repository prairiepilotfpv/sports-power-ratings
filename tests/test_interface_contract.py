from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from models import base as base_models
from models.poisson import PoissonModel
from models.registry import get_backtest_model, list_backtest_models

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "mini_nba.csv"


@pytest.mark.parametrize("model_name", list_backtest_models())
def test_backtest_model_predictions_contract(model_name: str) -> None:
    games_df = pd.read_csv(FIXTURE_PATH)
    games_df["date"] = pd.to_datetime(games_df["date"]).dt.date

    cutoff_date = sorted(games_df["date"].unique())[15]
    train_df = games_df[games_df["date"] < cutoff_date]
    upcoming_df = games_df[games_df["date"] >= cutoff_date].drop(
        columns=["home_score", "away_score"],
        errors="ignore",
    )

    model_cls = get_backtest_model(model_name)
    model = model_cls()
    model.fit(train_df)
    predictions = model.predict(upcoming_df)

    assert predictions, "Expected predictions for the fixture dataset."

    game_ids = []
    for prediction in predictions:
        assert prediction.game_id
        assert prediction.date
        assert prediction.home_team
        assert prediction.away_team
        assert prediction.p_home_win is not None
        assert 0.0 <= prediction.p_home_win <= 1.0
        assert prediction.win_prob_source is not None
        assert prediction.margin_dist_assumption is not None
        game_ids.append(prediction.game_id)

        margin_mean = prediction.margin_mean
        source = prediction.win_prob_source
        if margin_mean is None:
            continue
        if abs(margin_mean) > 0.1 and source != "direct":
            assert (prediction.p_home_win > 0.5) == (
                margin_mean > 0
            ), "model_p_home_win should agree with margin_mean sign"

        if prediction.margin_dist_assumption == "normal_approx" and source != "direct":
            derived = base_models._home_win_prob_from_margin(
                margin_mean, prediction.margin_sd
            )
            if derived is not None and abs(margin_mean) > 0.1:
                assert (derived > 0.5) == (
                    prediction.p_home_win > 0.5
                ), "derived normal win prob should match model_p_home_win side"

    assert len(game_ids) == len(set(game_ids)), "game_id values must be unique."


def test_poisson_predictions_skip_normal_consistency_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    games_df = pd.read_csv(FIXTURE_PATH)
    games_df["date"] = pd.to_datetime(games_df["date"]).dt.date

    cutoff_date = sorted(games_df["date"].unique())[15]
    train_df = games_df[games_df["date"] < cutoff_date]
    upcoming_df = games_df[games_df["date"] >= cutoff_date].drop(
        columns=["home_score", "away_score"],
        errors="ignore",
    )

    def _boom(*args: object, **kwargs: object) -> float | None:
        raise AssertionError("Poisson should not invoke normal-derived win prob.")

    monkeypatch.setattr(base_models, "_home_win_prob_from_margin", _boom)

    model = PoissonModel(random_seed=1)
    model.fit(train_df)
    predictions = model.predict(upcoming_df)

    assert predictions, "Expected predictions for the fixture dataset."
    # Poisson uses either "empirical" (sample-based) or "skellam" (analytical) distribution
    # but NOT the normal approximation
    valid_poisson_assumptions = {"empirical", "skellam"}
    assert all(
        pred.margin_dist_assumption in valid_poisson_assumptions for pred in predictions
    ), f"Expected Poisson assumptions, got: {set(p.margin_dist_assumption for p in predictions)}"
    assert any(
        pred.margin_mean is not None and abs(pred.margin_mean) < 0.5
        for pred in predictions
    ), "Expected a near-coinflip Poisson prediction."


@pytest.mark.parametrize(
    "model_name",
    ["elo", "gssd", "toor"],
)
def test_non_poisson_models_still_use_normal_consistency_check(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
) -> None:
    games_df = pd.read_csv(FIXTURE_PATH)
    games_df["date"] = pd.to_datetime(games_df["date"]).dt.date

    cutoff_date = sorted(games_df["date"].unique())[15]
    train_df = games_df[games_df["date"] < cutoff_date]
    upcoming_df = games_df[games_df["date"] >= cutoff_date].drop(
        columns=["home_score", "away_score"],
        errors="ignore",
    )

    original = base_models._home_win_prob_from_margin
    calls: list[tuple[float, float | None]] = []

    def _wrapped(mean: float, sd: float | None) -> float | None:
        calls.append((mean, sd))
        return original(mean, sd)

    monkeypatch.setattr(base_models, "_home_win_prob_from_margin", _wrapped)

    model_cls = get_backtest_model(model_name)
    model = model_cls()
    model.fit(train_df)
    predictions = model.predict(upcoming_df)

    assert predictions, "Expected predictions for the fixture dataset."
    assert calls, "Expected normal-derived win prob consistency check to run."
