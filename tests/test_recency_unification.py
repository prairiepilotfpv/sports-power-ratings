import pandas as pd
import numpy as np
import types

from models.calibration import recency_weight
from models.elo import EloModel
from models.gssd import GSSDModel
from models.toor import TOORModel
from backtest.runner import run_backtest


def test_recency_weight_curve_consistency():
    # Same game dates + same fit_end_date + same lambda -> identical weights
    fit_end = pd.Timestamp("2025-01-31")
    dates = [
        pd.Timestamp("2025-01-31"),  # age 0
        pd.Timestamp("2025-01-30"),  # age 1
        pd.Timestamp("2025-01-25"),  # age 6
    ]
    lam = 0.02
    weights = [recency_weight(d, fit_end, lam) for d in dates]
    # Exponential decay across days
    assert np.isclose(weights[0], 1.0)
    assert weights[1] < weights[0]
    assert weights[2] < weights[1]


def test_models_call_recency_weight_once_per_fit(monkeypatch):
    # Instrument recency_weight to count calls
    calls = {"elo": 0, "gssd": 0, "toor": 0}

    def make_counter(model_key):
        def _counter(game_date, fit_end_date, recency_lambda):
            calls[model_key] += 1
            return recency_weight(game_date, fit_end_date, recency_lambda)
        return _counter

    # Prepare a minimal games dataframe
    df = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "home_team": ["A", "B", "A"],
        "away_team": ["B", "A", "B"],
        "home_score": [100, 90, 95],
        "away_score": [90, 100, 96],
        "neutral": [False, False, False],
    })

    # Elo
    monkeypatch.setattr("models.elo.recency_weight", make_counter("elo"))
    EloModel(recency_lambda=0.01).fit(df, fit_end_date=pd.Timestamp("2025-01-03"))
    assert calls["elo"] == len(df), "ELO should call recency_weight once per game"

    # GSSD (patch module-local import)
    monkeypatch.setattr("models.gssd.recency_weight", make_counter("gssd"))
    GSSDModel(recency_lambda=0.01).fit(df, fit_end_date=pd.Timestamp("2025-01-03"))
    assert calls["gssd"] == len(df), "GSSD should call recency_weight once per game"

    # TOOR (patch module-local import)
    monkeypatch.setattr("models.toor.recency_weight", make_counter("toor"))
    TOORModel(recency_lambda=0.01).fit(df, fit_end_date=pd.Timestamp("2025-01-03"))
    assert calls["toor"] == len(df), "TOOR should call recency_weight once per game"


def test_backtest_fit_end_date_matches_train_slice(monkeypatch):
    # Create a tiny dataset with one game per day
    df = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
        "home_team": ["A", "B", "A", "B"],
        "away_team": ["B", "A", "B", "A"],
        "home_score": [100, 90, 95, 98],
        "away_score": [90, 100, 96, 97],
        "neutral": [False, False, False, False],
    })

    # Capture the fit_end_dates observed through recency_weight calls
    seen_fit_end_dates = []

    def recorder(game_date, fit_end_date, recency_lambda):
        if fit_end_date is not None:
            seen_fit_end_dates.append(pd.Timestamp(fit_end_date))
        return recency_weight(game_date, fit_end_date, recency_lambda)

    monkeypatch.setattr("models.toor.recency_weight", recorder)

    # Use TOOR (applies recency in core fit) for clarity
    outputs = run_backtest(
        model_factory=lambda: TOORModel(recency_lambda=0.01),
        games_df=df,
        start_date="2025-01-02",
        end_date="2025-01-04",
        window="expanding",
    )

    # Expect one fit_end_date per training slice equal to max train date
    # Slices evaluate on 01-02, 01-03, 01-04; train sets end on 01-01, 01-02, 01-03 respectively.
    expected_end_dates = [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")]
    # We called recorder once per game in each slice; collapse unique fit_end_dates in order
    unique_seen = []
    for ts in seen_fit_end_dates:
        if not unique_seen or ts != unique_seen[-1]:
            unique_seen.append(ts)
    # There should be at least one call per slice; verify the unique sequence matches expectation
    assert unique_seen[:3] == expected_end_dates
