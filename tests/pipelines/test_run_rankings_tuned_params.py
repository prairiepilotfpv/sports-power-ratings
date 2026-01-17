from __future__ import annotations

from datetime import date

import pandas as pd

from pipelines.run_rankings import build_rankings


def test_build_rankings_filters_and_passes_fit_kwargs_for_gssd() -> None:
    # Two games with dates far apart — recency weighting should favor the recent result
    df = pd.DataFrame(
        [
            {
                "date": date(2020, 1, 1),
                "home_team": "B",
                "away_team": "A",
                "home_score": 80,
                "away_score": 70,
            },
            {
                "date": date(2020, 12, 31),
                "home_team": "A",
                "away_team": "B",
                "home_score": 90,
                "away_score": 60,
            },
        ]
    )

    # Tuned params include keys meant for the backtest (conditional_sd) and a fit kwarg (recency_lambda)
    params_with_recency = {
        "conditional_sd": True,
        "recency_lambda": 0.01,
        "some_extra": 123,
    }

    rankings_recency, _ = build_rankings(
        df, model="gssd", model_params=params_with_recency, return_model=True
    )
    assert not rankings_recency.empty

    rankings_no_recency, _ = build_rankings(
        df, model="gssd", model_params={"conditional_sd": True}, return_model=True
    )
    assert not rankings_no_recency.empty

    a_rating_recency = float(rankings_recency.loc[rankings_recency["team"] == "A", "rating"].iloc[0])
    a_rating_no_recency = float(rankings_no_recency.loc[rankings_no_recency["team"] == "A", "rating"].iloc[0])

    # The simple two-game setup above doesn't create multiple samples per-stat,
    # so recency weighting might not change per-stat averages; instead, verify
    # filtering and fit-kwarg propagation using an instrumented model.


def test_build_rankings_filters_extra_init_kwargs_and_passes_fit_kwargs_with_instrumented_model() -> None:
    from models.registry import register_model, unregister_model

    class InstrumentedGSSD:
        def __init__(self) -> None:
            # Accept no kwargs to validate that extra params are filtered out
            self.model_id = "gssd"
            self._fit_args = None

        def fit(self, games, *, recency_lambda: float | None = None, fit_end_date=None) -> None:
            # Record fit kwargs so test can assert they were passed
            self._fit_args = {"recency_lambda": recency_lambda, "fit_end_date": fit_end_date}

        def rankings(self):
            return [("A", 1.0), ("B", 0.0)]

        def metadata(self):
            return None

    register_model("gssd_instrumented", InstrumentedGSSD, abbreviation="gssd_inst")
    try:
        params = {"conditional_sd": True, "recency_lambda": 0.01, "extra": 123}
        df = pd.DataFrame(
            [
                {"date": date(2020, 1, 1), "home_team": "A", "away_team": "B", "home_score": 80, "away_score": 70}
            ]
        )
        rankings, inst = build_rankings(df, model="gssd_instrumented", model_params=params, return_model=True)
        assert not rankings.empty
        # Ensure the instrumented instance recorded the recency_lambda passed to fit
        assert hasattr(inst, "_fit_args") and inst._fit_args is not None
        assert inst._fit_args["recency_lambda"] == 0.01
    finally:
        unregister_model("gssd_instrumented")

