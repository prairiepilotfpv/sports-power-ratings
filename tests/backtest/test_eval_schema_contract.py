from __future__ import annotations

import pandas as pd

from backtest.eval_schema import REQUIRED_ACTUAL_COLS, REQUIRED_PRED_COLS
from backtest.runner import ensure_eval_schema, _compute_metrics


def test_ensure_eval_schema_retains_required_columns_after_concat() -> None:
    base = pd.DataFrame(
        [
            {"date": pd.to_datetime("2025-01-01"), "home_team": "H1", "away_team": "A1"},
            {"date": pd.to_datetime("2025-01-02"), "home_team": "H2", "away_team": "A2", "home_score": 100, "away_score": 90},
        ]
    )

    framed = pd.concat(
        [ensure_eval_schema(base.iloc[[0]]), ensure_eval_schema(base.iloc[[1]])],
        ignore_index=True,
    )

    required = REQUIRED_ACTUAL_COLS | REQUIRED_PRED_COLS
    assert required.issubset(set(framed.columns))
    # First row remains NA-derived, second row fills from scores.
    assert pd.isna(framed.loc[0, "actual_margin"])
    assert framed.loc[1, "actual_margin"] == 10
    assert framed.loc[1, "actual_total"] == 190


def test_ensure_eval_schema_preserves_existing_values() -> None:
    base = pd.DataFrame(
        [
            {
                "home_score": 100,
                "away_score": 90,
                "actual_margin": 1.0,
                "actual_total": 999.0,
                "home_win": 0.25,
            }
        ]
    )

    framed = ensure_eval_schema(base)

    assert framed.loc[0, "actual_margin"] == 1.0
    assert framed.loc[0, "actual_total"] == 999.0
    assert framed.loc[0, "home_win"] == 0.25


def test_compute_metrics_handles_zero_scorable_games() -> None:
    df = pd.DataFrame(
        [
            {
                "p_home_win": pd.NA,
                "home_win": pd.NA,
                "pred_margin": pd.NA,
                "actual_margin": pd.NA,
                "pred_total": pd.NA,
                "actual_total": pd.NA,
            }
        ]
    )

    metrics = _compute_metrics(df)

    assert metrics["games"] == 1
    assert metrics["ml_games"] == 0
    assert metrics["margin_games"] == 0
    assert metrics["total_games"] == 0
    assert pd.isna(metrics["log_loss"])
    assert pd.isna(metrics["mae_margin"])
    assert pd.isna(metrics["mae_total"])


def test_compute_metrics_isolates_markets() -> None:
    df = pd.DataFrame(
        [
            {
                "p_home_win": 0.75,
                "home_win": 1.0,
                "pred_margin": pd.NA,
                "actual_margin": pd.NA,
                "pred_total": pd.NA,
                "actual_total": pd.NA,
            }
        ]
    )

    metrics = _compute_metrics(df)

    assert metrics["ml_games"] == 1
    assert metrics["margin_games"] == 0
    assert metrics["total_games"] == 0
    assert metrics["log_loss"] > 0
    assert pd.isna(metrics["mae_margin"])
    assert pd.isna(metrics["mae_total"])
