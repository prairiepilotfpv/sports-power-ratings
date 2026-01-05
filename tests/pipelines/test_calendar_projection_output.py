from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from pipelines.schedule import SCHEDULE_EXPORT_COLUMNS


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "calendar_projection"
    / "schedule_output.csv"
)


def _assert_boolean_series(series: pd.Series) -> None:
    for value in series.dropna():
        if isinstance(value, bool):
            continue
        if str(value).strip().lower() in {"true", "false"}:
            continue
        raise AssertionError(f"Expected boolean-like values, got {value!r}")


def _assert_numeric_series(series: pd.Series) -> None:
    numeric = pd.to_numeric(series, errors="coerce")
    assert (
        numeric.notna().sum() == series.notna().sum()
    ), "Found non-numeric values in numeric column"


def test_calendar_projection_output_schema_and_types() -> None:
    df = pd.read_csv(FIXTURE_PATH)

    assert list(df.columns) == SCHEDULE_EXPORT_COLUMNS

    string_columns = [
        "date",
        "game_id",
        "status",
        "home_team",
        "away_team",
        "projected_winner",
        "projected_win_prob_dist",
    ]
    boolean_columns = ["neutral", "overtime"]
    numeric_columns = [
        "home_score",
        "away_score",
        "result_margin",
        "result_total",
        "home_rating",
        "away_rating",
        "home_advantage",
        "projected_spread",
        "projected_home_spread",
        "projected_win_prob",
        "projected_home_score",
        "projected_away_score",
        "projected_total",
        "margin_std",
        "total_std",
    ]

    for column in string_columns:
        values = df[column].dropna()
        assert values.map(
            lambda value: isinstance(value, str)
        ).all(), f"{column} should be string-like"

    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    assert parsed_dates.notna().all(), "date column should be parseable"

    for column in boolean_columns:
        _assert_boolean_series(df[column])

    for column in numeric_columns:
        _assert_numeric_series(df[column])


def test_calendar_projection_output_has_unique_game_ids() -> None:
    df = pd.read_csv(FIXTURE_PATH)
    assert df["game_id"].is_unique


def test_final_results_match_scores() -> None:
    df = pd.read_csv(FIXTURE_PATH)
    final_rows = df[
        (df["status"] == "final") & df["home_score"].notna() & df["away_score"].notna()
    ]

    for _, row in final_rows.iterrows():
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])
        assert row["result_margin"] == home_score - away_score
        assert row["result_total"] == home_score + away_score


def test_projected_scores_sum_to_total() -> None:
    df = pd.read_csv(FIXTURE_PATH)
    with_scores = df[
        df["projected_home_score"].notna() & df["projected_away_score"].notna()
    ]

    for _, row in with_scores.iterrows():
        total = row["projected_home_score"] + row["projected_away_score"]
        assert math.isclose(total, row["projected_total"], rel_tol=1e-6, abs_tol=0.5)


def test_projected_spread_sign_convention() -> None:
    """Projected spread is away_minus_home; negative favors the home team."""
    df = pd.read_csv(FIXTURE_PATH)
    with_scores = df[
        df["projected_home_score"].notna() & df["projected_away_score"].notna()
    ]

    for _, row in with_scores.iterrows():
        margin = row["projected_home_score"] - row["projected_away_score"]
        assert math.isclose(
            row["projected_home_spread"], margin, rel_tol=1e-6, abs_tol=0.5
        )
        assert math.isclose(row["projected_spread"], -margin, rel_tol=1e-6, abs_tol=0.5)
