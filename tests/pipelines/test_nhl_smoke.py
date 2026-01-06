from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.repository import load_games, save_games
from ingest.sources import SportsReferenceSource
from pipelines.backtest import run_backtest_pipeline
from pipelines.ingest import ingest_games
from pipelines.run_rankings import run_rankings
from pipelines.schedule import build_schedule_with_projections


def test_nhl_pipeline_smoke(tmp_path: Path) -> None:
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "sports_reference"
        / "nhl"
        / "nhl_sample.csv"
    )
    db_path = tmp_path / "nhl.db"

    source = SportsReferenceSource()
    games = ingest_games(
        source,
        input_path=fixture_path,
        input_text=None,
        sport="nhl",
        season="2025-26",
    )
    assert games
    for game in games:
        if game.home_score is not None:
            assert isinstance(game.home_score, int)
        if game.away_score is not None:
            assert isinstance(game.away_score, int)

    save_games(db_path, games)
    assert load_games(db_path, sport="nhl", season="2025-26")

    rankings_path = run_rankings(
        db_path,
        sport="nhl",
        season="2025-26",
        model="elo",
        output_path=tmp_path / "rankings.csv",
    )
    rankings_df = pd.read_csv(rankings_path)
    teams = {game.home_team for game in games} | {game.away_team for game in games}
    assert teams.issubset(set(rankings_df["team"]))

    schedule_path = build_schedule_with_projections(
        db_path,
        sport="nhl",
        season="2025-26",
        model="elo",
        output_path=tmp_path / "schedule.csv",
    )
    schedule_df = pd.read_csv(schedule_path)
    for column in [
        "projected_home_score",
        "projected_away_score",
        "projected_total",
        "projected_win_prob",
    ]:
        assert column in schedule_df.columns

    poisson_schedule_path = build_schedule_with_projections(
        db_path,
        sport="nhl",
        season="2025-26",
        model="poisson",
        output_path=tmp_path / "schedule_poisson.csv",
    )
    poisson_schedule_df = pd.read_csv(poisson_schedule_path)
    assert poisson_schedule_df["projected_home_score"].notna().any()
    assert poisson_schedule_df["projected_away_score"].notna().any()
    assert poisson_schedule_df["projected_win_prob"].notna().any()

    outputs = run_backtest_pipeline(
        csv_path=fixture_path,
        model="elo",
        start_date="2025-10-08",
        end_date="2025-10-10",
        output_dir=tmp_path / "backtest",
        db_path=db_path,
        sport="nhl",
        season="2025-26",
    )
    assert not outputs.predictions.empty
    assert not outputs.metrics_by_date.empty
    assert not outputs.metrics_overall.empty

    poisson_outputs = run_backtest_pipeline(
        csv_path=fixture_path,
        model="poisson",
        start_date="2025-10-08",
        end_date="2025-10-10",
        output_dir=tmp_path / "backtest_poisson",
        db_path=db_path,
        sport="nhl",
        season="2025-26",
    )
    assert not poisson_outputs.predictions.empty
    assert not poisson_outputs.metrics_by_date.empty
    assert not poisson_outputs.metrics_overall.empty

    csv_outputs = list(outputs.output_dir.glob("*.csv"))
    excel_outputs = list(outputs.output_dir.glob("*.xlsx"))
    assert csv_outputs
    assert excel_outputs
    assert all(path.stat().st_size > 0 for path in csv_outputs)
    assert all(path.stat().st_size > 0 for path in excel_outputs)
