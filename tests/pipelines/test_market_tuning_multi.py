import sqlite3
from pathlib import Path

from pipelines.market_tuning import run_model_markets_tuning


def _create_fixture_csv(path: Path) -> None:
    path.write_text(
        "date,home_team,away_team,home_score,away_score\n"
        "2024-11-01,A,B,100,90\n"
        "2024-11-02,B,C,95,98\n"
        "2024-11-03,C,A,88,92\n",
        encoding="utf-8",
    )

def test_run_model_markets_tuning_records_all_markets(tmp_path: Path) -> None:
    csv_path = tmp_path / "games.csv"
    _create_fixture_csv(csv_path)
    db_path = tmp_path / "season.db"
    output_dir = tmp_path / "outputs"

    outcomes = run_model_markets_tuning(
        sport="nba",
        season="2025-26",
        model="bradley-terry",
        markets=None,
        start_date="2024-11-01",
        end_date="2024-11-04",
        window="expanding",
        rolling_days=None,
        rolling_games=None,
        csv_path=csv_path,
        output_dir=output_dir,
        grid_override={
            "temp": [3.0],
            "l2_lambda": [0.001],
            "learn_hfa": [False, True],
        },
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
            ("nba", "2025-26", "bradley-terry"),
        ).fetchall()
        assert {row[0] for row in rows} == {"ML", "SPREAD", "TOTAL"}
        for market, run_id in rows:
            assert market_run_ids[market] == run_id

        active_rows = conn.execute(
            "SELECT market, source_run_id FROM model_market_active_params "
            "WHERE sport=? AND season=? AND model=?",
            ("nba", "2025-26", "bradley-terry"),
        ).fetchall()
        assert len(active_rows) == 3
        for market, source_run_id in active_rows:
            assert market_run_ids[market] == source_run_id