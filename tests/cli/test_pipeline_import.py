from argparse import Namespace
from pathlib import Path

from cli import pipeline
from data.repository import load_games


def test_pipeline_import_resolves_data_raw(monkeypatch, tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    input_path = raw_dir / "nfl_sample.csv"
    input_path.write_text(
        "Date,Visitor,PTS,Home,PTS\n"
        "Sun Sep 1 2024,1:00p,Miami Dolphins,24,New England Patriots,20\n",
        encoding="utf-8",
    )

    # Run from temporary working directory so relative paths match test data
    monkeypatch.chdir(tmp_path)
    args = Namespace(
        command="import",
        source="sports-reference",
        input=input_path.name,
        input_text=None,
        sport="nfl",
        season="2024",
        db=None,
    )

    pipeline._import_games(args)

    db_path = Path("data/db/nfl/2024.db")
    assert db_path.exists()
    games = load_games(db_path, sport="nfl", season="2024")
    assert len(games) == 1
    assert games[0].away_team == "Miami Dolphins"
    assert games[0].home_team == "New England Patriots"
