from pathlib import Path

import pytest

from ingest.sports_reference import parse_sr_csv, parse_sr_csv_text, parse_sr_html


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "sports_reference" / "nba"


def test_parse_sr_csv() -> None:
    games = parse_sr_csv(FIXTURES_DIR / "2023.csv", sport="nba", season="2023-24")

    assert len(games) == 2
    assert games[0].away_team == "Atlanta Hawks"
    assert games[0].home_team == "Boston Celtics"
    assert games[0].away_score == 110
    assert games[0].home_score == 120
    assert games[0].overtime is False
    assert games[0].game_id == "202401010BOS"
    assert games[0].sport == "nba"
    assert games[0].season == "2023-24"


def test_parse_sr_html() -> None:
    games = parse_sr_html(FIXTURES_DIR / "2023.html", sport="nba", season="2023-24")

    assert len(games) == 2
    assert games[1].away_team == "Chicago Bulls"
    assert games[1].home_team == "Detroit Pistons"
    assert games[1].away_score == 99
    assert games[1].home_score == 101
    assert games[1].overtime is True
    assert games[1].game_id == "202401020DET"


def test_parse_sr_csv_text() -> None:
    text = (
        "Date,Visitor/Neutral,PTS,Home/Neutral,PTS,,,LOG,Notes\n"
        "Fri Nov 1 2024,Boston Celtics,124,Charlotte Hornets,109,Box Score,,2:17,\n"
        "Fri Nov 1 2024,Orlando Magic,109,Cleveland Cavaliers,120,Box Score,,2:20,\n"
    )
    games = parse_sr_csv_text(text, sport="nba", season="2024-25")
    assert len(games) == 2
    assert games[0].away_team == "Boston Celtics"
    assert games[0].home_team == "Charlotte Hornets"
    assert games[0].away_score == 124
    assert games[0].home_score == 109


@pytest.mark.parametrize(
    "parser,fixture",
    [
        (parse_sr_csv, "missing_columns.csv"),
        (parse_sr_html, "missing_columns.html"),
    ],
)
def test_missing_required_columns_raises(parser, fixture: str) -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        parser(FIXTURES_DIR / fixture)
