from pathlib import Path

import pytest

from ingest.sports_reference import parse_sr_csv, parse_sr_html


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "sports_reference" / "nba"


def test_parse_sr_csv() -> None:
    games = parse_sr_csv(FIXTURES_DIR / "2023.csv", sport="nba", season="2023-24")

    assert len(games) == 2
    assert games[0].visitor_team == "Atlanta Hawks"
    assert games[0].home_team == "Boston Celtics"
    assert games[0].visitor_pts == 110
    assert games[0].home_pts == 120
    assert games[0].ot is False
    assert games[0].game_id == "202401010BOS"
    assert games[0].sport == "nba"
    assert games[0].season == "2023-24"


def test_parse_sr_html() -> None:
    games = parse_sr_html(FIXTURES_DIR / "2023.html", sport="nba", season="2023-24")

    assert len(games) == 2
    assert games[1].visitor_team == "Chicago Bulls"
    assert games[1].home_team == "Detroit Pistons"
    assert games[1].visitor_pts == 99
    assert games[1].home_pts == 101
    assert games[1].ot is True
    assert games[1].game_id == "202401020DET"


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
