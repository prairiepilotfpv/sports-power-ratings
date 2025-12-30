from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from cli import pipeline


def test_parse_matchup_text_variants() -> None:
    assert pipeline._parse_matchup_text("A vs B") == ("A", "B")
    assert pipeline._parse_matchup_text("A v B") == ("A", "B")
    assert pipeline._parse_matchup_text("A VS B") == ("A", "B")
    assert pipeline._parse_matchup_text("A Vs B") == ("A", "B")


def test_parse_matchup_text_requires_two_teams() -> None:
    with pytest.raises(ValueError, match="Matchup must include 'vs'"):
        pipeline._parse_matchup_text("Team A Team B")
    with pytest.raises(ValueError, match="exactly two teams"):
        pipeline._parse_matchup_text("Team A vs Team B vs Team C")


def test_next_available_path_suffixes(tmp_path: Path) -> None:
    output_path = tmp_path / "rankings.csv"
    output_path.write_text("existing", encoding="utf-8")
    suffix1 = tmp_path / "rankings-1.csv"
    suffix1.write_text("existing", encoding="utf-8")

    next_path = pipeline._next_available_path(output_path)
    assert next_path == tmp_path / "rankings-2.csv"


def test_import_games_requires_input(tmp_path: Path) -> None:
    args = Namespace(
        source="sports-reference",
        input=None,
        input_text=None,
        sport="nba",
        season="2024-25",
        db=str(tmp_path / "games.db"),
    )
    with pytest.raises(ValueError, match="Provide --input or --input-text"):
        pipeline._import_games(args)


def test_import_games_rejects_unsupported_source(tmp_path: Path) -> None:
    args = Namespace(
        source="other",
        input="games.csv",
        input_text=None,
        sport="nba",
        season="2024-25",
        db=str(tmp_path / "games.db"),
    )
    with pytest.raises(ValueError, match="Unsupported source"):
        pipeline._import_games(args)
