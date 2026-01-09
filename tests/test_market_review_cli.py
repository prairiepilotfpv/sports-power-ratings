import sys

from src.cli import pipeline as pl
from src.data import betting_repository as br
from src.pipelines import market_review as mr


def test_market_review_argparse(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "market-review",
            "--sport",
            "nba",
            "--season",
            "2025-26",
            "--status",
            "matched,needs_review",
            "--limit",
            "3",
        ],
    )
    args = pl._parse_args()
    assert args.command == "market-review"
    assert args.status == "matched,needs_review"
    assert args.limit == 3


def test_market_review_accept_and_reject(tmp_path):
    db_path = tmp_path / "test.db"
    br.init_db(db_path)

    sid_match = br.add_staging_row(
        db_path,
        source="test",
        captured_at="2025-01-01",
        image_path=None,
        raw_text="Lakers +110",
        book="bk",
        market_type="ML",
        selection="Lakers",
        line=None,
        odds=110,
        team_home_raw="Lakers",
        team_away_raw="Clippers",
        game_date="2025-01-02",
        match_status="needs_review",
        match_confidence=0.55,
        game_id=None,
    )
    sid_reject = br.add_staging_row(
        db_path,
        source="test",
        captured_at="2025-01-01",
        image_path=None,
        raw_text="Clippers -120",
        book="bk",
        market_type="ML",
        selection="Clippers",
        line=None,
        odds=-120,
        team_home_raw="Clippers",
        team_away_raw="Lakers",
        game_date="2025-01-02",
        match_status="needs_review",
        match_confidence=0.42,
        game_id=None,
    )

    accepted = mr.accept_match(
        db_path,
        staging_id=sid_match,
        game_id="game-123",
        match_confidence=0.91,
    )
    assert accepted["match_status"] == "matched"
    assert accepted["game_id"] == "game-123"
    assert accepted["match_confidence"] == 0.91

    rejected = mr.reject_match(db_path, staging_id=sid_reject)
    assert rejected["match_status"] == "unmatched"
    assert rejected["game_id"] is None
    assert rejected["match_confidence"] == 0.0

    matched_rows = mr.list_staging_rows(db_path, match_statuses=["matched"])
    assert len(matched_rows) == 1
    assert matched_rows[0]["id"] == sid_match


def test_market_review_cli_list_and_accept(monkeypatch, capsys, tmp_path):
    db_path = tmp_path / "cli.db"
    br.init_db(db_path)

    staging_id = br.add_staging_row(
        db_path,
        source="test",
        captured_at="2025-01-01T00:00:00Z",
        image_path=None,
        raw_text="TeamA +110",
        book="bk",
        market_type="ML",
        selection="TeamA",
        line=None,
        odds=110,
        team_home_raw="TeamA",
        team_away_raw="TeamB",
        game_date="2025-01-02",
        match_status="needs_review",
        match_confidence=0.5,
        game_id=None,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "market-review",
            "--sport",
            "nba",
            "--season",
            "2025-26",
            "--db",
            str(db_path),
            "--status",
            "needs_review",
        ],
    )
    args = pl._parse_args()
    pl._run_market_review(args)
    listing = capsys.readouterr().out
    assert str(staging_id) in listing
    assert "needs_review" in listing

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "market-review",
            "--sport",
            "nba",
            "--season",
            "2025-26",
            "--db",
            str(db_path),
            "--accept",
            str(staging_id),
            "--game-id",
            "game-xyz",
            "--match-confidence",
            "0.88",
        ],
    )
    args = pl._parse_args()
    pl._run_market_review(args)
    accepted_out = capsys.readouterr().out
    assert "Accepted staging" in accepted_out

    matched_rows = br.list_staging_rows(db_path, match_statuses=["matched"])
    assert matched_rows
    assert matched_rows[0]["game_id"] == "game-xyz"
    assert matched_rows[0]["match_confidence"] == 0.88
