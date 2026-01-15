import json
import os
import sqlite3
from datetime import date
from pathlib import Path
import tempfile

from src.data import betting_repository as br
from src.data import repository as repo


def _create_sample_game(db_path: Path, *, game_date: date) -> None:
    repo.init_db(db_path)
    repo.save_games(
        db_path,
        [
            repo.GameResult(
                date=game_date,
                home_team="Los Angeles Lakers",
                away_team="Boston Celtics",
                home_score=None,
                away_score=None,
                neutral=False,
                overtime=False,
                decision_type=None,
                game_id="2025-01-02-lal-bos",
                sport="nba",
                season="2025-26",
                division=None,
                conference=None,
                notes=None,
            )
        ],
    )


def test_auto_match_staging_rows_matches_single_family(tmp_path: Path, monkeypatch) -> None:
    alias_path = tmp_path / "team_aliases.json"
    monkeypatch.setenv("TEAM_ALIAS_FILE", str(alias_path))
    db_path = tmp_path / "betting.db"
    _create_sample_game(db_path, game_date=date(2025, 1, 2))
    br.init_db(db_path)
    staging_id = br.add_staging_row(
        db_path,
        source="test",
        captured_at="2025-01-01T10:00:00Z",
        image_path=None,
        raw_text=None,
        book="test-book",
        market_type="ML",
        selection="Los Angeles Lakers",
        line=0.0,
        odds=110,
        team_home_raw="Los Angeles Lakers",
        team_away_raw="Boston Celtics",
        game_date="2025-01-02",
        match_status="unmatched",
        match_confidence=0.0,
        game_id=None,
        hold_reason=None,
    )

    summary = br.auto_match_staging_rows(db_path, sport="nba", season="2025-26")
    assert summary["matched"] == 1
    assert summary["total"] == 1
    assert staging_id in summary["matched_ids"]
    assert summary["skipped"] == 0

    if alias_path.exists():
        alias_data = json.loads(alias_path.read_text())
        assert "nba" in alias_data
        assert "Los Angeles Lakers" in alias_data["nba"] or "Boston Celtics" in alias_data["nba"]


def test_auto_match_staging_rows_records_alias_for_short_name(tmp_path: Path, monkeypatch) -> None:
    alias_path = tmp_path / "team_aliases.json"
    monkeypatch.setenv("TEAM_ALIAS_FILE", str(alias_path))
    db_path = tmp_path / "betting.db"
    repo.init_db(db_path)
    repo.save_games(
        db_path,
        [
            repo.GameResult(
                date=date(2025, 1, 2),
                home_team="Dallas Mavericks",
                away_team="Utah Jazz",
                home_score=None,
                away_score=None,
                neutral=False,
                overtime=False,
                decision_type=None,
                game_id="2025-01-02-dal-uta",
                sport="nba",
                season="2025-26",
                division=None,
                conference=None,
                notes=None,
            )
        ],
    )
    br.init_db(db_path)
    staging_id = br.add_staging_row(
        db_path,
        source="test",
        captured_at="2025-01-01T10:00:00Z",
        image_path=None,
        raw_text=None,
        book="test-book",
        market_type="spread",
        selection="Dallas Mavericks",
        line=-3.5,
        odds=-110,
        team_home_raw="Mavs",
        team_away_raw="Jazz",
        game_date="2025-01-02",
        match_status="unmatched",
        match_confidence=0.0,
        game_id=None,
        hold_reason=None,
    )

    summary = br.auto_match_staging_rows(db_path, sport="nba", season="2025-26")
    assert summary["matched"] == 1
    assert summary["total"] == 1
    assert staging_id in summary["matched_ids"]
    assert summary["skipped"] == 0

    assert alias_path.exists()
    alias_data = json.loads(alias_path.read_text())
    assert "nba" in alias_data
    assert "Dallas Mavericks" in alias_data["nba"]
    assert "Mavs" in alias_data["nba"]["Dallas Mavericks"]

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT match_status, game_id FROM market_snapshot_staging WHERE id = ?", (staging_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == "matched"
    assert row[1] == "2025-01-02-dal-uta"


def test_auto_match_staging_rows_skips_when_no_game(tmp_path: Path, monkeypatch) -> None:
    alias_path = tmp_path / "team_aliases.json"
    monkeypatch.setenv("TEAM_ALIAS_FILE", str(alias_path))
    db_path = tmp_path / "betting.db"
    repo.init_db(db_path)
    br.init_db(db_path)
    br.add_staging_row(
        db_path,
        source="test",
        captured_at="2025-01-01T10:00:00Z",
        image_path=None,
        raw_text=None,
        book="test-book",
        market_type="ML",
        selection="Some Team",
        line=0.0,
        odds=120,
        team_home_raw="Some Team",
        team_away_raw="Other Team",
        game_date="2025-01-02",
        match_status="unmatched",
        match_confidence=0.0,
        game_id=None,
        hold_reason=None,
    )

    summary = br.auto_match_staging_rows(db_path, sport="nba", season="2025-26")
    assert summary["matched"] == 0
    assert summary["total"] == 1
    assert summary["skipped"] == 1
