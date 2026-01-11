import csv
import sqlite3
import sys

from src.cli import pipeline as pl
from src.data import betting_repository as br


def test_betting_market_ocr_parsing(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "betting", "market-ocr", "--sport", "nba", "--season", "2025-26", "--images", "imgdir"])
    args = pl._parse_args()
    assert args.command == "betting"
    assert args.betting_cmd == "market-ocr"
    assert args.images == "imgdir"


def test_betting_market_commit_parsing(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "betting", "market-commit", "--sport", "nba", "--season", "2025-26", "--snapshot-run-id", "run1"])
    args = pl._parse_args()
    assert args.command == "betting"
    assert args.betting_cmd == "market-commit"
    assert args.snapshot_run_id == "run1"


def test_betting_review_generate_parsing(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "betting",
            "review-generate",
            "--sport",
            "nba",
            "--season",
            "2025-26",
            "--model",
            "elo",
            "--snapshot-run-id",
            "run1",
        ],
    )
    args = pl._parse_args()
    assert args.command == "betting"
    assert args.betting_cmd == "review-generate"
    assert args.model == "elo"
    assert args.include_ocr_raw is True


def test_betting_market_ocr_json_output_parsing(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "betting",
            "market-ocr",
            "--sport",
            "nba",
            "--season",
            "2025-26",
            "--images",
            "imgdir",
            "--json-output",
            "tmp/out.json",
        ],
    )
    args = pl._parse_args()
    assert args.command == "betting"
    assert args.betting_cmd == "market-ocr"
    assert args.json_output == "tmp/out.json"


def test_betting_report_type_and_format_parsing(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "betting",
            "report",
            "--sport",
            "nba",
            "--season",
            "2025-26",
            "--type",
            "weekly",
            "--format",
            "xlsx",
        ],
    )
    args = pl._parse_args()
    assert args.command == "betting"
    assert args.betting_cmd == "report"
    assert args.report_type == "weekly"
    assert args.format == "xlsx"


def test_betting_clv_csv_parsing(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "betting",
            "clv-csv",
            "--sport",
            "nba",
            "--season",
            "2025-26",
            "--csv",
            "clv.csv",
            "--default-market-type",
            "ML",
            "--no-update-bets",
        ],
    )
    args = pl._parse_args()
    assert args.command == "betting"
    assert args.betting_cmd == "clv-csv"
    assert args.csv_path == "clv.csv"
    assert args.default_market_type == "ML"
    assert args.no_update_bets is True


def test_betting_market_csv_parsing(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "betting",
            "market-csv",
            "--sport",
            "nba",
            "--season",
            "2025-26",
            "--csv",
            "markets.csv",
            "--snapshot-run-id",
            "run-csv",
            "--default-book",
            "dn",
            "--no-commit-matched",
        ],
    )
    args = pl._parse_args()
    assert args.command == "betting"
    assert args.betting_cmd == "market-csv"
    assert args.csv_path == "markets.csv"
    assert args.snapshot_run_id == "run-csv"
    assert args.default_book == "dn"
    assert args.commit_matched is False


def test_betting_market_csv_ingestion_counts_and_placement(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "betting.db"
    br.init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO games (
                date, home_team, away_team, home_score, away_score, neutral, overtime,
                decision_type, game_id, sport, season, division, conference, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2025-01-02",
                "Los Angeles Lakers",
                "Boston Celtics",
                None,
                None,
                0,
                0,
                None,
                "2025-01-02-lal-bos",
                "nba",
                "2025-26",
                None,
                None,
                None,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    csv_path = tmp_path / "markets.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "market_type",
                "selection",
                "line",
                "odds",
                "team_home",
                "team_away",
                "game_date",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "market_type": "spread",
                "selection": "Los Angeles Lakers",
                "line": "-3.5",
                "odds": "-110",
                "team_home": "Los Angeles Lakers",
                "team_away": "Boston Celtics",
                "game_date": "2025-01-02",
            }
        )
        writer.writerow(
            {
                "market_type": "total",
                "selection": "over",
                "line": "215.5",
                "odds": "-105",
                "team_home": "New York Knicks",
                "team_away": "Chicago Bulls",
                "game_date": "2025-01-02",
            }
        )
        writer.writerow(
            {
                "market_type": "spread",
                "selection": "Boston Celtics",
                "line": "+3.5",
                "odds": "2001",
                "team_home": "Los Angeles Lakers",
                "team_away": "Boston Celtics",
                "game_date": "2025-01-02",
            }
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "betting",
            "market-csv",
            "--sport",
            "nba",
            "--season",
            "2025-26",
            "--csv",
            str(csv_path),
            "--snapshot-run-id",
            "run-csv",
            "--db",
            str(db_path),
            "--default-book",
            "dn",
        ],
    )
    args = pl._parse_args()
    pl._run_betting(args)
    output = capsys.readouterr().out
    assert "committed=1" in output
    assert "staged=1" in output
    assert "rejected=1" in output

    conn = sqlite3.connect(db_path)
    try:
        committed = conn.execute("SELECT COUNT(*) FROM market_snapshots").fetchone()[0]
        staged = conn.execute("SELECT COUNT(*) FROM market_snapshot_staging").fetchone()[0]
        snapshot = conn.execute(
            "SELECT snapshot_run_id, game_id FROM market_snapshots"
        ).fetchone()
        staged_status = conn.execute(
            "SELECT match_status FROM market_snapshot_staging"
        ).fetchone()[0]
    finally:
        conn.close()

    assert committed == 1
    assert staged == 1
    assert snapshot == ("run-csv", "2025-01-02-lal-bos")
    assert staged_status == "unmatched"
