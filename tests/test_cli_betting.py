import sys
from src.cli import pipeline as pl


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
    monkeypatch.setattr(sys, "argv", ["prog", "betting", "review-generate", "--sport", "nba", "--season", "2025-26", "--model", "elo"])
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
