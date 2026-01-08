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
