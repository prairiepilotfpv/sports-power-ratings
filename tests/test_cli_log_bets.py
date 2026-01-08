import sys
from pathlib import Path
import tempfile

from src.cli import pipeline as pl


def test_cli_log_bets_parsing(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "betting", "log-bets", "--workbook", "wb.xlsx", "--db", "db.sqlite", "--writeback"])
    args = pl._parse_args()
    assert args.command == "betting"
    assert args.betting_cmd == "log-bets"
    assert args.writeback is True


def test_cli_log_bets_dry_run(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "betting", "log-bets", "--workbook", "wb.xlsx", "--db", "db.sqlite", "--dry-run"])
    args = pl._parse_args()
    assert args.command == "betting"
    assert args.betting_cmd == "log-bets"
    assert args.dry_run is True
