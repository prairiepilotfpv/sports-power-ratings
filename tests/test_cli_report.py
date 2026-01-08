import sys
from pathlib import Path
import tempfile

from src.cli import pipeline as pl
from src.data import repository as repo
from src.data import betting_repository as br


def test_cli_report_writes_csv(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        repo.init_db(db_path)
        br.init_db(db_path)
        # seed a game
        repo.save_games(db_path, [repo.GameResult(date=repo.date.fromisoformat("2025-11-10"), home_team="H", away_team="A", home_score=None, away_score=None, neutral=False, overtime=False, decision_type=None, game_id="g1", sport="nba", season="2025-26", division=None, conference=None, notes=None)])
        out = Path(td) / "report.csv"
        monkeypatch.setattr(sys, "argv", ["prog", "betting", "report", "--sport", "nba", "--season", "2025-26", "--db", str(db_path), "--output", str(out)])
        pl.main()
        assert out.exists()
