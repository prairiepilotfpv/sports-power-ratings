from pathlib import Path
import json


def test_team_aliases_present():
    p = Path("data/config/team_aliases.json")
    assert p.exists(), "team_aliases.json should exist and be versioned"
    data = json.loads(p.read_text())
    assert "nba" in data
    assert "Los Angeles Lakers" in data["nba"]
