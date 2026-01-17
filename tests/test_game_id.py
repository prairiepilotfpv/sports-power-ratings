from src.utils.game_id import make_game_id


def test_make_game_id_stable_insensitive():
    a = make_game_id("NBA", "2024-25", "2024-11-01", "L.A. Lakers", "Boston Celtics")
    b = make_game_id("nba", "2024-25", "2024-11-01", "la lakers", "boston celtics")
    c = make_game_id("nba", "2024-25", "2024-11-01T15:00:00", " LA Lakers ", "Boston   Celtics")
    assert a == b == c
    # format checks
    parts = a.split(":")
    assert parts[0] == "nba"
    assert parts[1] == "2024-25"
    assert parts[2] == "2024-11-01"
    assert len(parts[3]) == 12
