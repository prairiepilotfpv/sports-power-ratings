from src.utils import identity as idu


def test_normalize_team_name():
    assert idu.normalize_team_name("  LA Clippers  ") == "la clippers"
    assert idu.normalize_team_name("L.A. Clippers") == "la clippers"


def test_resolve_team_alias():
    alias_map = {"Los Angeles Lakers": ["LA Lakers", "Lakers"]}
    assert idu.resolve_team_alias("LA Lakers", alias_map) == "Los Angeles Lakers"
    assert idu.resolve_team_alias("Los Angeles Lakers", alias_map) == "Los Angeles Lakers"
    assert idu.resolve_team_alias("Unknown", alias_map) is None


def test_fuzzy_match_team():
    candidates = ["Los Angeles Lakers", "LA Clippers", "Chicago Bulls"]
    match, score = idu.fuzzy_match_team("L A Clippers", candidates)
    assert match == "LA Clippers"
    assert score > 0.6
