from src.parsers import paste_parser as pp


def test_parse_basic_matchup():
    text = """
Lakers vs Celtics
Lakers -3 -110
Celtics +3 -110
O/U 200.5 -110
Lakers +150
Celtics -170
"""
    rows = pp.parse_paste(text)
    # Expect at least spread, total and ML rows
    types = {r["market_type"] for r in rows}
    assert "spread" in types
    assert "total" in types
    assert "ML" in types


def test_team_matching_with_list():
    text = """
Bucks vs Heat
Bucks -5 -110
Heat +5 -110
"""
    team_list = ["Milwaukee Bucks", "Miami Heat"]
    rows = pp.parse_paste(text, team_list=team_list)
    # Should attempt to match home/away to canonical names when team_list provided
    for r in rows:
        assert r["team_home_raw"] in team_list or r["team_home_raw"] is None
        assert r["team_away_raw"] in team_list or r["team_away_raw"] is None


def test_filter_ads_and_logos():
    text = """
BetMGM WY logo
Lakers vs Celtics
Lakers -3 -110
promotion logo
"""
    rows = pp.parse_paste(text)
    # Ads should be filtered; we should still see a parsed market row
    assert any(r["selection"] and "Lakers" in str(r["selection"]) for r in rows)
