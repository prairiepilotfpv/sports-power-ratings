from __future__ import annotations

from src.utils.action_paste_parser import parse_action_paste, game_to_bets_hi_block


SAMPLE = """
Nets at Pelicans Odds
Spread, Total, Moneyline
Matchup
Open
Spread
Total
Moneyline
Nets logo
Nets
11-26
-1.5
-1.5
-112
o228.5
-110
-125
Pelicans logo
Pelicans
9-32
u230.5
+1.5
-108
u228.5
-110
+105
location pin
Wednesday 6:00 p.m.
January 14, 2026
"""


def test_parse_action_paste_basic():
    games = parse_action_paste(SAMPLE)
    assert len(games) == 1
    g = games[0]
    # current fields present
    assert g.ml_away_odds is not None
    assert g.ml_home_odds is not None
    assert g.spread_away_line is not None
    assert g.spread_home_line is not None
    assert g.total_line is not None

    block = game_to_bets_hi_block(g)
    lines = block.strip().splitlines()
    assert len(lines) == 6
    for ln in lines:
        parts = ln.split("\t")
        assert len(parts) == 2

    # first two ML lines must start with 0
    assert lines[0].split("\t")[0] == "0"
    assert lines[1].split("\t")[0] == "0"
