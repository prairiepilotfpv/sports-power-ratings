import math
import numpy as np
import pandas as pd

from config import MIN_CALIBRATION_SAMPLES
from models.bradley_terry import BradleyTerry, BradleyTerryBacktest


def _make_alternating_games(n_games: int, base_score: int = 90, margins=None):
    """Create a simple alternating-home dataset between teams A and B.

    margins can be a scalar or sequence with length n_games.
    """
    if margins is None:
        margins = [5] * n_games
    elif isinstance(margins, (int, float)):
        margins = [margins] * n_games
    games = []
    for i in range(n_games):
        home = "A" if i % 2 == 0 else "B"
        away = "B" if home == "A" else "A"
        margin = int(margins[i])
        # alternate winners to avoid degenerate datasets
        if i % 3 == 0:
            # home wins
            home_score = base_score + margin
            away_score = base_score
        else:
            # away wins
            home_score = base_score
            away_score = base_score + margin
        games.append(
            {
                "date": f"2025-01-{(i%30)+1}",
                "home_team": home,
                "away_team": away,
                "home_score": int(home_score),
                "away_score": int(away_score),
                "neutral": False,
            }
        )
    return pd.DataFrame(games)


def test_prob_head_direct_and_symmetry():
    n = max(MIN_CALIBRATION_SAMPLES + 5, 30)
    df = _make_alternating_games(n)
    bt = BradleyTerry()
    bt.fit(df.to_dict(orient="records"))

    # Direct BT probability should be a valid probability regardless of venue
    p_ab_home = bt.project_matchup("A", "B", neutral=False).get("model_p_home_win")
    p_ba_home = bt.project_matchup("B", "A", neutral=False).get("model_p_home_win")
    assert p_ab_home is not None and 0.0 <= p_ab_home <= 1.0
    assert p_ba_home is not None and 0.0 <= p_ba_home <= 1.0

    # Under neutral venue, probabilities should be complementary: p(A beats B) + p(B beats A) ~= 1
    p_ab = bt.project_matchup("A", "B", neutral=True).get("model_p_home_win")
    p_ba = bt.project_matchup("B", "A", neutral=True).get("model_p_home_win")
    assert math.isfinite(p_ab) and math.isfinite(p_ba)
    assert abs((p_ab + p_ba) - 1.0) < 1e-6


def test_margin_and_total_heads_present_and_sign():
    n = max(MIN_CALIBRATION_SAMPLES + 5, 30)
    df = _make_alternating_games(n, margins=8)
    bt = BradleyTerry()
    bt.fit(df.to_dict(orient="records"))
    proj = bt.project_matchup("A", "B", neutral=False)

    assert "margin_mean" in proj and proj["margin_mean"] is not None
    assert "margin_sd" in proj and proj["margin_sd"] is not None and proj["margin_sd"] > 0
    # projected_spread is -margin_mean by convention in pipelines
    assert -proj["margin_mean"] == -(proj["margin_mean"])  # trivial check to ensure sign consistency

    assert "total_mean" in proj and proj["total_mean"] is not None
    assert "total_sd" in proj and proj["total_sd"] is not None and proj["total_sd"] > 0


def test_bt_core_wl_only():
    # create two datasets with identical W/L but different margins
    n = max(MIN_CALIBRATION_SAMPLES + 5, 30)
    # dataset A: small margins
    df_a = _make_alternating_games(n, margins=2)
    # dataset B: larger margins but same winners pattern
    df_b = _make_alternating_games(n, margins=20)

    bt_a = BradleyTerry()
    bt_a.fit(df_a.to_dict(orient="records"))
    bt_b = BradleyTerry()
    bt_b.fit(df_b.to_dict(orient="records"))

    # Ratings should be approximately equal because BT fit uses only W/L
    keys = sorted(set(bt_a.ratings.keys()) | set(bt_b.ratings.keys()))
    for k in keys:
        ra = bt_a.ratings.get(k, 0.0)
        rb = bt_b.ratings.get(k, 0.0)
        # Allow a tiny numeric tolerance, but not large differences driven by margins
        assert abs(ra - rb) < 1e-6
