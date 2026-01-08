from src.utils.odds import american_to_implied, expected_value


def test_american_to_implied_positive():
    assert abs(american_to_implied(200) - (100 / 300)) < 1e-6


def test_american_to_implied_negative():
    assert abs(american_to_implied(-150) - (150 / 250)) < 1e-6


def test_expected_value_with_odds_positive():
    # odds +200, model_prob 0.5 -> payout = 2.0, EV = 0.5*2 - 0.5 = 0.5
    ev = expected_value(implied_prob=0.0, model_prob=0.5, odds=200)
    assert abs(ev - 0.5) < 1e-6


def test_expected_value_with_odds_negative():
    # odds -150, payout = 100/150 = 0.666666..., model_prob 0.6
    ev = expected_value(implied_prob=0.0, model_prob=0.6, odds=-150)
    expected = (0.6 * (100.0 / 150.0)) - 0.4
    assert abs(ev - expected) < 1e-6
