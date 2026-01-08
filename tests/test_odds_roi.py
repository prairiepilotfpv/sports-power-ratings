from src.utils.odds import payout_per_unit, roi_from_profit_and_stake


def test_payout_per_unit_positive():
    assert abs(payout_per_unit(200) - 2.0) < 1e-6


def test_payout_per_unit_negative():
    assert abs(payout_per_unit(-150) - (100.0/150.0)) < 1e-6


def test_roi_from_profit_and_stake():
    assert roi_from_profit_and_stake(10, 100) == 0.1
    assert roi_from_profit_and_stake(0, 0) is None
