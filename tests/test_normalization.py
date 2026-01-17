from utils import identity as idu
from utils.normalization import (
    normalize_evaluation_market_type,
    normalize_market_type_value,
    normalize_team_label,
    normalize_total_selection,
)


def test_normalize_market_type_value():
    assert normalize_market_type_value("Money Line") == "ML"
    assert normalize_market_type_value("ml") == "ML"
    assert normalize_market_type_value("spread") == "spread"
    assert normalize_market_type_value("Over/Under") == "total"


def test_normalize_evaluation_market_type():
    assert normalize_evaluation_market_type("ML") == "moneyline"
    assert normalize_evaluation_market_type("moneyline") == "moneyline"
    assert normalize_evaluation_market_type("total") == "total"


def test_normalize_total_selection():
    assert normalize_total_selection("Over") == "Over"
    assert normalize_total_selection("u") == "Under"
    assert normalize_total_selection("Push") is None


def test_normalize_team_label_from_aliases():
    alias_map = idu.load_alias_map("nba")
    assert normalize_team_label("Lakers", alias_map=alias_map) == "Los Angeles Lakers"
