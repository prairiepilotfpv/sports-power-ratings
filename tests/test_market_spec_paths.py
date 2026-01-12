from pathlib import Path

from markets.registry import get_market_spec


def test_market_spec_ensemble_path_ml():
    spec = get_market_spec("ML")
    path = spec.ensemble_weights_path("nba", "2025-26", "ensemble_ml_v1")
    expected = Path("outputs") / "ensembles" / "nba" / "2025-26" / "ML" / "ensemble_ml_v1.json"
    assert path == expected
