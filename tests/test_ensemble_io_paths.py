import json

from ensemble.io import load_ml_weights


def test_load_ml_weights_falls_back_to_legacy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    legacy_dir = tmp_path / "outputs" / "ensembles" / "nba" / "2025-26"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    (legacy_dir / "ensemble_ml_v1.json").write_text(json.dumps({"model_a": 0.7}))

    market_dir = legacy_dir / "ML"
    market_dir.mkdir(parents=True, exist_ok=True)

    weights = load_ml_weights("nba", "2025-26", "ensemble_ml_v1", market="ML")
    assert weights == {"model_a": 0.7}
