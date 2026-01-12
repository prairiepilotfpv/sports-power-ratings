from __future__ import annotations

from pathlib import Path

from ensemble.config import load_ensemble_config


def test_load_ensemble_config_uses_global_defaults(monkeypatch, tmp_path):
    """When no user configs exist, global defaults should be applied per market."""
    monkeypatch.chdir(tmp_path)
    config = load_ensemble_config(
        sport="nba",
        season="2025-26",
        available_models=["elo", "bradley-terry", "gssd", "toor", "poisson"],
    )
    markets = config.get("markets", {})
    meta = (config.get("_meta", {}) or {}).get("markets", {})
    assert markets["ML"]["models"] == ["elo", "bradley-terry"]
    assert markets["SPREAD"]["models"] == ["elo", "gssd", "toor"]
    assert markets["TOTAL"]["models"] == ["poisson", "gssd"]
    assert meta.get("ML", {}).get("source") == "global_default"
    assert Path(meta.get("ML", {}).get("path")).name == "ML.json"


def test_custom_market_config_overrides_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    custom_path = Path("outputs") / "ensembles" / "nba" / "2025-26" / "ML"
    custom_path.mkdir(parents=True, exist_ok=True)
    custom_file = custom_path / "ensemble_ml_v1.json"
    custom_file.write_text(
        """
        {
          "sport": "nba",
          "season": "2025-26",
          "market": "ML",
          "ensemble_id": "ensemble_ml_v1",
          "metric_slot": "log_loss",
          "models": ["elo", "custom"],
          "weights": {"elo": 0.7, "custom": 0.3}
        }
        """,
        encoding="utf-8",
    )
    default_market = Path("outputs") / "ensembles" / "nba" / "2025-26" / "ML" / "default.json"
    default_market.write_text(
        """
        {"market": "ML", "ensemble_id": "ensemble_ml_v1", "models": ["elo"]}
        """,
        encoding="utf-8",
    )

    config = load_ensemble_config(
        sport="nba",
        season="2025-26",
        available_models=["elo", "custom"],
    )
    ml_config = config["markets"]["ML"]
    ml_meta = config["_meta"]["markets"]["ML"]

    assert ml_config["models"] == ["elo", "custom"]
    assert ml_meta["source"] == "custom"
    assert ml_meta["path"].endswith("ensemble_ml_v1.json")
    # Other markets should still resolve (fallback to defaults/global files)
    assert "SPREAD" in config["markets"]
    assert "TOTAL" in config["markets"]


def test_ensemble_docs_exist():
    assert Path("docs/ensembles.md").exists()
