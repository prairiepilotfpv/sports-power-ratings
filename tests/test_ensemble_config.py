from __future__ import annotations

import json
from pathlib import Path

import pytest

from ensemble.config import load_ensemble_config, validate_ensemble_config


def test_validate_normalizes_and_warns() -> None:
    raw = {
        "sport": "nba",
        "season": "2025-26",
        "markets": {
            "ML": {
                "ensemble_id": "ensemble_ml_v1",
                "metric_slot": "log_loss",
                "models": ["elo", "gssd"],
                "weights": {"elo": 2, "gssd": 1, "extra": 5},
            }
        },
    }
    normalized, warnings = validate_ensemble_config(raw)
    market = normalized["markets"]["ML"]
    weights = market["weights"]
    assert pytest.approx(sum(weights.values())) == 1.0
    assert set(weights) == {"elo", "gssd"}
    # Weights should be renormalized and trimmed; warnings should reflect it.
    joined = " ".join(warnings)
    assert "renormalized" in joined
    assert "trimmed" in joined


def test_load_ensemble_config_reads_meta(tmp_path: Path) -> None:
    cfg_path = tmp_path / "outputs" / "ensembles" / "nba" / "2025-26"
    cfg_path.mkdir(parents=True)
    config_file = cfg_path / "ensemble_config.json"
    payload = {
        "sport": "nba",
        "season": "2025-26",
        "markets": {
            "ML": {"ensemble_id": "ensemble_ml_v1", "metric_slot": "log_loss"}
        },
    }
    config_file.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_ensemble_config("nba", "2025-26", path_override=config_file)
    assert loaded is not None
    assert loaded["_meta"]["path"] == str(config_file)
    assert loaded["_meta"]["sha256"]
    # sport/season defaulting should keep values even if omitted in file
    assert loaded["sport"] == "nba"
    assert loaded["season"] == "2025-26"
