from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.model_params import resolve_model_params


def test_resolve_model_params_with_file(tmp_path: Path) -> None:
    payload = {"elo": {"k_factor": 10.0}, "toor": {"max_iter": 200}}
    path = tmp_path / "params.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert resolve_model_params("elo", params_file=path) == {"k_factor": 10.0}


def test_resolve_model_params_with_inline() -> None:
    assert resolve_model_params("elo", params={"k_factor": 20.0}) == {"k_factor": 20.0}


def test_resolve_model_params_rejects_both(tmp_path: Path) -> None:
    path = tmp_path / "params.json"
    path.write_text(json.dumps({"elo": {"k_factor": 10.0}}), encoding="utf-8")

    with pytest.raises(ValueError):
        resolve_model_params("elo", params={"k_factor": 20.0}, params_file=path)
