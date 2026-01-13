from __future__ import annotations

from pathlib import Path

import pytest

from data.repository import init_db
from pipelines.schedule import _validate_market_tuning_inputs


def test_validate_market_tuning_inputs_strict_fails(tmp_path: Path) -> None:
    db_path = tmp_path / "params.db"
    init_db(db_path)

    with pytest.raises(ValueError) as excinfo:
        _validate_market_tuning_inputs(
            db_path=db_path,
            sport="nba",
            season="2024-25",
            models=["elo"],
            ensemble_ids={},
            strict=True,
        )

    message = str(excinfo.value)
    assert "bootstrap-market-actives" in message


def test_validate_market_tuning_inputs_non_strict_warns(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "params.db"
    init_db(db_path)

    _validate_market_tuning_inputs(
        db_path=db_path,
        sport="nba",
        season="2024-25",
        models=["elo"],
        ensemble_ids={},
        strict=False,
    )

    output = capsys.readouterr().out
    assert "Missing active params for model=elo market=ML" in output
    assert "bootstrap-market-actives" in output
