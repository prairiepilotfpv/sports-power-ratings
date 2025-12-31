from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from parsers.sr_table_parser import parse_sr_workbook


def test_parse_sr_workbook_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "schedule.csv"
    csv_path.write_text(
        "Date,Visitor/Neutral,PTS,Home/Neutral,PTS,OT,Notes\n"
        "2024-01-01,Away,90,Home,100,OT,Note\n",
        encoding="utf-8",
    )

    rows = parse_sr_workbook(csv_path)

    assert len(rows) == 1
    assert rows[0]["away_team"] == "Away"
    assert rows[0]["home_team"] == "Home"
    assert rows[0]["overtime"] == "OT"
    assert rows[0]["notes"] == "Note"
    assert "game_id" in rows[0]


def test_parse_sr_workbook_missing_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    excel_path = tmp_path / "schedule.xlsx"
    excel_path.write_text("fake", encoding="utf-8")

    def raise_import_error(*_args, **_kwargs):
        raise ImportError("missing engine")

    monkeypatch.setattr(pd, "read_excel", raise_import_error)

    with pytest.raises(RuntimeError, match="Missing Excel engine"):
        parse_sr_workbook(excel_path)
