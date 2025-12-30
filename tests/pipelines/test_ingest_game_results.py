from __future__ import annotations

from pathlib import Path

import pytest

from pipelines import ingest_game_results


def test_ingest_workbook_to_csv(tmp_path: Path) -> None:
    input_path = tmp_path / "schedule.csv"
    input_path.write_text(
        "Date,Visitor/Neutral,PTS,Home/Neutral,PTS,OT\n"
        "2024-01-01,Away,90,Home,100,\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "out.csv"

    count = ingest_game_results.ingest_workbook_to_csv(input_path, output_path)

    assert count == 1
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "Away" in content
    assert "Home" in content


def test_ingest_html_to_csv_requires_parser(tmp_path: Path) -> None:
    input_path = tmp_path / "schedule.html"
    input_path.write_text("<html></html>", encoding="utf-8")

    with pytest.raises(ImportError, match="parse_sr_scores"):
        ingest_game_results.ingest_html_to_csv(input_path, tmp_path / "out.csv")


def test_ingest_image_to_csv_uses_structured_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_ocr(path: str) -> str:
        return "fake text"

    def fake_extract(text: str):
        return [
            {
                "date": "2024-01-01",
                "away_team": "Away",
                "away_score": 90,
                "home_team": "Home",
                "home_score": 100,
                "overtime": False,
                "game_id": "gid-1",
                "notes": None,
            }
        ]

    monkeypatch.setattr(ingest_game_results, "ocr_image", fake_ocr)
    monkeypatch.setattr(ingest_game_results, "_extract_games_structured", fake_extract)

    output_path = tmp_path / "out.csv"
    count = ingest_game_results.ingest_image_to_csv(tmp_path / "image.png", output_path)

    assert count == 1
    assert output_path.exists()


def test_extract_games_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY not set"):
        ingest_game_results._extract_games_structured("text")
