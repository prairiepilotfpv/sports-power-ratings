"""Regression tests for market OCR bundling and confidence heuristics.

Tests exercise the OCR parser without real Tesseract/images by mocking
ocr_image to return fixture text dumps. This enables fast, deterministic
validation of team bundling, gap flags, and confidence scoring.
"""

import tempfile
from pathlib import Path
from unittest import mock
import json

from src.pipelines.market_ocr import (
    ingest_screenshots,
    parse_market_line,
    _bundle_team_markets,
    _compute_line_confidence,
)
from src.data import betting_repository as br
from src.data import repository as repo


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ocr"


def test_parse_market_line_moneyline():
    """Verify moneyline parsing: team name + odds."""
    result = parse_market_line("Lakers +110")
    assert result is not None
    assert result["market_type"] == "ML"
    assert result["selection"] == "Lakers"
    assert result["odds"] == 110
    assert result["line"] is None


def test_parse_market_line_spread():
    """Verify spread parsing: team name + line + odds."""
    result = parse_market_line("Lakers -3.5 -110")
    assert result is not None
    assert result["market_type"] == "spread"
    assert result["line"] == -3.5
    assert result["odds"] == -110


def test_parse_market_line_total_over():
    """Verify total parsing: team name + 'Over' keyword + line + odds."""
    result = parse_market_line("Over 215.5 -110")
    assert result is not None
    assert result["market_type"] == "total"
    assert result["line"] == 215.5
    assert result["odds"] == -110


def test_parse_market_line_total_under():
    """Verify total parsing with 'Under' keyword."""
    result = parse_market_line("Under 205.5 +100")
    assert result is not None
    assert result["market_type"] == "total"
    assert result["line"] == 205.5
    assert result["odds"] == 100


def test_compute_line_confidence_valid_odds():
    """Verify confidence scoring with valid odds."""
    parsed = {"odds": 110, "market_type": "ML", "selection": "Lakers", "raw": "Lakers +110"}
    confidence = _compute_line_confidence(parsed, "Lakers +110", font_row_index=1)
    assert confidence["odds_confidence"] == 1.0
    assert confidence["font_row_index"] == 1
    assert "Lakers" in confidence["keywords"]


def test_compute_line_confidence_header_penalty():
    """Verify row position penalty for header lines."""
    parsed = {"odds": 110, "market_type": "ML", "selection": "Header", "raw": "Header +110"}
    confidence = _compute_line_confidence(parsed, "Header +110", font_row_index=0)
    # Row index 0 should incur a penalty
    assert confidence["combined_confidence"] < confidence["odds_confidence"]


def test_compute_line_confidence_missing_odds():
    """Verify low confidence when odds are missing."""
    parsed = {"odds": None, "market_type": "ML", "selection": "Lakers", "raw": "Lakers"}
    confidence = _compute_line_confidence(parsed, "Lakers", font_row_index=1)
    assert confidence["odds_confidence"] == 0.3  # fallback score
    assert confidence["combined_confidence"] <= 0.5


def test_compute_line_confidence_total_keywords():
    """Verify confidence boost for total with keyword evidence."""
    parsed = {"odds": -110, "market_type": "total", "selection": None, "line": 215.5, "raw": "Over 215.5 -110"}
    confidence = _compute_line_confidence(parsed, "Over 215.5 -110", font_row_index=3)
    assert confidence["market_confidence"] == 1.0
    assert "over" in confidence["keywords"] or "under" in confidence["keywords"]


def test_bundle_three_markets_per_team():
    """Verify team bundling produces ordered ML, spread, total."""
    parsed_lines = [
        ({"market_type": "ML", "selection": "Lakers", "odds": 110, "line": None, "raw": "Lakers +110"}, 1),
        ({"market_type": "spread", "selection": "Lakers", "odds": -110, "line": -3.5, "raw": "Lakers -3.5 -110"}, 2),
        ({"market_type": "total", "selection": None, "odds": -110, "line": 215.5, "raw": "Over 215.5 -110"}, 3),
    ]
    bundle = _bundle_team_markets(parsed_lines, "Los Angeles Lakers", "LA Clippers")
    
    assert bundle["team_home_raw"] == "Los Angeles Lakers"
    assert bundle["team_away_raw"] == "LA Clippers"
    assert len(bundle["markets"]) == 3
    assert bundle["markets"][0]["market_type"] == "ML"
    assert bundle["markets"][1]["market_type"] == "spread"
    assert bundle["markets"][2]["market_type"] == "total"
    assert bundle["gap_flags"] == []


def test_bundle_gap_flags_missing_spread():
    """Verify gap flags when market types are missing."""
    parsed_lines = [
        ({"market_type": "ML", "selection": "Lakers", "odds": 110, "line": None, "raw": "Lakers +110"}, 1),
        ({"market_type": "total", "selection": None, "odds": -110, "line": 215.5, "raw": "Over 215.5 -110"}, 3),
    ]
    bundle = _bundle_team_markets(parsed_lines, "Lakers", "Clippers")
    
    assert len(bundle["markets"]) == 2
    assert "spread" in bundle["gap_flags"]
    assert bundle["markets"][0]["market_type"] == "ML"
    assert bundle["markets"][1]["market_type"] == "total"
    assert len(bundle["gap_flags"]) == 1


def test_bundle_gap_flags_missing_ml_and_spread():
    """Verify multiple gap flags."""
    parsed_lines = [
        ({"market_type": "total", "selection": None, "odds": -110, "line": 215.5, "raw": "Over 215.5 -110"}, 2),
    ]
    bundle = _bundle_team_markets(parsed_lines, "Lakers", "Clippers")
    
    assert len(bundle["markets"]) == 1
    assert "ML" in bundle["gap_flags"]
    assert "spread" in bundle["gap_flags"]


def test_golden_fixture_parsing_fixture_1():
    """Regression test: load fixture text 1 and compare JSON to expected."""
    text_path = FIXTURE_DIR / "sample_output_text_1.txt"
    expected_path = FIXTURE_DIR / "expected_bundle_1.json"
    
    if not text_path.exists() or not expected_path.exists():
        # Skip if fixtures not present
        return
    
    text = text_path.read_text()
    expected = json.loads(expected_path.read_text())

    # Debug: ensure parse_market_line works on the fixture text
    print("\n=== parse_market_line on fixture lines ===")
    for i, ln in enumerate(text.splitlines()):
        parsed = parse_market_line(ln)
        print(f"  Line {i}: {ln!r:30} -> {(parsed or {}).get('market_type', 'NONE'):>8} {parsed}")
    
    # Mock OCR to return fixture text
    def fake_ocr(img):
        print(f"  [fake_ocr called for {img}]")
        return text
    
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        repo.init_db(db_path)
        br.init_db(db_path)
        
        img = Path(td) / "img1.png"
        img.write_text("dummy")
        
        json_out = Path(td) / "out.json"

        print("\n=== running ingest_screenshots (fixture 1) ===")
        
        with mock.patch("src.ocr.ocr.ocr_image", side_effect=fake_ocr):
            count = ingest_screenshots(
                [str(img)],
                db_path=db_path,
                sport="nba",
                season="2025-26",
                source="screenshot",
                book="BookA",
                captured_at="2025-11-10",
                json_output=str(json_out),
            )
        
        assert count == 1
        result = json.loads(json_out.read_text())
        
        # Debug: print what we got
        print(f"\n\nFixture text:\n{text}")
        print(f"\nJSON result ({len(result)} bundles):\n{json.dumps(result, indent=2)}")
        dbg_path = json_out.parent / (json_out.stem + ".debug.json")
        if dbg_path.exists():
            print(f"\nDebug file found: {dbg_path}\n{dbg_path.read_text()}")
        else:
            print("\nNo debug file found.")
        
        assert len(result) == 1, f"Expected 1 bundle, got {len(result)}"
        
        # Verify structure
        bundle = result[0]
        assert bundle["team_home_raw"] == expected[0]["team_home_raw"]
        assert bundle["team_away_raw"] == expected[0]["team_away_raw"]
        assert len(bundle["markets"]) == len(expected[0]["markets"])
        assert bundle["gap_flags"] == expected[0]["gap_flags"]


def test_golden_fixture_parsing_fixture_2():
    """Regression test: load fixture text 2 and compare JSON to expected."""
    text_path = FIXTURE_DIR / "sample_output_text_2.txt"
    expected_path = FIXTURE_DIR / "expected_bundle_2.json"
    
    if not text_path.exists() or not expected_path.exists():
        # Skip if fixtures not present
        return
    
    text = text_path.read_text()
    expected = json.loads(expected_path.read_text())
    
    def fake_ocr(img):
        return text
    
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        repo.init_db(db_path)
        br.init_db(db_path)
        
        img = Path(td) / "img1.png"
        img.write_text("dummy")
        
        json_out = Path(td) / "out.json"
        
        with mock.patch("src.ocr.ocr.ocr_image", side_effect=fake_ocr):
            count = ingest_screenshots(
                [str(img)],
                db_path=db_path,
                sport="nba",
                season="2025-26",
                source="screenshot",
                book="BookA",
                captured_at="2025-11-10",
                json_output=str(json_out),
            )
        
        assert count == 1
        result = json.loads(json_out.read_text())
        assert len(result) == 1
        
        bundle = result[0]
        assert bundle["team_home_raw"] == expected[0]["team_home_raw"]
        assert bundle["team_away_raw"] == expected[0]["team_away_raw"]
        assert len(bundle["markets"]) == len(expected[0]["markets"])


def test_ingest_json_mode_vs_db_mode_backward_compatibility():
    """Verify DB write path still works and matches JSON output."""
    text = """Los Angeles Lakers vs LA Clippers
Lakers +110
Lakers -3.5 -110
Over 215.5 -110
"""
    
    def fake_ocr(img):
        return text
    
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        repo.init_db(db_path)
        br.init_db(db_path)
        
        img = Path(td) / "img1.png"
        img.write_text("dummy")
        
        # Run in DB mode
        with mock.patch("src.ocr.ocr.ocr_image", side_effect=fake_ocr):
            db_count = ingest_screenshots(
                [str(img)],
                db_path=db_path,
                sport="nba",
                season="2025-26",
                source="screenshot",
                book="BookA",
                captured_at="2025-11-10",
            )
        
        # Verify staging rows exist
        import sqlite3
        conn = sqlite3.connect(db_path)
        try:
            staging_count = conn.execute("SELECT COUNT(*) FROM market_snapshot_staging").fetchone()[0]
            staging_rows = conn.execute(
                "SELECT market_type, odds, line FROM market_snapshot_staging ORDER BY id"
            ).fetchall()
            
            assert staging_count == 3  # ML, spread, total
            assert staging_count == db_count
            
            # Verify row types match expected bundle order
            types = [row[0] for row in staging_rows]
            assert "ML" in types
            assert "spread" in types
            assert "total" in types
        finally:
            conn.close()


def test_ingest_json_mode_skips_db_writes():
    """Verify JSON mode doesn't write to DB."""
    text = "Lakers vs Clippers\nLakers +110\n-3.5 -110\nOver 215.5 -110\n"
    
    def fake_ocr(img):
        return text
    
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        repo.init_db(db_path)
        br.init_db(db_path)
        
        img = Path(td) / "img1.png"
        img.write_text("dummy")
        json_out = Path(td) / "out.json"
        
        with mock.patch("src.ocr.ocr.ocr_image", side_effect=fake_ocr):
            count = ingest_screenshots(
                [str(img)],
                db_path=db_path,
                sport="nba",
                season="2025-26",
                source="screenshot",
                book="BookA",
                captured_at="2025-11-10",
                json_output=str(json_out),
            )
        
        # Verify JSON was written
        assert json_out.exists()
        assert count == 1
        
        # Verify no staging rows in DB
        import sqlite3
        conn = sqlite3.connect(db_path)
        try:
            staging_count = conn.execute("SELECT COUNT(*) FROM market_snapshot_staging").fetchone()[0]
            assert staging_count == 0
        finally:
            conn.close()
