import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import sys
import os

from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI

# Ensure 'src' is on sys.path when running as a script
sys.path.append(str(Path(__file__).resolve().parents[1]))

from parsers.sr_table_parser import parse_sr_scores
from ocr.ocr import ocr_image


def ingest_html_to_csv(input_path: Path, output_path: Path) -> int:
    html = input_path.read_text(encoding="utf-8", errors="ignore")
    rows: List[Dict] = parse_sr_scores(html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["date", "visitor_team", "visitor_pts", "home_team", "home_pts", "ot", "game_id"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(output_path, index=False)
    return len(df)


class GameRow(BaseModel):
    date: str
    visitor_team: str
    visitor_pts: int
    home_team: str
    home_pts: int
    ot: bool
    game_id: str | None = None


class GameRows(BaseModel):
    games: List[GameRow]


def _extract_games_structured(text: str) -> List[Dict[str, Any]]:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set. Add it to your environment or .env.")

    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    instruction = (
        "Extract basketball game results from the text. Return an array 'games' of objects "
        "with fields: date (string), visitor_team (string), visitor_pts (int), home_team (string), "
        "home_pts (int), ot (boolean), game_id (nullable string). If OT/2OT/3OT indicated, set ot=true."
    )

    try:
        resp = client.responses.parse(
            model=model,
            input=text,
            instruction=instruction,
            temperature=0,
            response_format=GameRows,
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI structured extraction failed: {e}")

    parsed: GameRows = resp.output_parsed  # type: ignore[attr-defined]
    rows: List[Dict[str, Any]] = [g.model_dump() for g in parsed.games]
    return rows


def ingest_image_to_csv(input_path: Path, output_path: Path) -> int:
    text = ocr_image(str(input_path))
    rows = _extract_games_structured(text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["date", "visitor_team", "visitor_pts", "home_team", "home_pts", "ot", "game_id"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(output_path, index=False)
    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse Sports-Reference 'Schedule & Results' from HTML or OCR screenshot and write CSV."
    )
    parser.add_argument(
        "input",
        help="Input HTML or image file. If not an existing path, resolved under data/raw/",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "html", "image"],
        default="auto",
        help="Input mode detection. Default auto by file extension.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output CSV path. Defaults to data/processed/<input_stem>.csv",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output file.",
    )

    args = parser.parse_args()

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")

    in_path = Path(args.input)
    if not in_path.exists():
        candidate = raw_dir / args.input
        if candidate.exists():
            in_path = candidate
        else:
            raise FileNotFoundError(f"Input not found: '{args.input}' (also tried '{candidate}')")

    if args.output:
        out_path = Path(args.output)
        if out_path.is_dir():
            out_path = out_path / f"{in_path.stem}.csv"
    else:
        out_path = processed_dir / f"{in_path.stem}.csv"

    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: '{out_path}'. Use --overwrite to replace.")

    # Determine mode
    mode = args.mode
    if mode == "auto":
        ext = in_path.suffix.lower()
        if ext in {".html", ".htm"}:
            mode = "html"
        elif ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
            mode = "image"
        else:
            # Peek at file to guess
            try:
                head = in_path.read_text(encoding="utf-8", errors="ignore")[0:2000]
                mode = "html" if ("<html" in head.lower() or "<table" in head.lower()) else "image"
            except Exception:
                mode = "image"

    if mode == "html":
        count = ingest_html_to_csv(in_path, out_path)
    else:
        count = ingest_image_to_csv(in_path, out_path)
    print(f"Wrote {count} rows -> {out_path}")


if __name__ == "__main__":
    main()
