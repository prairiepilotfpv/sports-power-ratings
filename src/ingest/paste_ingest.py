"""Small ingest shim to parse pasted market text files.

Usage:
  from src.ingest.paste_ingest import ingest_file
  rows = ingest_file('data/raw/NBA-market.txt')

This module intentionally does not write to DB; it returns parsed rows ready
to be passed to `src.data.betting_repository.add_staging_row` or inspected.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional
import json

from src.parsers.paste_parser import parse_paste
import csv
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("outputs/paste_parsed")


def ingest_file_to_csv(path: str, out_csv: str | None = None, *, team_list: Optional[List[str]] = None) -> str:
    """Parse a pasted file and write results to CSV. Returns output path."""
    rows = ingest_file(path, team_list=team_list)
    out_dir = DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = out_dir / (Path(path).stem + ".csv")

    # canonical column order
    cols = [
        "source",
        "captured_at",
        "raw_text",
        "book",
        "team_home_raw",
        "team_away_raw",
        "game_date",
        "market_type",
        "selection",
        "line",
        "odds",
        "match_status",
        "match_confidence",
        "game_id",
        "created_at",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        seen = set()
        for r in rows:
            key = (
                r.get("market_type"),
                (r.get("selection") or "").strip().lower(),
                str(r.get("line")) if r.get("line") is not None else "",
                str(r.get("odds")) if r.get("odds") is not None else "",
                (r.get("team_home_raw") or "").strip().lower(),
                (r.get("team_away_raw") or "").strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            # ensure all keys present
            row = {c: r.get(c) for c in cols}
            writer.writerow(row)

    return str(out_path)


def ingest_file(path: str, *, team_list: Optional[List[str]] = None) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    text = p.read_text(encoding="utf-8", errors="ignore")
    rows = parse_paste(text, team_list=team_list)
    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse a pasted market text file and emit JSON")
    parser.add_argument("file", help="Path to pasted market text file")
    parser.add_argument("--json", help="Write parsed JSON to this path", default=None)
    args = parser.parse_args()

    out = ingest_file(args.file)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {len(out)} parsed rows to {args.json}")
    else:
        print(json.dumps(out[:50], indent=2, ensure_ascii=False))
