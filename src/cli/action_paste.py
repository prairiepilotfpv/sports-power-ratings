from __future__ import annotations

from pathlib import Path
import sys


def _ensure_src_on_path() -> None:
    src_dir = Path(__file__).resolve().parents[2] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main(argv: list[str] | None = None) -> None:
    """CLI: convert Action paste file into a CSV ready for manual ingestion.

    Usage:
      python -m src.cli.action_paste --in market.txt --out outputs/paste_parsed/out.csv
    """
    import argparse
    _ensure_src_on_path()
    from src.utils.action_paste_parser import parse_file

    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out", dest="outfile", default=None)
    p.add_argument("--include-opens-json", dest="opens_json", default=None)
    args = p.parse_args(argv)

    games = parse_file(args.infile)
    if not games:
        print("No games parsed from input.")
        return

    import os

    if args.outfile:
        candidate = Path(args.outfile)
        # If the user provided a path that clearly refers to a directory
        # (trailing slash or existing dir), treat it as a target directory
        # and write a file named <infile_stem>.csv inside it.
        if args.outfile.endswith(os.sep) or args.outfile.endswith("/") or (candidate.exists() and candidate.is_dir()):
            out_path = candidate / (Path(args.infile).stem + ".csv")
        else:
            out_path = candidate
    else:
        out_path = Path("outputs/paste_parsed") / (Path(args.infile).stem + ".csv")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    import csv

    # write CSV rows: one row per line in the Excel paste order
    fieldnames = [
        "source",
        "captured_at",
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

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for g in games:
            rows = [
                {
                    "source": "paste",
                    "captured_at": "",
                    "team_home_raw": g.home_team,
                    "team_away_raw": g.away_team,
                    "game_date": g.game_date or "",
                    "market_type": "ML",
                    "selection": g.away_team,
                    "line": 0,
                    "odds": g.ml_away_odds,
                    "match_status": "unmatched",
                    "match_confidence": 0.5,
                    "game_id": "",
                    "created_at": "",
                },
                {
                    "source": "paste",
                    "captured_at": "",
                    "team_home_raw": g.home_team,
                    "team_away_raw": g.away_team,
                    "game_date": g.game_date or "",
                    "market_type": "ML",
                    "selection": g.home_team,
                    "line": 0,
                    "odds": g.ml_home_odds,
                    "match_status": "unmatched",
                    "match_confidence": 0.5,
                    "game_id": "",
                    "created_at": "",
                },
                {
                    "source": "paste",
                    "captured_at": "",
                    "team_home_raw": g.home_team,
                    "team_away_raw": g.away_team,
                    "game_date": g.game_date or "",
                    "market_type": "spread",
                    "selection": g.away_team,
                    "line": g.spread_away_line,
                    "odds": g.spread_away_odds,
                    "match_status": "unmatched",
                    "match_confidence": 0.5,
                    "game_id": "",
                    "created_at": "",
                },
                {
                    "source": "paste",
                    "captured_at": "",
                    "team_home_raw": g.home_team,
                    "team_away_raw": g.away_team,
                    "game_date": g.game_date or "",
                    "market_type": "spread",
                    "selection": g.home_team,
                    "line": g.spread_home_line,
                    "odds": g.spread_home_odds,
                    "match_status": "unmatched",
                    "match_confidence": 0.5,
                    "game_id": "",
                    "created_at": "",
                },
                {
                    "source": "paste",
                    "captured_at": "",
                    "team_home_raw": g.home_team,
                    "team_away_raw": g.away_team,
                    "game_date": g.game_date or "",
                    "market_type": "total",
                    "selection": "Over",
                    "line": g.total_line,
                    "odds": g.over_odds,
                    "match_status": "unmatched",
                    "match_confidence": 0.5,
                    "game_id": "",
                    "created_at": "",
                },
                {
                    "source": "paste",
                    "captured_at": "",
                    "team_home_raw": g.home_team,
                    "team_away_raw": g.away_team,
                    "game_date": g.game_date or "",
                    "market_type": "total",
                    "selection": "Under",
                    "line": g.total_line,
                    "odds": g.under_odds,
                    "match_status": "unmatched",
                    "match_confidence": 0.5,
                    "game_id": "",
                    "created_at": "",
                },
            ]
            for r in rows:
                w.writerow(r)

    print(f"Wrote CSV to: {out_path}")
    if args.opens_json:
        from src.utils.action_paste_parser import _write_opens_json

        _write_opens_json(games, args.opens_json)


if __name__ == "__main__":
    main(sys.argv[1:])
