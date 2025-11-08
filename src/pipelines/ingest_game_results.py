import sys, csv
from pathlib import Path

# Ensure 'src' is on sys.path when running as a script
sys.path.append(str(Path(__file__).resolve().parents[1]))

from parsers.sr_table_parser import parse_sr_workbook

def main():
    if len(sys.argv) != 3:
        print("Usage: python src/pipelines/ingest_game_results.py <in(.xlsx|.xls|.csv)> <out.csv>")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    rows = parse_sr_workbook(str(in_path))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["date","visitor_team","visitor_pts","home_team","home_pts","ot","game_id"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {out_path} with {len(rows)} rows.")

if __name__ == "__main__":
    main()
