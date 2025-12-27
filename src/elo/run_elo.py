from __future__ import annotations

from pathlib import Path

try:  # Allow execution from repository root or nested directories
    from bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - fallback when bootstrap isn't on sys.path
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
from typing import Any, Dict, List

import pandas as pd

from models.elo import Elo


def load_games(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize and sort by date if possible
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        if dt.notna().any():
            df = df.assign(_dt=dt).sort_values(["_dt", "game_id"]).drop(columns=["_dt"], errors="ignore")
    return df


def run_elo_on_games(df: pd.DataFrame) -> Elo:
    e = Elo()
    # Expect columns: date, visitor_team, visitor_pts, home_team, home_pts
    for _, r in df.iterrows():
        home = str(r.get("home_team", "")).strip()
        away = str(r.get("visitor_team", "")).strip()
        try:
            hp = int(r.get("home_pts"))
            ap = int(r.get("visitor_pts"))
        except Exception:
            continue
        if not home or not away:
            continue
        e.update(home, away, hp, ap, neutral=False)
    return e


def save_ratings(e: Elo, out_path: Path) -> None:
    items: List[Dict[str, Any]] = [
        {"team": team, "elo": rating, "games": e.N[team]}
        for team, rating in e.R.items()
    ]
    df = pd.DataFrame(items).sort_values("elo", ascending=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Elo updates over processed game CSV.")
    parser.add_argument("csv", help="Input CSV from data/processed with game results.")
    parser.add_argument(
        "-o",
        "--output",
        help="Optional output CSV for final Elo ratings. Defaults to data/processed/elo_<stem>.csv",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Print top N teams by Elo (default: 10)",
    )
    args = parser.parse_args()

    in_path = Path(args.csv)
    if not in_path.exists():
        # try under data/processed
        candidate = Path("data/processed") / args.csv
        if candidate.exists():
            in_path = candidate
        else:
            raise FileNotFoundError(f"Input not found: '{args.csv}' (also tried '{candidate}')")

    df = load_games(in_path)
    e = run_elo_on_games(df)

    # Output ratings
    if args.output:
        out_path = Path(args.output)
        if out_path.is_dir():
            out_path = out_path / f"elo_{in_path.stem}.csv"
    else:
        out_path = Path("data/processed") / f"elo_{in_path.stem}.csv"

    save_ratings(e, out_path)

    # Print summary
    ratings = sorted(((t, r) for t, r in e.R.items()), key=lambda x: x[1], reverse=True)
    top = ratings[: max(0, args.top)]
    print(f"Processed {len(df)} rows. Teams rated: {len(ratings)}")
    print("Top teams:")
    for team, rating in top:
        print(f"  {team}: {rating:.1f} (G={e.N[team]})")
    print(f"Saved ratings -> {out_path}")


if __name__ == "__main__":
    main()
