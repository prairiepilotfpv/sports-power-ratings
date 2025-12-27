from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

try:  # Allow execution from repository root or nested directories
    from bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - fallback when bootstrap isn't on sys.path
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from bootstrap import ensure_src_on_path

ensure_src_on_path()

import pandas as pd

from data.repository import load_games
from elo.run_elo import run_elo_on_games


def _normalize_games(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        if dt.notna().any():
            df = df.assign(_dt=dt).sort_values(["_dt", "game_id"]).drop(columns=["_dt"], errors="ignore")
    return df


def build_rankings(df: pd.DataFrame, model: str = "elo") -> pd.DataFrame:
    if model != "elo":
        raise ValueError(f"Unsupported model: {model}")
    elo = run_elo_on_games(df)
    items = [
        {"team": team, "rating": rating, "games": elo.N[team]}
        for team, rating in elo.R.items()
    ]
    return pd.DataFrame(items).sort_values("rating", ascending=False)


def run_rankings(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str = "elo",
    output_path: str | Path | None = None,
) -> Path:
    rows = load_games(db_path, sport=sport, season=season)
    df = _normalize_games(rows)
    if df.empty:
        raise ValueError(f"No games found for sport={sport!r}, season={season!r}")
    rankings = build_rankings(df, model=model)
    if output_path is None:
        output_path = Path("data/processed") / sport / season / "rankings.csv"
    else:
        output_path = Path(output_path)
        if output_path.is_dir() or output_path.suffix == "":
            output_path = output_path / "rankings.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(output_path, index=False)
    return output_path
