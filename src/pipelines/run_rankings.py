from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

try:  # Allow execution from repository root or nested directories
    from bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - fallback when bootstrap isn't on sys.path
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from bootstrap import ensure_src_on_path

ensure_src_on_path()

import pandas as pd

from data.repository import load_games
from models.registry import get_model


def _normalize_games(rows: Iterable[Any]) -> pd.DataFrame:
    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        if hasattr(row, "model_dump"):
            normalized_rows.append(row.model_dump())
        else:
            normalized_rows.append(dict(row))
    df = pd.DataFrame(normalized_rows)
    if df.empty:
        return df
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        if dt.notna().any():
            df = df.assign(_dt=dt).sort_values(["_dt", "game_id"]).drop(columns=["_dt"], errors="ignore")
    return df


def build_rankings(df: pd.DataFrame, model: str = "bradley-terry") -> pd.DataFrame:
    model_cls = get_model(model)
    model_instance = model_cls()
    model_instance.fit(df.to_dict(orient="records"))

    games_played: Dict[str, int] = {}
    for _, row in df.iterrows():
        home = str(row.get("home_team", "")).strip()
        away = str(row.get("away_team", "")).strip()
        if not home or not away:
            continue
        games_played[home] = games_played.get(home, 0) + 1
        games_played[away] = games_played.get(away, 0) + 1

    items = [
        {"team": team, "rating": rating, "games": games_played.get(team, 0)}
        for team, rating in model_instance.rankings()
    ]
    return pd.DataFrame(items).sort_values("rating", ascending=False)


def run_rankings(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str = "bradley-terry",
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
