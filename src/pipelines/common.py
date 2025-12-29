from __future__ import annotations

from typing import Any, Dict, Iterable, List

import pandas as pd


def normalize_games(rows: Iterable[Any]) -> pd.DataFrame:
    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        if hasattr(row, "model_dump"):
            normalized_rows.append(row.model_dump())
        else:
            normalized_rows.append(dict(row))
    df = pd.DataFrame(normalized_rows)
    if df.empty:
        return df
    for score_col in ("home_score", "away_score"):
        if score_col in df.columns:
            df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        if dt.notna().any():
            df = df.assign(_dt=dt).sort_values(["_dt", "game_id"]).drop(columns=["_dt"], errors="ignore")
    return df
