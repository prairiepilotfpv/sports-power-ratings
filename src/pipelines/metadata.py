from __future__ import annotations

import hashlib
from typing import Sequence

import pandas as pd


def canonical_csv_bytes(df: pd.DataFrame, columns: Sequence[str]) -> bytes:
    """Return canonical CSV bytes for hashing (stable column order)."""
    frame = df.loc[:, list(columns)].copy() if columns else df.copy()
    frame = frame.fillna("")
    if "date" in frame.columns:
        dates = pd.to_datetime(frame["date"], errors="coerce")
        formatted = dates.dt.strftime("%Y-%m-%d")
        frame["date"] = formatted.fillna(frame["date"].astype(str))
    csv_text = frame.to_csv(index=False, lineterminator="\n", float_format="%.12g")
    return csv_text.encode("utf-8")


def prediction_hash(df: pd.DataFrame, columns: Sequence[str]) -> str:
    """Compute SHA256 hash of canonical prediction CSV bytes."""
    return hashlib.sha256(canonical_csv_bytes(df, columns)).hexdigest()
