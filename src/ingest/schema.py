"""Pydantic schemas for normalized ingest data."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


class GameResult(BaseModel):
    """Normalized game row produced by ingestion parsers."""

    date: date
    start_time: Optional[datetime] = None
    home_team: str
    away_team: str
    home_score: Optional[int] = Field(default=None, ge=0)
    away_score: Optional[int] = Field(default=None, ge=0)
    neutral: bool = False
    overtime: bool = False
    decision_type: Optional[str] = None
    game_id: Optional[str] = None
    sport: Optional[str] = None
    season: Optional[str] = None
    division: Optional[str] = None
    conference: Optional[str] = None
    notes: Optional[str] = None
