from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class GameResult(BaseModel):
    date: date
    home_team: str
    away_team: str
    home_score: Optional[int] = Field(default=None, ge=0)
    away_score: Optional[int] = Field(default=None, ge=0)
    neutral: bool = False
    overtime: bool = False
    game_id: Optional[str] = None
    sport: Optional[str] = None
    season: Optional[str] = None
    notes: Optional[str] = None
