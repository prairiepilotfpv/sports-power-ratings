from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class GameResult(BaseModel):
    date: date
    visitor_team: str
    visitor_pts: Optional[int] = Field(default=None, ge=0)
    home_team: str
    home_pts: Optional[int] = Field(default=None, ge=0)
    ot: bool = False
    game_id: Optional[str] = None
    sport: Optional[str] = None
    season: Optional[str] = None
