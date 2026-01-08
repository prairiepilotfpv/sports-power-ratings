"""Review run orchestration (skeleton).

Responsibilities:
- Create a ReviewRun record
- Generate forecast_snapshots from model projections
- Generate opportunities by comparing model probabilities to market odds
- Emit an Excel review workbook with EV and BETS sheets
"""

from __future__ import annotations


import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from openpyxl import load_workbook

from data.paths import processed_path_for
from data.repository import load_games
from data.betting_repository import get_opportunities_with_game_info, create_review_run


def _resolve_output_path(sport: str, season: str, review_run_id: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"review_{review_run_id}_{ts}.xlsx"
    return processed_path_for(sport, season, f"review/{filename}")


def build_review_workbook(db_path: str | Path, *, review_run_id: str, sport: str, season: str, output_path: str | Path | None = None) -> Path:
    """Build a review workbook for a review_run_id.

    - EV sheet is read-only/protected
    - BETS sheet is editable and contains base columns + stake/book/price/bet_id/logged_at/notes
    - META sheet contains key/value pairs (review_run_id, sport, season, created_at)
    """
    out = Path(output_path) if output_path else _resolve_output_path(sport, season, review_run_id)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = get_opportunities_with_game_info(db_path, review_run_id=review_run_id)

    # Build EV dataframe
    ev_cols = [
        "review_run_id",
        "game_id",
        "date",
        "away_team",
        "home_team",
        "market_type",
        "selection",
        "line",
        "odds",
        "implied_prob",
        "model_prob",
        "edge",
        "ev",
        "book",
        "source_snapshot_id",
        "opportunity_id",
    ]
    ev_rows = []
    for r in rows:
        ev_rows.append(
            {
                "review_run_id": r["review_run_id"],
                "game_id": r["game_id"],
                "date": r["date"],
                "away_team": r["away_team"],
                "home_team": r["home_team"],
                "market_type": r["market_type"],
                "selection": r["selection"],
                "line": r["line"],
                "odds": r["odds"],
                "implied_prob": r["implied_prob"],
                "model_prob": r["model_prob"],
                "edge": r["edge"],
                "ev": r["ev"],
                "book": None,
                "source_snapshot_id": r["source_market_snapshot_id"],
                "opportunity_id": r["opportunity_id"],
            }
        )

    ev_df = pd.DataFrame(ev_rows, columns=ev_cols)

    # BETS sheet: copy EV base columns and add editable columns
    bets_cols = ev_cols + ["stake", "book", "price", "bet_id", "logged_at", "notes"]
    bets_df = ev_df.copy(deep=True)
    for c in ["stake", "book", "price", "bet_id", "logged_at", "notes"]:
        bets_df[c] = ""

    # META sheet
    meta_rows = [
        {"key": "review_run_id", "value": review_run_id},
        {"key": "sport", "value": sport},
        {"key": "season", "value": season},
        {"key": "created_at", "value": datetime.now(timezone.utc).isoformat()},
    ]
    meta_df = pd.DataFrame(meta_rows)

    # Write via pandas then post-process with openpyxl
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        ev_df.to_excel(writer, sheet_name="EV", index=False)
        bets_df.to_excel(writer, sheet_name="BETS", index=False)
        meta_df.to_excel(writer, sheet_name="META", index=False)

    # Post-process protection and hidden META
    wb = load_workbook(out)
    if "EV" in wb.sheetnames:
        ws = wb["EV"]
        ws.protection.sheet = True
        ws.protection.enable = True
    if "BETS" in wb.sheetnames:
        ws = wb["BETS"]
        ws.protection.sheet = False
    if "META" in wb.sheetnames:
        ws = wb["META"]
        ws.sheet_state = "hidden"

    wb.save(out)
    return out


def create_and_build_review(db_path: str | Path, *, sport: str, season: str, model: str, notes: str | None = None) -> Path:
    """Create a review_run and immediately build the workbook. Returns path."""
    rid = create_review_run(db_path, sport=sport, season=season, model=model, notes=notes)
    return build_review_workbook(db_path, review_run_id=rid, sport=sport, season=season)
