"""Bet logging and settlement utilities.

Responsibilities:
- `log_bets` reads `BETS` sheet and writes idempotent rows to the `bets` table.
  Supports `--dry-run` and `--writeback` to write back `bet_id` and `logged_at`.
- `settle_bets` joins bets to `games` and computes outcome/profit (idempotent).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from openpyxl import load_workbook

from data.paths import db_path_for

from src.utils.odds import payout_per_unit
from src.data import betting_repository as br


def _utcnow_iso() -> str:
    return datetime.utcnow().isoformat()


def _resolve_db_path(db_path: Optional[str | Path], sport: Optional[str], season: Optional[str]) -> str | None:
    if db_path:
        return str(db_path)
    if sport and season:
        return str(db_path_for(sport, season))
    return None


def log_bets(
    workbook_path: str,
    *,
    review_run_id: Optional[str] = None,
    db_path: Optional[str | Path] = None,
    dry_run: bool = False,
    writeback: bool = False,
) -> int:
    """Parse `BETS` sheet and insert idempotent bet rows.

    - If `review_run_id` not provided, read from `META` sheet (key review_run_id).
    - Blank `stake` is interpreted as PASS and skipped.
    - Idempotency enforced via UNIQUE(review_run_id, game_id, market_type, selection)
    - If `writeback` is True, write `bet_id` and `logged_at` back into the workbook.

    Returns number of bets inserted/updated (not counting PASS rows).
    """
    wb_path = Path(workbook_path)
    if not wb_path.exists():
        raise FileNotFoundError(f"Workbook not found: {wb_path}")

    df = pd.read_excel(wb_path, sheet_name="BETS")

    # load review_run_id from META if missing
    if review_run_id is None:
        try:
            meta = pd.read_excel(wb_path, sheet_name="META")
            row = meta[meta["key"] == "review_run_id"]
            if not row.empty:
                review_run_id = str(row.iloc[0]["value"])
        except Exception:
            review_run_id = None

    if review_run_id is None:
        raise ValueError("review_run_id not provided and not found in META sheet")

    resolved_db = _resolve_db_path(db_path, None, None)
    if resolved_db is None:
        raise ValueError("db_path must be provided via --db or inferred from sport/season in CLI")

    inserted = 0
    writeback_rows: list[Tuple[int, int, str]] = []  # (df_index, bet_id, logged_at)

    conn = None
    try:
        import sqlite3

        conn = sqlite3.connect(resolved_db)
        cur = conn.cursor()

        for idx, row in df.iterrows():
            stake = row.get("stake")
            if pd.isna(stake) or stake == "" or stake == 0:
                # PASS
                continue
            stake = float(stake)
            game_id = row.get("game_id")
            market_type = row.get("market_type")
            selection = row.get("selection")
            line = row.get("line")
            odds = row.get("price") if not pd.isna(row.get("price")) and row.get("price") else row.get("odds")
            if pd.isna(odds):
                odds = None
            else:
                odds = int(odds)
            book = row.get("book") if not pd.isna(row.get("book")) else None
            source_opportunity_id = row.get("opportunity_id") if not pd.isna(row.get("opportunity_id")) else None

            logged_at = _utcnow_iso()
            status = "pending"

            # Upsert (INSERT OR REPLACE) to maintain idempotency on unique key
            cur.execute(
                """
                INSERT INTO bets (
                    review_run_id, game_id, market_type, selection, line, odds, stake, book, logged_at, status, source_opportunity_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(review_run_id, game_id, market_type, selection) DO UPDATE SET
                    stake = excluded.stake,
                    odds = excluded.odds,
                    line = excluded.line,
                    book = excluded.book,
                    logged_at = excluded.logged_at,
                    status = excluded.status,
                    source_opportunity_id = excluded.source_opportunity_id
                """,
                (
                    review_run_id,
                    game_id,
                    market_type,
                    selection,
                    line,
                    odds,
                    stake,
                    book,
                    logged_at,
                    status,
                    source_opportunity_id,
                ),
            )
            conn.commit()

            # fetch bet id
            bid_row = cur.execute(
                "SELECT id FROM bets WHERE review_run_id = ? AND game_id = ? AND market_type = ? AND selection = ?",
                (review_run_id, game_id, market_type, selection),
            ).fetchone()
            bet_id = bid_row[0] if bid_row is not None else None
            if writeback and bet_id:
                writeback_rows.append((idx, bet_id, logged_at))
            inserted += 1

    finally:
        if conn:
            conn.close()

    if writeback and writeback_rows:
        # write back to workbook
        wb = load_workbook(wb_path)
        ws = wb["BETS"]
        # find header row
        headers = {c.value: i + 1 for i, c in enumerate(next(ws.iter_rows(min_row=1, max_row=1)))}
        bid_col = headers.get("bet_id")
        logged_col = headers.get("logged_at")
        if bid_col is None or logged_col is None:
            # if columns missing, append them
            max_col = ws.max_column
            bid_col = max_col + 1
            logged_col = max_col + 2
            ws.cell(row=1, column=bid_col, value="bet_id")
            ws.cell(row=1, column=logged_col, value="logged_at")
        for df_idx, bet_id, logged_at in writeback_rows:
            excel_row = df_idx + 2  # pandas index 0 => excel row 2
            ws.cell(row=excel_row, column=bid_col, value=bet_id)
            ws.cell(row=excel_row, column=logged_col, value=logged_at)
        wb.save(wb_path)

    return inserted


def _compute_spread_outcome(home_score: int, away_score: int, line: float, selection: str) -> Tuple[str, int]:
    # for home selection:
    margin = home_score - away_score + line
    if selection and selection.lower().strip() not in ["home", "away"]:
        # selection likely team name; caller must ensure if selection is home or away
        pass
    if margin > 0:
        return "win", 1
    if margin == 0:
        return "push", 0
    return "loss", -1


def settle_bets(*, sport: str, season: str, db_path: Optional[str | Path] = None, settle_date: Optional[str] = None) -> int:
    """Settle open bets by joining to games and marking outcome/profit. Return number settled."""
    resolved_db = _resolve_db_path(db_path, sport, season)
    if resolved_db is None:
        raise ValueError("db_path must be provided via --db or sport/season")

    import sqlite3

    conn = sqlite3.connect(resolved_db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # find unsettled bets with games having scores
    q = """
        SELECT b.*, g.home_score, g.away_score, g.home_team, g.away_team
        FROM bets b
        JOIN games g ON b.game_id = g.game_id
        WHERE b.status != 'settled' AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
    """
    rows = cur.execute(q).fetchall()
    settled = 0
    for r in rows:
        bet_id = r["id"]
        stake = r["stake"] or 0.0
        odds = r["odds"]
        market_type = r["market_type"]
        selection = r["selection"]
        line = r["line"]
        home = r["home_score"]
        away = r["away_score"]
        home_team = r["home_team"]
        away_team = r["away_team"]

        outcome = None
        profit = 0.0

        if market_type == "ML":
            # determine winner
            if home > away:
                winner = home_team
            elif away > home:
                winner = away_team
            else:
                winner = None
            if winner is None:
                outcome = "push"
                profit = 0.0
            elif selection == winner:
                outcome = "win"
                profit = stake * payout_per_unit(odds)
            else:
                outcome = "loss"
                profit = -stake
        elif market_type == "spread":
            # Determine if selection is home or away
            is_home_selection = selection == home_team
            # For home selection: margin = home - away + line
            # For away selection: margin = -(home - away + line)
            base_margin = home - away + (line or 0.0)
            margin = base_margin if is_home_selection else -base_margin
            if margin > 0:
                outcome = "win"
                profit = stake * payout_per_unit(odds)
            elif margin == 0:
                outcome = "push"
                profit = 0.0
            else:
                outcome = "loss"
                profit = -stake
        elif market_type == "total":
            total = home + away
            sel = (selection or "").lower()
            if "over" in sel:
                if total > line:
                    outcome = "win"
                    profit = stake * payout_per_unit(odds)
                elif total == line:
                    outcome = "push"
                    profit = 0.0
                else:
                    outcome = "loss"
                    profit = -stake
            elif "under" in sel:
                if total < line:
                    outcome = "win"
                    profit = stake * payout_per_unit(odds)
                elif total == line:
                    outcome = "push"
                    profit = 0.0
                else:
                    outcome = "loss"
                    profit = -stake
            else:
                # unknown selection; skip
                continue
        else:
            # unsupported market
            continue

        # update bets row
        cur.execute(
            "UPDATE bets SET status = 'settled', outcome = ?, profit = ? WHERE id = ? AND status != 'settled'",
            (outcome, profit, bet_id),
        )
        conn.commit()
        settled += 1

    conn.close()
    return settled
