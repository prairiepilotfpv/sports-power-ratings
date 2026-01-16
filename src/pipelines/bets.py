"""Bet logging and settlement utilities.

Responsibilities:
- `log_bets` reads `BETS` sheet and writes idempotent rows to the `bets` table.
  Supports `--dry-run` and `--writeback` to write back `bet_id` and `logged_at`.
- `settle_bets` joins bets to `games` and computes outcome/profit (idempotent).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from openpyxl import load_workbook

from data.paths import db_path_for

from src.utils.odds import payout_per_unit
from src.data import betting_repository as br


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    # Ensure writeback columns exist with an object dtype so assigning
    # strings/timestamps does not trigger pandas' incompatible-dtype FutureWarning.
    for _col in ("bet_id", "logged_at", "log_status"):
        if _col not in df.columns:
            df[_col] = pd.NA
        df[_col] = df[_col].astype(object)

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

        # detect duplicates inside the workbook (same game_id, market_type, selection)
        seen = set()
        for idx, row in df.iterrows():
            stake = row.get("stake")
            if pd.isna(stake) or stake == "" or stake == 0:
                # PASS
                continue
            try:
                stake = float(stake)
            except Exception:
                continue

            game_id = row.get("game_id")
            market_type = row.get("market_type")
            selection = row.get("selection")
            key = (game_id, market_type, selection)
            if key in seen:
                # mark as hold in the workbook and skip logging (duplicate within same snapshot)
                df.at[idx, "log_status"] = "hold"
                if writeback:
                    # we'll write back the log_status later
                    pass
                continue
            seen.add(key)

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

            # attach latest CLV snapshot if present
            clv = br.get_latest_clv(resolved_db, game_id=game_id, market_type=market_type, selection=selection)
            clv_line = clv.get("close_line") if clv else None
            clv_odds = clv.get("close_odds") if clv else None

            # If workbook provided a line/odds and they differ from latest market snapshot,
            # persist them as a new market_snapshot with snapshot_run_id = review_run_id.
            try:
                provided_line = None if pd.isna(line) or line == "" else float(line)
            except Exception:
                provided_line = None
            try:
                provided_odds = None if odds is None or odds == "" or pd.isna(odds) else int(odds)
            except Exception:
                provided_odds = None

            try:
                latest_snap = br.get_latest_market_snapshot(resolved_db, game_id=game_id or "", market_type=market_type or "", selection=selection or "")
            except Exception:
                latest_snap = None

            if not dry_run and (provided_line is not None or provided_odds is not None):
                different = True
                if latest_snap is not None:
                    # Compare numeric values; if both equal, treat as same
                    try:
                        snap_line = latest_snap.get("line")
                        snap_odds = latest_snap.get("odds")
                        if (snap_line is None and provided_line is None) or (snap_line is not None and provided_line is not None and float(snap_line) == float(provided_line)):
                            if (snap_odds is None and provided_odds is None) or (snap_odds is not None and provided_odds is not None and int(snap_odds) == int(provided_odds)):
                                different = False
                    except Exception:
                        different = True
                if different:
                    try:
                        br.add_market_snapshot(
                            resolved_db,
                            snapshot_run_id=review_run_id,
                            captured_at=logged_at,
                            book="manual",
                            market_type=market_type,
                            selection=selection,
                            line=provided_line,
                            odds=provided_odds,
                            game_id=game_id,
                            source_staging_id=None,
                        )
                    except Exception:
                        # best-effort: do not fail logging if snapshot insert fails
                        pass

            # Upsert (INSERT OR REPLACE) to maintain idempotency on unique key
            cur.execute(
                """
                INSERT INTO bets (
                    review_run_id, game_id, market_type, selection, line, odds, stake, book, logged_at, status, source_opportunity_id, clv_close_odds, clv_close_line
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(review_run_id, game_id, market_type, selection) DO UPDATE SET
                    stake = excluded.stake,
                    odds = excluded.odds,
                    line = excluded.line,
                    book = excluded.book,
                    logged_at = excluded.logged_at,
                    status = excluded.status,
                    source_opportunity_id = excluded.source_opportunity_id,
                    clv_close_odds = excluded.clv_close_odds,
                    clv_close_line = excluded.clv_close_line
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
                    clv_odds,
                    clv_line,
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
                df.at[idx, "bet_id"] = bet_id
                df.at[idx, "logged_at"] = logged_at
                df.at[idx, "log_status"] = "logged"
            inserted += 1

    finally:
        if conn:
            conn.close()

    if writeback and not dry_run:
        # write modified dataframe back to the workbook (replace BETS sheet)
        import pandas as _pd
        with _pd.ExcelWriter(wb_path, engine="openpyxl", mode="a") as writer:
            try:
                writer.book.remove(writer.book["BETS"])
            except Exception:
                pass
            df.to_excel(writer, sheet_name="BETS", index=False)

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
