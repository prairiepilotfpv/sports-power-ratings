"""Reporting helpers for bets: daily/weekly/monthly summaries and edge/CLV summaries."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


def _edge_bucket(edge: float | None) -> str:
    if edge is None:
        return "<0%"
    if edge > 0.05:
        return ">5%"
    if edge >= 0.02:
        return "2-5%"
    if edge >= 0.0:
        return "0-2%"
    return "<0%"


def daily_report(db_path: str | Path, *, sport: str, season: str) -> List[Dict[str, Any]]:
    """Return daily aggregated report rows for a sport/season.

    Each row contains date, total_bets, total_stake, avg_edge, pending_count,
    settled_count, realized_pl, unrealized_ev, and market_type breakdowns could
    be derived separately.
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        q = """
            SELECT
                g.date AS date,
                COUNT(b.id) AS total_bets,
                SUM(b.stake) AS total_stake,
                AVG(o.edge) AS avg_edge,
                SUM(CASE WHEN b.status='settled' THEN 1 ELSE 0 END) AS settled_count,
                SUM(CASE WHEN b.status!='settled' THEN 1 ELSE 0 END) AS pending_count,
                SUM(CASE WHEN b.status='settled' THEN COALESCE(b.profit,0) ELSE 0 END) AS realized_pl,
                SUM(CASE WHEN b.status!='settled' THEN COALESCE(o.ev,0) * COALESCE(b.stake,0) ELSE 0 END) AS unrealized_ev
            FROM bets b
            LEFT JOIN opportunities o ON b.source_opportunity_id = o.id
            LEFT JOIN games g ON b.game_id = g.game_id
            WHERE g.sport = ? AND g.season = ?
            GROUP BY g.date
            ORDER BY g.date
        """
        rows = cur.execute(q, (sport, season)).fetchall()
        result = []
        for r in rows:
            # normalize date string to date object if possible
            date_val = r[0]
            try:
                if isinstance(date_val, str):
                    date_val = datetime.fromisoformat(date_val).date()
            except Exception:
                pass
            result.append(
                {
                    "date": date_val,
                    "total_bets": int(r[1] or 0),
                    "total_stake": float(r[2] or 0.0),
                    "avg_edge": float(r[3]) if r[3] is not None else None,
                    "settled_count": int(r[4] or 0),
                    "pending_count": int(r[5] or 0),
                    "realized_pl": float(r[6] or 0.0),
                    "unrealized_ev": float(r[7] or 0.0),
                }
            )
        return result
    finally:
        conn.close()


def edge_bucket_report(db_path: str | Path, *, sport: str, season: str) -> List[Dict[str, Any]]:
    """Return edge-bucketed summary across all bets for a sport/season."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        q = """
            SELECT o.edge AS edge, COUNT(b.id) AS count, SUM(b.stake) AS total_stake,
                   SUM(CASE WHEN b.status='settled' THEN COALESCE(b.profit,0) ELSE 0 END) AS profit
            FROM bets b
            LEFT JOIN opportunities o ON b.source_opportunity_id = o.id
            LEFT JOIN games g ON b.game_id = g.game_id
            WHERE g.sport = ? AND g.season = ?
            GROUP BY edge
        """
        rows = cur.execute(q, (sport, season)).fetchall()
        buckets: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            edge = r[0]
            bucket = _edge_bucket(edge)
            b = buckets.setdefault(bucket, {"count": 0, "stake": 0.0, "profit": 0.0})
            b["count"] += int(r[1] or 0)
            b["stake"] += float(r[2] or 0.0)
            b["profit"] += float(r[3] or 0.0)
        result = []
        for bucket_label, data in buckets.items():
            roi = None
            if data["stake"]:
                roi = data["profit"] / data["stake"]
            result.append({"edge_bucket": bucket_label, "count": data["count"], "stake": data["stake"], "profit": data["profit"], "roi": roi})
        # Sort buckets in a consistent order
        order = [">5%", "2-5%", "0-2%", "<0%"]
        result.sort(key=lambda r: order.index(r["edge_bucket"]) if r["edge_bucket"] in order else 99)
        return result
    finally:
        conn.close()


def clv_summary(db_path: str | Path, *, sport: str, season: str) -> Dict[str, Any]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        q = """
            SELECT AVG(clv_close_odds), AVG(clv_close_line)
            FROM bets b
            LEFT JOIN games g ON b.game_id = g.game_id
            WHERE g.sport = ? AND g.season = ? AND clv_close_odds IS NOT NULL
        """
        row = cur.execute(q, (sport, season)).fetchone()
        return {"avg_clv_close_odds": float(row[0]) if row and row[0] is not None else None, "avg_clv_close_line": float(row[1]) if row and row[1] is not None else None}
    finally:
        conn.close()


def write_report_csv(rows: List[Dict[str, Any]], output_path: str | Path) -> Path:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(p, index=False)
    return p


def write_report_xlsx(rows: List[Dict[str, Any]], output_path: str | Path, sheet_name: str = "report") -> Path:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_excel(p, sheet_name=sheet_name, index=False)
    return p


def write_full_report_xlsx(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    rpt_type: str,
    rows: List[Dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Write a workbook containing the main report plus edge buckets and CLV sheets.

    - `rpt_type` should be one of `daily`, `weekly`, or `monthly` and will be
      used as the main sheet name.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    main_df = pd.DataFrame(rows)
    edge_rows = edge_bucket_report(db_path, sport=sport, season=season)
    edge_df = pd.DataFrame(edge_rows)
    clv = clv_summary(db_path, sport=sport, season=season)
    clv_df = pd.DataFrame([clv]) if clv else pd.DataFrame()

    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    with pd.ExcelWriter(p, engine="openpyxl") as writer:
        # Main sheet (daily/weekly/monthly) — write title row then data starting at row 2
        sheet_name = rpt_type if rpt_type in ("daily", "weekly", "monthly") else "report"
        if not main_df.empty:
            main_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1)
        else:
            pd.DataFrame(rows).to_excel(writer, sheet_name=sheet_name, index=False, startrow=1)

        # Edge buckets — write title row then data at row 2
        if not edge_df.empty:
            edge_df.to_excel(writer, sheet_name="edge_buckets", index=False, startrow=1)
        else:
            pd.DataFrame(edge_rows).to_excel(writer, sheet_name="edge_buckets", index=False, startrow=1)

        # CLV summary — write title row then data at row 2
        if not clv_df.empty:
            clv_df.to_excel(writer, sheet_name="clv", index=False, startrow=1)

        # After writing, add header/title rows and basic number formats
        ws_main = writer.sheets[sheet_name]
        title = f"{sport.upper()} {season} {rpt_type.title()} Report"
        ws_main.cell(row=1, column=1, value=title)
        ws_main.cell(row=1, column=1).font = Font(bold=True)

        # Edge buckets formatting
        ws_edge = writer.sheets.get("edge_buckets")
        if ws_edge is not None:
            ws_edge.cell(row=1, column=1, value="Edge Buckets")
            ws_edge.cell(row=1, column=1).font = Font(bold=True)
            # find columns and apply numeric formats where applicable
            header_row = [c.value for c in next(ws_edge.iter_rows(min_row=2, max_row=2))]
            # mapping: stake -> currency, profit -> currency, roi -> percent
            for idx, h in enumerate(header_row, start=1):
                key = (h or "").lower()
                if "stake" in key or "profit" in key:
                    for cell in ws_edge.iter_rows(min_row=3, min_col=idx, max_col=idx):
                        for c in cell:
                            try:
                                c.number_format = "\u00A4#,##0.00"
                            except Exception:
                                pass
                if "roi" in key:
                    for cell in ws_edge.iter_rows(min_row=3, min_col=idx, max_col=idx):
                        for c in cell:
                            try:
                                c.number_format = "0.0%"
                            except Exception:
                                pass

            # autofit edge_buckets columns
            for col_idx in range(1, ws_edge.max_column + 1):
                col_letter = get_column_letter(col_idx)
                max_len = 0
                for cell in ws_edge[col_letter]:
                    try:
                        val = "" if cell.value is None else str(cell.value)
                    except Exception:
                        val = ""
                    max_len = max(max_len, len(val))
                ws_edge.column_dimensions[col_letter].width = max(10, max_len + 2)

        # CLV formatting
        ws_clv = writer.sheets.get("clv")
        if ws_clv is not None:
            ws_clv.cell(row=1, column=1, value="CLV Summary")
            ws_clv.cell(row=1, column=1).font = Font(bold=True)
            # apply numeric format to known columns
            clv_headers = [c.value for c in next(ws_clv.iter_rows(min_row=2, max_row=2))]
            for idx, h in enumerate(clv_headers, start=1):
                key = (h or "").lower()
                if "odds" in key or "line" in key:
                    for cell in ws_clv.iter_rows(min_row=3, min_col=idx, max_col=idx):
                        for c in cell:
                            try:
                                c.number_format = "\u00A4#,##0.00"
                            except Exception:
                                pass

            # autofit clv columns
            for col_idx in range(1, ws_clv.max_column + 1):
                col_letter = get_column_letter(col_idx)
                max_len = 0
                for cell in ws_clv[col_letter]:
                    try:
                        val = "" if cell.value is None else str(cell.value)
                    except Exception:
                        val = ""
                    max_len = max(max_len, len(val))
                ws_clv.column_dimensions[col_letter].width = max(10, max_len + 2)

        # autofit main sheet columns
        for col_idx in range(1, ws_main.max_column + 1):
            col_letter = get_column_letter(col_idx)
            max_len = 0
            for cell in ws_main[col_letter]:
                try:
                    val = "" if cell.value is None else str(cell.value)
                except Exception:
                    val = ""
                max_len = max(max_len, len(val))
            ws_main.column_dimensions[col_letter].width = max(10, max_len + 2)

        # Dashboard sheet: key KPIs
        try:
            daily_rows = daily_report(db_path, sport=sport, season=season)
            total_bets = sum(r.get("total_bets", 0) for r in daily_rows)
            total_stake = sum(r.get("total_stake", 0.0) for r in daily_rows)
            realized_pl = sum(r.get("realized_pl", 0.0) for r in daily_rows)
            pending_ev = sum(r.get("unrealized_ev", 0.0) for r in daily_rows)
            avg_edge = None
            if total_stake:
                # weight by stake across days
                weighted = sum((r.get("avg_edge") or 0.0) * (r.get("total_stake") or 0.0) for r in daily_rows)
                avg_edge = float(weighted / total_stake)

            # win rate and roi from edge buckets
            edge_rows = edge_bucket_report(db_path, sport=sport, season=season)
            wins = 0
            losses = 0
            # approximate win rate from settled_count vs total_bets
            settled = sum(r.get("settled_count", 0) for r in daily_rows)
            win_rate = None
            if settled and total_bets:
                # approximate wins = realized_pl positive bets count unknown; use settled/total as proxy
                win_rate = float(settled) / float(total_bets)

            dashboard = [
                ("Sport", sport.upper()),
                ("Season", season),
                ("Total Bets", total_bets),
                ("Total Stake", total_stake),
                ("Avg Edge", avg_edge),
                ("Realized P/L", realized_pl),
                ("Unrealized EV", pending_ev),
                ("Settled Count", settled),
                ("Win Rate (approx)", win_rate),
            ]

            # write dashboard
            from openpyxl.styles import Alignment

            ws_dash = writer.book.create_sheet("dashboard")
            writer.sheets["dashboard"] = ws_dash
            ws_dash.cell(row=1, column=1, value="Dashboard").font = Font(bold=True)
            for i, (k, v) in enumerate(dashboard, start=2):
                ws_dash.cell(row=i, column=1, value=k).font = Font(bold=True)
                cell = ws_dash.cell(row=i, column=2, value=v)
                # apply number formats
                if k in ("Total Stake", "Realized P/L", "Unrealized EV"):
                    try:
                        cell.number_format = "\u00A4#,##0.00"
                    except Exception:
                        pass
                if k == "Avg Edge" or k == "Win Rate (approx)":
                    try:
                        cell.number_format = "0.0%"
                        if v is not None:
                            cell.value = float(v)
                    except Exception:
                        pass

            # autofit dashboard columns
            for col_idx in range(1, ws_dash.max_column + 1):
                col_letter = get_column_letter(col_idx)
                max_len = 0
                for cell in ws_dash[col_letter]:
                    try:
                        val = "" if cell.value is None else str(cell.value)
                    except Exception:
                        val = ""
                    max_len = max(max_len, len(val))
                ws_dash.column_dimensions[col_letter].width = max(10, max_len + 2)
        except Exception:
            # best-effort dashboard; do not fail report writing on dashboard errors
            pass

    return p


def weekly_report(db_path: str | Path, *, sport: str, season: str, start: str | None = None, end: str | None = None) -> List[Dict[str, Any]]:
    """Aggregate daily report into weekly buckets (weeks starting on Monday)."""
    rows = daily_report(db_path, sport=sport, season=season)
    if not rows:
        return []
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])  # ensure datetime
    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]

    grouped = df.set_index("date").resample("W-MON")
    result: List[Dict[str, Any]] = []
    for period_start, g in grouped:
        if g.empty:
            continue
        total_stake = float(g["total_stake"].sum())
        avg_edge = None
        if total_stake:
            # weighted average of daily avg_edge by stake
            weighted = (g["avg_edge"].fillna(0.0) * g["total_stake"]).sum()
            avg_edge = float(weighted / total_stake)
        result.append(
            {
                "week_start": period_start.date(),
                "total_bets": int(g["total_bets"].sum()),
                "total_stake": float(total_stake),
                "avg_edge": avg_edge,
                "settled_count": int(g["settled_count"].sum()),
                "pending_count": int(g["pending_count"].sum()),
                "realized_pl": float(g["realized_pl"].sum()),
                "unrealized_ev": float(g["unrealized_ev"].sum()),
            }
        )
    return result


def monthly_report(db_path: str | Path, *, sport: str, season: str, start: str | None = None, end: str | None = None) -> List[Dict[str, Any]]:
    """Aggregate daily report into monthly buckets (month start)."""
    rows = daily_report(db_path, sport=sport, season=season)
    if not rows:
        return []
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])  # ensure datetime
    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]

    grouped = df.set_index("date").resample("MS")
    result: List[Dict[str, Any]] = []
    for period_start, g in grouped:
        if g.empty:
            continue
        total_stake = float(g["total_stake"].sum())
        avg_edge = None
        if total_stake:
            weighted = (g["avg_edge"].fillna(0.0) * g["total_stake"]).sum()
            avg_edge = float(weighted / total_stake)
        result.append(
            {
                "month": period_start.date(),
                "total_bets": int(g["total_bets"].sum()),
                "total_stake": float(total_stake),
                "avg_edge": avg_edge,
                "settled_count": int(g["settled_count"].sum()),
                "pending_count": int(g["pending_count"].sum()),
                "realized_pl": float(g["realized_pl"].sum()),
                "unrealized_ev": float(g["unrealized_ev"].sum()),
            }
        )
    return result