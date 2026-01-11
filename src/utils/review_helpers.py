from pathlib import Path
from typing import Optional
import sqlite3

import pandas as pd


def populate_game_ids(workbook_path: str, db_path: str) -> int:
    """Populate missing `game_id` in BETS sheet by looking up `market_snapshot_staging`.

    Matching heuristic: exact match on (`market_type`, `selection`, `line`) and
    `match_status = 'matched'`. If exactly one staging row matches and contains
    a `game_id`, that `game_id` will be written into the workbook. Rows with
    0 or multiple matches are left unchanged.

    Returns number of rows updated.
    """
    wb = Path(workbook_path)
    if not wb.exists():
        raise FileNotFoundError(f"Workbook not found: {wb}")

    df = pd.read_excel(wb, sheet_name="BETS")

    # Ensure target columns exist with object dtype so assigning string values
    # does not trigger pandas' incompatible-dtype FutureWarning in newer pandas.
    for _col in ("game_id", "log_status"):
        if _col not in df.columns:
            df[_col] = pd.NA
        df[_col] = df[_col].astype(object)

    conn = sqlite3.connect(Path(db_path))
    cur = conn.cursor()

    updated = 0
    for idx, row in df.iterrows():
        gid = row.get("game_id")
        if not (pd.isna(gid) or gid == "" or gid is None):
            continue
        market_type = row.get("market_type")
        selection = row.get("selection")
        line = row.get("line")
        if pd.isna(market_type) or pd.isna(selection) or pd.isna(line):
            continue

        q = """
        SELECT game_id FROM market_snapshot_staging
        WHERE market_type = ? AND selection = ? AND line = ? AND match_status = 'matched'
        """
        cur.execute(q, (str(market_type), str(selection), float(line)))
        rows = cur.fetchall()
        if len(rows) == 1 and rows[0][0]:
            df.at[idx, "game_id"] = rows[0][0]
            df.at[idx, "log_status"] = "filled_from_staging"
            updated += 1

    conn.close()

    if updated:
        # write back BETS sheet
        with pd.ExcelWriter(wb, engine="openpyxl", mode="a") as writer:
            try:
                writer.book.remove(writer.book["BETS"])
            except Exception:
                pass
            df.to_excel(writer, sheet_name="BETS", index=False)

    return updated


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", required=True)
    parser.add_argument("--db", required=True)
    args = parser.parse_args()
    n = populate_game_ids(args.workbook, args.db)
    print(f"Populated {n} game_id(s)")
