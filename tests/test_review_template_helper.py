import sqlite3
from pathlib import Path
import pandas as pd

from scripts.generate_review_template import generate
from src.utils.review_helpers import populate_game_ids


def test_populate_game_id_from_staging(tmp_path):
    # Setup temp DB
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # create minimal staging table
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS market_snapshot_staging (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_type TEXT,
            selection TEXT,
            line REAL,
            match_status TEXT,
            game_id TEXT
        );
        """
    )
    # Insert a matched staging row
    cur.execute(
        "INSERT INTO market_snapshot_staging (market_type, selection, line, match_status, game_id) VALUES (?, ?, ?, ?, ?)",
        ("ML", "Lakers", 0.0, "matched", "2025-10-01-lal-at-bos"),
    )
    conn.commit()
    conn.close()

    # Generate template workbook
    out = tmp_path / "review.xlsx"
    generate(str(out))

    # Add a BETS row lacking game_id but matching staging row
    df = pd.read_excel(out, sheet_name="BETS")
    new_row = {"market_type": "ML", "selection": "Lakers", "line": 0.0, "stake": 1.0}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    with pd.ExcelWriter(out, engine="openpyxl", mode="a") as writer:
        try:
            writer.book.remove(writer.book["BETS"])
        except Exception:
            pass
        df.to_excel(writer, sheet_name="BETS", index=False)

    updated = populate_game_ids(str(out), str(db_path))
    assert updated == 1

    df2 = pd.read_excel(out, sheet_name="BETS")
    assert df2.loc[0, "game_id"] == "2025-10-01-lal-at-bos"
