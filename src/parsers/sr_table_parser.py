import pandas as pd
from pathlib import Path

# Expected SR columns (typical):
# Date, Visitor/Neutral, PTS, Home/Neutral, PTS, OT, Attend., Notes, Box Score
# Sometimes 'Box Score' is a link column; if not present, we create a synthetic game_id.

def parse_sr_workbook(path: str) -> list[dict]:
    p = Path(path)
    suf = p.suffix.lower()
    if suf in {".xlsx", ".xls"}:
        engine = "openpyxl" if suf == ".xlsx" else "xlrd"
        try:
            df = pd.read_excel(p, engine=engine)
        except ImportError as e:
            raise RuntimeError(
                f"Missing Excel engine '{engine}' to read '{suf}' files. Please install it in requirements."
            ) from e
        except Exception as e:
            # Attempt without specifying engine as a fallback
            try:
                df = pd.read_excel(p)
            except Exception:
                raise
    else:
        df = pd.read_csv(p)

    # Normalize column names (strip spaces, lower)
    df.columns = [c.strip().lower() for c in df.columns]

    # Try common SR names
    col_map = {
        "date": "date",
        "visitor/neutral": "visitor",
        "visitor": "visitor",
        "pts": "pts",  # there are two "pts" columns; we’ll disambiguate
        "home/neutral": "home",
        "home": "home",
        "ot": "ot",
        "box score": "box_score",
        "boxscore": "box_score",
        "box": "box_score",
    }

    # Disambiguate the two PTS columns by position
    # Assume first PTS belongs to visitor, second to home
    pts_cols = [c for c in df.columns if c == "pts"]
    if len(pts_cols) >= 2:
        v_pts_col, h_pts_col = pts_cols[0], pts_cols[1]
    else:
        # fallback common labels
        v_pts_col = next((c for c in df.columns if "visitor pts" in c), None)
        h_pts_col = next((c for c in df.columns if "home pts" in c), None)

    # Find other key columns
    def find(name, *alts):
        for c in (name, *alts):
            if c in df.columns:
                return c
        return None

    date_col = find("date")
    vis_col  = find("visitor", "visitor/neutral")
    home_col = find("home", "home/neutral")
    ot_col   = find("ot")
    box_col  = find("box_score")

    rows = []
    for _, r in df.iterrows():
        # Skip blank rows
        if pd.isna(r.get(date_col)) or pd.isna(r.get(vis_col)) or pd.isna(r.get(home_col)):
            continue

        visitor_team = str(r[vis_col]).strip()
        home_team    = str(r[home_col]).strip()

        # Parse points robustly
        def as_int(val):
            try:
                return int(val)
            except Exception:
                try:
                    return int(float(str(val).strip()))
                except Exception:
                    return None

        visitor_pts = as_int(r.get(v_pts_col))
        home_pts    = as_int(r.get(h_pts_col))

        # OT string (e.g., 'OT', '2OT') or blank
        ot = str(r.get(ot_col)).strip() if ot_col and pd.notna(r.get(ot_col)) else ""

        # game_id from box score link (often missing in workbook export)
        game_id = ""
        if box_col and pd.notna(r.get(box_col)):
            game_id = str(r.get(box_col)).strip()
        if not game_id:
            # synthetic id: date|visitor|home
            game_id = f"{pd.to_datetime(r[date_col]).date()}|{visitor_team}|{home_team}"

        rows.append({
            "date": str(pd.to_datetime(r[date_col]).date()),
            "visitor_team": visitor_team,
            "visitor_pts": visitor_pts,
            "home_team": home_team,
            "home_pts": home_pts,
            "ot": ot,
            "game_id": game_id,
        })
    return rows
