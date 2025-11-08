# Parse Sports-Reference 'Schedule & Results' table into rows
# (date, visitor_team, visitor_pts, home_team, home_pts, ot, game_id)
from typing import List, Dict, Any, Optional
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import Comment
import re


def _extract_text(cell) -> str:
    if cell is None:
        return ""
    return cell.get_text(strip=True)


def _to_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None


def _find_candidate_tables(soup: BeautifulSoup):
    tables = list(soup.find_all("table"))
    # Sports-Reference sometimes wraps tables in HTML comments
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in c:
            try:
                cs = BeautifulSoup(c, "html.parser")
                tables.extend(cs.find_all("table"))
            except Exception:
                pass
    return tables


def parse_sr_scores(html: str) -> List[Dict]:
    """Parse a Sports-Reference 'Schedule & Results' table HTML.

    Returns list of dicts with keys:
      - date (str)
      - visitor_team (str)
      - visitor_pts (int)
      - home_team (str)
      - home_pts (int)
      - ot (bool)
      - game_id (str | None)
    """
    soup = BeautifulSoup(html, "html.parser")

    required_stats = {
        "date_game",
        "visitor_team_name",
        "visitor_pts",
        "home_team_name",
        "home_pts",
    }
    optional_stats = {"overtimes", "box_score_text"}

    rows_out: List[Dict[str, Any]] = []

    for table in _find_candidate_tables(soup):
        # Validate table by checking available data-stat columns
        header_cells = []
        thead = table.find("thead")
        if thead:
            tr = thead.find("tr")
            if tr:
                header_cells = tr.find_all(["th", "td"])
        if not header_cells:
            # Fallback: first row in table
            first_tr = table.find("tr")
            if first_tr:
                header_cells = first_tr.find_all(["th", "td"])

        header_stats = {c.get("data-stat") for c in header_cells if c.has_attr("data-stat")}
        if not required_stats.issubset(header_stats):
            # Try looser validation: check if body rows have the required data-stat
            body = table.find("tbody") or table
            sample_tr = body.find("tr") if body else None
            if not sample_tr:
                continue
            sample_stats = {c.get("data-stat") for c in sample_tr.find_all(["th", "td"]) if c.has_attr("data-stat")}
            if not required_stats.issubset(sample_stats):
                continue

        tbody = table.find("tbody") or table
        for tr in tbody.find_all("tr"):
            cls = tr.get("class", [])
            if any(c in ("thead", "over_header", "spacer") for c in cls):
                continue

            cells = {c.get("data-stat"): c for c in tr.find_all(["th", "td"]) if c.has_attr("data-stat")}
            if not required_stats.issubset(cells.keys()):
                continue

            date = _extract_text(cells.get("date_game"))
            vteam = _extract_text(cells.get("visitor_team_name"))
            hteam = _extract_text(cells.get("home_team_name"))
            vpts = _to_int(_extract_text(cells.get("visitor_pts")))
            hpts = _to_int(_extract_text(cells.get("home_pts")))

            # Skip games without scores yet
            if vpts is None or hpts is None:
                continue

            ot_text = _extract_text(cells.get("overtimes")) if "overtimes" in cells else ""
            ot = bool(ot_text and ("OT" in ot_text.upper()))

            # Try to get game_id from box score link
            game_id = None
            link_cell = cells.get("box_score_text")
            link = None
            if link_cell:
                link = link_cell.find("a", href=True)
            if not link:
                # Fallback: search any link in row containing /boxscores/
                link = tr.find("a", href=lambda h: isinstance(h, str) and "/boxscores/" in h)
            if link and link.has_attr("href"):
                m = re.search(r"/boxscores/([A-Za-z0-9_-]+)\.html", link["href"]) 
                if m:
                    game_id = m.group(1)

            rows_out.append(
                {
                    "date": date,
                    "visitor_team": vteam,
                    "visitor_pts": vpts,
                    "home_team": hteam,
                    "home_pts": hpts,
                    "ot": ot,
                    "game_id": game_id,
                }
            )

        # If we successfully parsed rows from this table, we can stop
        if rows_out:
            break

    return rows_out


def parse_sr_workbook(path: str | Path) -> List[Dict[str, Any]]:
    """Parse a Sports-Reference Excel export into structured game rows."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Workbook not found: '{p}'")

    # Sports-Reference exports are simple workbooks with a single sheet.
    # Allow pandas to pick the appropriate engine based on file type.
    engine: Optional[str]
    if p.suffix.lower() in {".xlsx", ".xlsm"}:
        engine = "openpyxl"
    else:
        engine = None

    try:
        df = pd.read_excel(p, engine=engine)
    except FileNotFoundError:
        # Propagate a consistent error message for CLI handling.
        raise FileNotFoundError(f"Workbook not found: '{p}'")

    # Normalise column names. Sports-Reference exports duplicate "PTS" columns
    # for the visitor and home scores. Pandas de-duplicates those as "PTS" and
    # "PTS.1" which we map explicitly below.
    column_map: Dict[str, str] = {}

    for col in df.columns:
        normalised = str(col).strip().lower()
        if normalised in {"date", "date_game"}:
            column_map[str(col)] = "date"
        elif normalised in {"visitor/neutral", "visitor", "visitor team"}:
            column_map[str(col)] = "visitor_team"
        elif normalised in {"home/neutral", "home", "home team"}:
            column_map[str(col)] = "home_team"
        elif normalised.startswith("pts"):
            if "visitor_pts" not in column_map.values():
                column_map[str(col)] = "visitor_pts"
            else:
                column_map[str(col)] = "home_pts"
        elif normalised in {"ot", "overtimes"}:
            column_map[str(col)] = "ot"

    required_cols = {"date", "visitor_team", "visitor_pts", "home_team", "home_pts"}
    if not required_cols.issubset(column_map.values()):
        missing = required_cols - set(column_map.values())
        raise ValueError(
            "Workbook format not recognised. Missing columns: " + ", ".join(sorted(missing))
        )

    df = df.rename(columns=column_map)

    rows_out: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        date = row.get("date")
        visitor = row.get("visitor_team")
        home = row.get("home_team")
        vpts = row.get("visitor_pts")
        hpts = row.get("home_pts")

        # Skip rows that do not look like completed games.
        if pd.isna(date) or pd.isna(visitor) or pd.isna(home):
            continue
        if pd.isna(vpts) or pd.isna(hpts):
            continue

        ot_val = row.get("ot")
        ot = False
        if isinstance(ot_val, str):
            ot = bool(ot_val.strip()) and "OT" in ot_val.upper()
        elif pd.notna(ot_val):
            # Numeric OT indicators (e.g. 1, 0). Any positive number marks OT.
            try:
                ot = int(ot_val) > 0
            except Exception:
                ot = False

        rows_out.append(
            {
                "date": str(date).strip(),
                "visitor_team": str(visitor).strip(),
                "visitor_pts": int(vpts),
                "home_team": str(home).strip(),
                "home_pts": int(hpts),
                "ot": ot,
                "game_id": None,
            }
        )

    return rows_out
