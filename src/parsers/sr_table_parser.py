# Parse Sports-Reference 'Schedule & Results' table into rows
# (date, visitor_team, visitor_pts, home_team, home_pts, ot, game_id)
from typing import List, Dict, Any, Optional
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
