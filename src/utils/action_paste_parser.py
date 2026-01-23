from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


_HEADER_RE = re.compile(r"^(?P<away>.+?) at (?P<home>.+?) Odds$")
_DATE_RE = re.compile(r"^(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}$")
_RECORD_RE = re.compile(r"^\d{1,2}-\d{1,2}$")
_SPREAD_RE = re.compile(r"^[+-]\d+(?:\.\d+)?$")
_TOTAL_TOKEN_RE = re.compile(r"^[ouOU]\d+(?:\.\d+)?$")
_ODDS_RE = re.compile(r"^[+-]\d{2,3}$")


@dataclass
class GameMarkets:
    away_team: str
    home_team: str
    game_date: Optional[str]

    spread_away_line: float
    spread_away_odds: int
    spread_home_line: float
    spread_home_odds: int

    total_line: float
    over_odds: int
    under_odds: int

    ml_away_odds: int
    ml_home_odds: int

    # optional opens
    open_spread_line: Optional[float] = None
    open_total_line: Optional[float] = None


def _normalize(text: str) -> str:
    return text.replace("−", "-").replace("–", "-").strip()


def _split_team_block(block: List[str]) -> tuple[list[str], list[str]]:
    """Return the away and home-specific lines for a market block."""
    cleaned = [ln.strip() for ln in block if ln.strip()]
    start_idx = None
    for idx, ln in enumerate(cleaned):
        if ln.lower().endswith("logo"):
            start_idx = idx
            break
    if start_idx is None:
        raise ValueError("Could not find team markup (logo lines) in market block")

    cleaned = cleaned[start_idx:]
    logo_positions = [i for i, ln in enumerate(cleaned) if ln.lower().endswith("logo")]
    if len(logo_positions) < 2:
        raise ValueError("Expected two team logo lines but found fewer in market block")

    first_home_logo = logo_positions[1]
    away_lines = cleaned[:first_home_logo]
    home_lines = cleaned[first_home_logo:]
    return away_lines, home_lines


def _extract_numeric_values(lines: List[str]) -> List[str]:
    values = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        low = s.lower()
        if _RECORD_RE.match(s):
            continue
        if any(k in low for k in ("matchup", "open", "spread", "total", "moneyline", "logo", "location", "odds")):
            continue
        if _SPREAD_RE.match(s) or _TOTAL_TOKEN_RE.match(s) or _ODDS_RE.match(s):
            values.append(s)
    return values


def parse_action_paste(text: str) -> List[GameMarkets]:
    t = _normalize(text)
    lines = [ln for ln in t.splitlines() if ln.strip()]

    # find header indices
    headers = []
    for i, ln in enumerate(lines):
        m = _HEADER_RE.match(ln)
        if m:
            headers.append((i, m.group("away").strip(), m.group("home").strip()))

    games: List[GameMarkets] = []
    if not headers:
        return games

    for idx, (start_i, away, home) in enumerate(headers):
        end_i = headers[idx + 1][0] if idx + 1 < len(headers) else len(lines)
        block = lines[start_i:end_i]

        # parse date
        game_date = None
        for ln in block:
            if _DATE_RE.match(ln):
                try:
                    dt = datetime.strptime(ln, "%B %d, %Y")
                    game_date = dt.strftime("%Y-%m-%d")
                except Exception:
                    game_date = None
                break

        away_lines, home_lines = _split_team_block(block)
        away_values = _extract_numeric_values(away_lines)
        home_values = _extract_numeric_values(home_lines)

        def _require_values(count: int, section: str) -> None:
            if count < 5:
                raise ValueError(f"Missing market values for {section} in matchup '{away} at {home}'")

        open_spread_line = None
        away_idx = 0
        if (
            len(away_values) >= 6
            and _SPREAD_RE.match(away_values[0])
            and _SPREAD_RE.match(away_values[1])
        ):
            open_spread_line = float(away_values[0])
            away_idx = 1

        _require_values(len(away_values) - away_idx, "away spread/total/moneyline")
        spread_away_line = float(away_values[away_idx])
        spread_away_odds = int(away_values[away_idx + 1])
        over_token = away_values[away_idx + 2]
        if not over_token.lower().startswith("o"):
            raise ValueError(f"Expected over total token for away side but got '{over_token}' in matchup '{away} at {home}'")
        over_line = float(over_token[1:])
        over_odds = int(away_values[away_idx + 3])
        ml_away_odds = int(away_values[away_idx + 4])

        open_total_line = None
        home_idx = 0
        if (
            len(home_values) >= 6
            and _TOTAL_TOKEN_RE.match(home_values[home_idx])
            and home_values[home_idx][0].lower() in ("u", "o")
            and _SPREAD_RE.match(home_values[home_idx + 1])
        ):
            open_total_line = float(home_values[home_idx][1:])
            home_idx = 1

        _require_values(len(home_values) - home_idx, "home spread/total/moneyline")
        spread_home_line = float(home_values[home_idx])
        spread_home_odds = int(home_values[home_idx + 1])
        under_token = home_values[home_idx + 2]
        if not under_token.lower().startswith("u"):
            raise ValueError(f"Expected under total token for home side but got '{under_token}' in matchup '{away} at {home}'")
        under_line = float(under_token[1:])
        under_odds = int(home_values[home_idx + 3])

        try:
            ml_home_odds = int(home_values[home_idx + 4])
        except (IndexError, ValueError):
            ml_home_odds = 0

        # validate totals equal - prefer the over value when there's a small mismatch
        if abs(over_line - under_line) > 1e-6:
            try:
                import warnings

                warnings.warn(
                    f"Total mismatch for matchup '{away} at {home}': over {over_line} vs under {under_line}; using over value",
                    stacklevel=2,
                )
            except Exception:
                # fallback: do not fail parsing for real-world paste quirks
                pass

        gm = GameMarkets(
            away_team=away,
            home_team=home,
            game_date=game_date,
            spread_away_line=spread_away_line,
            spread_away_odds=spread_away_odds,
            spread_home_line=spread_home_line,
            spread_home_odds=spread_home_odds,
            total_line=over_line,
            over_odds=over_odds,
            under_odds=under_odds,
            ml_away_odds=ml_away_odds,
            ml_home_odds=ml_home_odds,
            open_spread_line=open_spread_line,
            open_total_line=open_total_line,
        )

        games.append(gm)

    return games


def game_to_bets_hi_block(game: GameMarkets) -> str:
    lines = [
        f"0\t{game.ml_away_odds}",
        f"0\t{game.ml_home_odds}",
        f"{game.spread_away_line}\t{game.spread_away_odds}",
        f"{game.spread_home_line}\t{game.spread_home_odds}",
        f"{game.total_line}\t{game.over_odds}",
        f"{game.total_line}\t{game.under_odds}",
    ]
    return "\n".join(lines) + "\n"


def parse_file(path: str) -> List[GameMarkets]:
    with open(path, "r", encoding="utf-8") as fh:
        txt = fh.read()
    return parse_action_paste(txt)


def _write_opens_json(games: List[GameMarkets], path: str) -> None:
    out = []
    for g in games:
        out.append({
            "away": g.away_team,
            "home": g.home_team,
            "open_spread_line": g.open_spread_line,
            "open_total_line": g.open_total_line,
        })
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    print("Run via tools/action_to_bets_paste.py")
