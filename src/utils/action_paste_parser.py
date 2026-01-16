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


def _tokenize_block(lines: List[str]) -> List[tuple]:
    """Return a list of (type, value) tokens in order.

    Types: 'spread', 'total', 'odds'
    """
    tokens = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        low = s.lower()
        # skip records and headers and logo/location markers
        if _RECORD_RE.match(s):
            continue
        if any(k in low for k in ("matchup", "open", "spread", "total", "moneyline", "logo", "location", "odds")):
            continue
        # odds should be detected before spread to avoid treating 3-digit odds as spreads
        if _ODDS_RE.match(s):
            tokens.append(("odds", s))
            continue
        if _SPREAD_RE.match(s):
            tokens.append(("spread", s))
            continue
        if _TOTAL_TOKEN_RE.match(s):
            tokens.append(("total", s))
            continue
        
        # other lines ignored
    return tokens


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

        # tokenize market-relevant values
        tokens = _tokenize_block(block)

        # positional parsing per spec
        i_tok = 0
        def _pop(expected_type: str):
            nonlocal i_tok
            if i_tok >= len(tokens):
                raise ValueError(f"Missing token of type {expected_type} for matchup '{away} at {home}'")
            tp, val = tokens[i_tok]
            if tp != expected_type:
                raise ValueError(f"Expected token type {expected_type} but got {tp} (value={val}) for matchup '{away} at {home}'")
            i_tok += 1
            return val

        # Away side
        open_spread_line = None
        if i_tok < len(tokens) and tokens[i_tok][0] == "spread":
            open_spread_line = float(tokens[i_tok][1])
            i_tok += 1

        spread_away_line = float(_pop("spread"))
        spread_away_odds = int(_pop("odds"))

        over_token = _pop("total")
        if not over_token.lower().startswith("o"):
            raise ValueError(f"Expected over total token for away side but got '{over_token}' in matchup '{away} at {home}'")
        over_line = float(over_token[1:])
        over_odds = int(_pop("odds"))

        ml_away_odds = int(_pop("odds"))

        # Home side
        open_total_line = None
        if i_tok < len(tokens) and tokens[i_tok][0] == "total" and tokens[i_tok][1].lower().startswith("u"):
            open_total_line = float(tokens[i_tok][1][1:])
            i_tok += 1

        spread_home_line = float(_pop("spread"))
        spread_home_odds = int(_pop("odds"))

        under_token = _pop("total")
        if not under_token.lower().startswith("u"):
            raise ValueError(f"Expected under total token for home side but got '{under_token}' in matchup '{away} at {home}'")
        under_line = float(under_token[1:])
        under_odds = int(_pop("odds"))

        # Home moneyline may be missing in some paste formats; default to 0
        try:
            ml_home_odds = int(_pop("odds"))
        except ValueError:
            ml_home_odds = 0

        # validate totals equal
        if abs(over_line - under_line) > 1e-6:
            raise ValueError(f"Total mismatch for matchup '{away} at {home}': over {over_line} vs under {under_line}; using over value")

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
