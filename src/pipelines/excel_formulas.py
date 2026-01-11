"""Shared Excel formula helpers for betting workbooks."""

from __future__ import annotations

from openpyxl.styles import Protection
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet


def _header_index(ws: Worksheet) -> dict[str, int]:
    return {cell.value: cell.column for cell in ws[1] if cell.value}


def _cell_ref(col: int, row: int) -> str:
    return f"{get_column_letter(col)}{row}"


def apply_ev_formulas(ws: Worksheet) -> None:
    header = _header_index(ws)
    required = {"odds", "implied_prob", "model_prob", "edge", "ev"}
    if not required.issubset(header):
        return

    odds_col = header["odds"]
    price_col = header.get("price")
    implied_col = header["implied_prob"]
    model_col = header["model_prob"]
    edge_col = header["edge"]
    ev_col = header["ev"]

    for row in range(2, ws.max_row + 1):
        odds_cell = _cell_ref(odds_col, row)
        price_cell = _cell_ref(price_col, row) if price_col else None
        implied_cell = _cell_ref(implied_col, row)
        model_cell = _cell_ref(model_col, row)
        edge_cell = _cell_ref(edge_col, row)
        ev_cell = _cell_ref(ev_col, row)

        if price_cell:
            odds_expr = f"IF(OR({price_cell}=\"\",{price_cell}=0),{odds_cell},{price_cell})"
        else:
            odds_expr = odds_cell

        ws[implied_cell].value = (
            f"=IF(OR({odds_expr}=\"\",{odds_expr}=0),\"\","
            f"IF({odds_expr}>0,100/({odds_expr}+100),-{odds_expr}/(-{odds_expr}+100)))"
        )
        ws[edge_cell].value = f"=IF(OR({model_cell}=\"\",{implied_cell}=\"\"),\"\",{model_cell}-{implied_cell})"
        ws[ev_cell].value = (
            f"=IF(OR({model_cell}=\"\",{odds_expr}=\"\",{odds_expr}=0),\"\","
            f"({model_cell}*IF({odds_expr}>0,{odds_expr}/100,100/ABS({odds_expr})))-(1-{model_cell}))"
        )


def unlock_input_columns(ws: Worksheet, columns: list[str]) -> None:
    header = _header_index(ws)
    target_cols = [header[col] for col in columns if col in header]
    if not target_cols:
        return
    for row in range(2, ws.max_row + 1):
        for col in target_cols:
            ws.cell(row=row, column=col).protection = Protection(locked=False)


def apply_model_prob_formulas_for_bets_sheet(ws: Worksheet) -> None:
    header = _header_index(ws)
    required = {
        "market_type",
        "selection",
        "line",
        "model_prob",
        "home_team",
        "away_team",
        "home_win_prob",
        "away_win_prob",
        "margin_mean",
        "margin_sd",
        "total",
        "total_sd",
    }
    if not required.issubset(header):
        return

    market_col = header["market_type"]
    selection_col = header["selection"]
    line_col = header["line"]
    model_col = header["model_prob"]
    home_team_col = header["home_team"]
    away_team_col = header["away_team"]
    home_win_col = header["home_win_prob"]
    away_win_col = header["away_win_prob"]
    margin_mean_col = header["margin_mean"]
    margin_sd_col = header["margin_sd"]
    total_col = header["total"]
    total_sd_col = header["total_sd"]

    for row in range(2, ws.max_row + 1):
        market_cell = _cell_ref(market_col, row)
        selection_cell = _cell_ref(selection_col, row)
        line_cell = _cell_ref(line_col, row)
        model_cell = _cell_ref(model_col, row)
        home_team_cell = _cell_ref(home_team_col, row)
        away_team_cell = _cell_ref(away_team_col, row)
        home_win_cell = _cell_ref(home_win_col, row)
        away_win_cell = _cell_ref(away_win_col, row)
        margin_mean_cell = _cell_ref(margin_mean_col, row)
        margin_sd_cell = _cell_ref(margin_sd_col, row)
        total_cell = _cell_ref(total_col, row)
        total_sd_cell = _cell_ref(total_sd_col, row)

        ws[model_cell].value = (
            f"=IF({market_cell}=\"ML\","
            f"IF({selection_cell}={home_team_cell},{home_win_cell},"
            f"IF({selection_cell}={away_team_cell},{away_win_cell},\"\")),"
            f"IF({market_cell}=\"spread\","
            f"IF(OR({line_cell}=\"\",{margin_sd_cell}=\"\"),\"\","
            f"IF({selection_cell}={home_team_cell},"
            f"1-NORM.DIST(ABS({line_cell}),{margin_mean_cell},{margin_sd_cell},TRUE),"
            f"IF({selection_cell}={away_team_cell},"
            f"NORM.DIST(ABS({line_cell}),{margin_mean_cell},{margin_sd_cell},TRUE),\"\"))),"
            f"IF({market_cell}=\"total\","
            f"IF(OR({line_cell}=\"\",{total_sd_cell}=\"\"),\"\","
            f"IF({selection_cell}=\"Over\","
            f"1-NORM.DIST({line_cell},{total_cell},{total_sd_cell},TRUE),"
            f"IF({selection_cell}=\"Under\","
            f"NORM.DIST({line_cell},{total_cell},{total_sd_cell},TRUE),\"\"))),"
            "\"\")))"
        )
