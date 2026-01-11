from openpyxl import Workbook
from openpyxl.utils import get_column_letter

from src.pipelines.excel_formulas import apply_model_prob_formulas_for_bets_sheet


def test_bets_model_prob_formula_uses_erf():
    wb = Workbook()
    ws = wb.active
    ws.title = "BETS"

    headers = [
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
    ]
    ws.append(headers)
    ws.append(
        [
            "spread",
            "Home",
            -3.5,
            "",
            "Home",
            "Away",
            0.6,
            0.4,
            4.0,
            12.0,
            210.5,
            15.0,
        ]
    )

    apply_model_prob_formulas_for_bets_sheet(ws)

    model_col = headers.index("model_prob") + 1
    model_cell = f"{get_column_letter(model_col)}2"
    formula = ws[model_cell].value

    assert "ERF(" in formula
    assert "NORM.DIST" not in formula
    assert "NORMDIST" not in formula
