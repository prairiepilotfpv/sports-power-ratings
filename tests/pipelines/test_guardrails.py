import pandas as pd

from pipelines.guardrails import enforce_margin_sd_guardrail
from config import MARGIN_SD_GUARDRAIL_MAX, MARGIN_SD_GUARDRAIL_MIN


def _build_predictions(margins):
    return pd.DataFrame({
        "game_id": [f"g{i+1}" for i in range(len(margins))],
        "margin_sd": margins,
    })


def test_guardrail_filters_nba_predictions():
    df = _build_predictions(
        [MARGIN_SD_GUARDRAIL_MIN - 1, 8.0, MARGIN_SD_GUARDRAIL_MAX + 5]
    )
    filtered = enforce_margin_sd_guardrail(df, sport="nba")
    assert filtered["game_id"].tolist() == ["g2"]
    assert filtered.iloc[0]["margin_sd"] == 8.0


def test_guardrail_skips_other_sports_by_default():
    df = _build_predictions([MARGIN_SD_GUARDRAIL_MIN - 1, 9.0])
    filtered = enforce_margin_sd_guardrail(df, sport="nhl")
    assert filtered.shape[0] == 2


def test_guardrail_can_enforce_globally():
    df = _build_predictions(
        [MARGIN_SD_GUARDRAIL_MIN - 1, 11.0, MARGIN_SD_GUARDRAIL_MAX + 5]
    )
    filtered = enforce_margin_sd_guardrail(df, sport=None, enabled_sports=None)
    assert filtered.shape[0] == 1
    assert filtered.iloc[0]["margin_sd"] == 11.0