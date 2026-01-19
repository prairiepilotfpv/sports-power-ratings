from __future__ import annotations

from pipelines.tuning import _resolve_tuning_candidates


def test_elo_market_grid_candidate_counts() -> None:
    ml_candidates, _ = _resolve_tuning_candidates(
        "elo",
        "log_loss",
        grid_override=None,
        db_path=None,
        sport=None,
        season=None,
    )
    spread_candidates, _ = _resolve_tuning_candidates(
        "elo",
        "mae_margin",
        grid_override=None,
        db_path=None,
        sport=None,
        season=None,
    )
    total_candidates, _ = _resolve_tuning_candidates(
        "elo",
        "mae_total",
        grid_override=None,
        db_path=None,
        sport=None,
        season=None,
    )

    assert len(ml_candidates) == 9
    assert len(spread_candidates) == 9
    assert len(total_candidates) == 45
