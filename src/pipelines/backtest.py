"""Pipeline wrapper for model backtesting."""

from __future__ import annotations

from pathlib import Path

from backtest.runner import BacktestOutputs, load_games_df_from_csv, run_backtest
from models.registry import get_backtest_model


def run_backtest_pipeline(
    *,
    csv_path: str | Path,
    model: str,
    start_date: str,
    end_date: str,
    window: str = "expanding",
    rolling_days: int | None = None,
    rolling_games: int | None = None,
    output_dir: str | Path | None = None,
    db_path: str | Path | None = None,
    sport: str | None = None,
    season: str | None = None,
) -> BacktestOutputs:
    """Run a backtest for a single model and export outputs."""
    model_cls = get_backtest_model(model)
    games_df = load_games_df_from_csv(csv_path)
    return run_backtest(
        model_factory=model_cls,
        games_df=games_df,
        start_date=start_date,
        end_date=end_date,
        window=window,
        rolling_days=rolling_days,
        rolling_games=rolling_games,
        output_dir=output_dir,
        model_name=model,
        db_path=db_path,
        sport=sport,
        season=season,
    )
