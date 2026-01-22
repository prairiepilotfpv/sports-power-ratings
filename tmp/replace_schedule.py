from pathlib import Path

path = Path("src/pipelines/schedule.py")
text = path.read_text()
start = text.index("def _build_schedule_dataframe")
end = text.index("def _apply_calibration")
replacement = """def _build_schedule_dataframe(
    df: pd.DataFrame,
    *,
    db_path: str | Path,
    sport: str,
    season: str,
    model: str,
    upcoming_only: bool,
    model_params: dict[str, float] | None,
    params_source: str,
    params_source_label: str | None = None,
    params_source_run_id: str | None = None,
    tuned_metric_used: str | None = None,
    params_metric_optimized: str | None = None,
    params_best_score: float | None = None,
    params_fingerprint: str | None = None,
    params_nonempty: bool | None = None,
    params_run_id: str | None = None,
    params_market: str | None = None,
) -> pd.DataFrame:
    return build_forecasts_df(
        db_path=db_path,
        sport=sport,
        season=season,
        model=model,
        games_df=df,
        include_played=not upcoming_only,
        include_upcoming=True,
        model_params=model_params,
        params_source=params_source,
        params_source_label=params_source_label,
        params_source_run_id=params_source_run_id,
        tuned_metric_used=tuned_metric_used,
        params_metric_optimized=params_metric_optimized,
        params_best_score=params_best_score,
        params_fingerprint=params_fingerprint,
        params_nonempty=params_nonempty,
        params_run_id=params_run_id,
        params_market=params_market,
    )

"""
path.write_text(text[:start] + replacement + text[end:])
