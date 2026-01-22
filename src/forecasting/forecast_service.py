"""Shared forecasting utilities for schedule projections."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

import pandas as pd

from config import CALIBRATION_RESIDUAL_GAMES, DEFAULT_WIN_PROB_K
from data.repository import load_games, load_model_metrics
from pipelines.common import normalize_games
from pipelines.matchups import team_home_advantages
from pipelines.projection_engines import BT_NATIVE_PROJECTION_KEY, get_projection_engine
from pipelines.projections import average_total_points, fit_total_model, team_scoring_averages
from pipelines.run_rankings import build_rankings
from markets.base import Market
from models.adapters.bt_forecast_adapter import BTForecastAdapter
from models.base import resolve_model_identity
from models.forecast_contract import ForecastContract


def _completed_games(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df["home_score"].notna() & df["away_score"].notna()
    return df[mask]


def _upcoming_games(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df["home_score"].isna() | df["away_score"].isna()
    return df[mask]


def _margin_sd_fit_stats(
    played: pd.DataFrame,
    ratings: Dict[str, float],
    *,
    home_advantage: float,
) -> tuple[int, float | None, float | None]:
    if played.empty:
        return 0, None, None
    calibration_games = (
        played.sort_values("date").tail(CALIBRATION_RESIDUAL_GAMES)
        if "date" in played.columns
        else played
    )
    residuals: list[float] = []
    for _, row in calibration_games.iterrows():
        home = str(row.get("home_team", "")).strip()
        away = str(row.get("away_team", "")).strip()
        if not home or not away:
            continue
        if home not in ratings or away not in ratings:
            continue
        try:
            margin = float(row.get("home_score")) - float(row.get("away_score"))
        except Exception:
            continue
        neutral_raw = row.get("neutral", False)
        neutral = False if pd.isna(neutral_raw) else bool(neutral_raw)
        predicted_margin = (ratings[home] - ratings[away]) + (
            0.0 if neutral else home_advantage
        )
        residuals.append(margin - predicted_margin)

    if not residuals:
        return 0, None, None
    return len(residuals), float(min(residuals)), float(max(residuals))


def _rating_lookup(rankings: pd.DataFrame) -> Dict[str, float]:
    if "implied_points" not in rankings.columns:
        raise ValueError(
            "Schedule projections require implied_points; call build_rankings(include_implied_points=True)."
        )
    return {
        str(row["team"]).strip(): float(row["implied_points"])
        for _, row in rankings.iterrows()
    }


def _safe_date(value: Any) -> str:
    if value is None:
        return ""
    try:
        parsed = pd.to_datetime(value)
        if isinstance(value, date) and not isinstance(value, pd.Timestamp):
            return parsed.date().isoformat()
        text = str(value)
        if any(token in text for token in ("T", ":", "Z")):
            return parsed.isoformat()
        if parsed.time() != pd.Timestamp.min.time():
            return parsed.isoformat()
        return parsed.date().isoformat()
    except Exception:
        return str(value)


def _base_schedule_row(row: pd.Series) -> Dict[str, Any]:
    raw_date = row.get("date")
    start_time = row.get("start_time")
    sort_source = start_time if pd.notna(start_time) else raw_date
    sort_dt = pd.to_datetime(sort_source, errors="coerce")
    neutral_raw = row.get("neutral", False)
    overtime_raw = row.get("overtime", False)
    neutral = False if pd.isna(neutral_raw) else bool(neutral_raw)
    overtime = False if pd.isna(overtime_raw) else bool(overtime_raw)

    return {
        "date": _safe_date(raw_date),
        "home_team": str(row.get("home_team", "")).strip(),
        "away_team": str(row.get("away_team", "")).strip(),
        "home_score": row.get("home_score"),
        "away_score": row.get("away_score"),
        "neutral": neutral,
        "overtime": overtime,
        "game_id": row.get("game_id"),
        "_sort_dt": sort_dt,
    }


def _native_forecast_for_game(
    *,
    base: Dict[str, Any],
    projection: Dict[str, Any],
    model_instance: Any | None,
) -> ForecastContract | None:
    if model_instance is None:
        return None
    identity = resolve_model_identity(model_instance)
    if identity.get("model_id") != "bradley-terry":
        return None
    native_projection = projection.get(BT_NATIVE_PROJECTION_KEY)
    return BTForecastAdapter.from_native_projection(
        model=model_instance,
        projection=native_projection,
        game_id=base.get("game_id"),
        date=base.get("date"),
        home_team=base.get("home_team", ""),
        away_team=base.get("away_team", ""),
    )


def _project_row(
    row: pd.Series,
    *,
    ratings: Dict[str, float],
    status: str,
    home_advantage: float,
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
    model_instance: Any | None = None,
    projection_engine: Any | None = None,
    projection_context: Dict[str, Any] | None = None,
    base_total: float | None = None,
    scoring_averages: Dict[str, Any] | None = None,
    win_prob_k: float | None = None,
    total_intercept: float | None = None,
    total_slope: float | None = None,
    margin_std: float | None = None,
    total_std: float | None = None,
    conditional_sd_intercept: float | None = None,
    conditional_sd_slope: float | None = None,
) -> Dict[str, Any]:
    if projection_context is None:
        projection_context = {
            "ratings": ratings,
            "base_total": base_total,
            "scoring_averages": scoring_averages or {},
            "total_intercept": total_intercept,
            "total_slope": total_slope,
            "margin_std": margin_std,
            "total_std": total_std,
            "conditional_sd_intercept": conditional_sd_intercept,
            "conditional_sd_slope": conditional_sd_slope,
            "win_prob_k": win_prob_k,
            "rating_units": "points",
        }
    if projection_engine is None:
        projection_engine = get_projection_engine(model_instance)
    base = _base_schedule_row(row)
    home = base["home_team"]
    away = base["away_team"]

    market_label = Market.ML.name
    if params_market is not None:
        normalized_market = str(params_market).strip()
        if normalized_market:
            market_label = normalized_market.upper()
    applied_home_advantage = 0.0 if base["neutral"] else home_advantage
    home_rating = ratings.get(home)
    away_rating = ratings.get(away)

    projected_winner = None
    projected_spread = None
    projected_home_spread = None
    projected_win_prob = None
    model_p_home_win = None
    normal_p_home_win = None
    logistic_home_win_prob = None
    home_win_prob = None
    away_win_prob = None
    winner_win_prob = None
    projected_home_score = None
    projected_away_score = None
    projected_total = None
    projected_win_prob_dist = None
    model_win_prob_samples = None
    model_win_prob = None
    margin_mean = None
    total_mean = None
    margin_sd_value = None
    total_sd_value = None
    margin_dist_params = None
    total_dist_params = None
    win_prob_source = None
    margin_dist_assumption = None
    projection_status = None

    can_project = (
        home_rating is not None and away_rating is not None
    ) or (
        model_instance is not None
        and hasattr(model_instance, "simulate_matchup")
        and callable(getattr(model_instance, "simulate_matchup"))
    )
    if not can_project:
        projection_status = "missing_ratings"
    else:
        projection = projection_engine(
            home,
            away,
            model_instance,
            {
                **projection_context,
                "home_advantage": applied_home_advantage,
                "neutral": base["neutral"],
                "game_id": base.get("game_id"),
                "game_date": base.get("date"),
            },
        )
        projected_home_score = projection.get("projected_home_score")
        projected_away_score = projection.get("projected_away_score")
        projected_total = projection.get("projected_total")
        normal_p_home_win = projection.get("normal_p_home_win")
        projected_win_prob = projection.get("projected_win_prob", normal_p_home_win)
        model_p_home_win = projection.get("model_p_home_win")
        logistic_home_win_prob = projection.get("logistic_home_win_prob")
        win_prob_source = projection.get("win_prob_source")
        margin_dist_assumption = projection.get("margin_dist_assumption")
        margin_mean = projection.get("margin_mean")
        margin_sd_value = projection.get("margin_sd")
        total_mean = projection.get("total_mean")
        total_sd_value = projection.get("total_sd")
        native_forecast = _native_forecast_for_game(
            base=base,
            projection=projection,
            model_instance=model_instance,
        )
        if native_forecast is not None:
            if native_forecast.projected_home_score is not None:
                projected_home_score = native_forecast.projected_home_score
            if native_forecast.projected_away_score is not None:
                projected_away_score = native_forecast.projected_away_score
            if native_forecast.projected_total is not None:
                projected_total = native_forecast.projected_total
            if native_forecast.spread is not None:
                margin_mean = native_forecast.spread.margin_mean
                margin_sd_value = native_forecast.spread.margin_sd
            if native_forecast.total is not None:
                total_mean = native_forecast.total.total_mean
                total_sd_value = native_forecast.total.total_sd
                projected_total = (
                    native_forecast.projected_total
                    if native_forecast.projected_total is not None
                    else native_forecast.total.total_mean
                )
            if native_forecast.ml is not None:
                model_p_home_win = native_forecast.ml.p_home_win
                home_win_prob = native_forecast.ml.p_home_win
                away_win_prob = native_forecast.ml.p_away_win
                projected_win_prob = native_forecast.ml.p_home_win
        projection_has_output = any(
            value is not None
            for value in (
                projected_home_score,
                projected_away_score,
                projected_total,
                projected_win_prob,
                model_p_home_win,
                margin_mean,
                total_mean,
            )
        )
        if not projection_has_output and model_instance is not None and hasattr(
            model_instance, "simulate_matchup"
        ):
            projection_status = "no_samples"
        else:
            projection_status = "ok"

        if margin_mean is not None:
            projected_spread = -margin_mean
            projected_home_spread = -projected_spread
            projected_winner = home if margin_mean > 0 else away
        if model_p_home_win is not None:
            home_win_prob = model_p_home_win
            model_win_prob = model_p_home_win
            projected_win_prob_dist = None
            model_win_prob_samples = None
            away_win_prob = 1.0 - model_p_home_win
            if projected_winner == home:
                winner_win_prob = model_p_home_win
            elif projected_winner == away:
                winner_win_prob = away_win_prob

    result_margin = None
    result_total = None
    if base["home_score"] is not None and base["away_score"] is not None:
        try:
            result_margin = int(base["home_score"]) - int(base["away_score"])
            result_total = int(base["home_score"]) + int(base["away_score"])
        except Exception:
            result_margin = None
            result_total = None

    base.update(
        {
            "status": status,
            "projection_status": projection_status,
            "params_source": params_source,
            "params_source_label": params_source_label or params_source,
            "params_source_run_id": params_source_run_id or params_run_id,
            "tuned_metric_used": tuned_metric_used,
            "params_metric_optimized": params_metric_optimized,
            "params_best_score": params_best_score,
            "params_fingerprint": params_fingerprint,
            "params_nonempty": bool(params_nonempty) if params_nonempty is not None else bool(params_source_run_id),
            "tuning_run_id": params_run_id,
            "params_market": market_label,
            "home_rating": home_rating,
            "away_rating": away_rating,
            "projected_winner": projected_winner,
            "projected_spread": projected_spread,
            "projected_home_spread": projected_home_spread,
            "projected_win_prob": projected_win_prob,
            "model_p_home_win": model_p_home_win,
            "normal_p_home_win": normal_p_home_win,
            "home_win_prob": home_win_prob,
            "away_win_prob": away_win_prob,
            "winner_win_prob": winner_win_prob,
            "logistic_home_win_prob": logistic_home_win_prob,
            "win_prob_source": win_prob_source,
            "margin_dist_assumption": margin_dist_assumption,
            "projected_win_prob_dist": None,
            "model_win_prob_samples": model_win_prob_samples,
            "model_win_prob": model_win_prob,
            "home_advantage": applied_home_advantage,
            "projected_home_score": projected_home_score,
            "projected_away_score": projected_away_score,
            "projected_total": projected_total,
            "margin_std": projection_context.get("margin_std"),
            "total_std": projection_context.get("total_std"),
            "result_margin": result_margin,
            "result_total": result_total,
            "margin_mean": margin_mean,
            "margin_sd": margin_sd_value if margin_mean is not None else None,
            "total_mean": total_mean,
            "total_sd": total_sd_value if total_mean is not None else None,
            "margin_dist_params": margin_dist_params,
            "total_dist_params": total_dist_params,
        }
    )
    return base


def _build_forecasts_df_legacy(
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
    played = _completed_games(df)
    upcoming = _upcoming_games(df)

    rankings, model_instance = build_rankings(
        played,
        model=model,
        require_scores=False,
        model_params=model_params,
        include_implied_points=True,
        return_model=True,
    )
    ratings = _rating_lookup(rankings)
    projection_engine = get_projection_engine(model_instance)
    fallback_total = average_total_points(played.to_dict(orient="records"))
    metrics = load_model_metrics(db_path, sport=sport, season=season, model=model) or {}
    home_advantages = team_home_advantages(played, ratings)
    fallback_home_advantage = float(metrics.get("home_advantage", 0.0))
    backtest_win_prob_k = metrics.get("backtest_win_prob_k")
    win_prob_k = float(
        backtest_win_prob_k
        if backtest_win_prob_k is not None
        else metrics.get("win_prob_k", DEFAULT_WIN_PROB_K)
    )
    if win_prob_k <= 0:
        win_prob_k = DEFAULT_WIN_PROB_K
    try:
        winprob_bias = float(
            metrics.get("winprob_bias")
            if metrics.get("winprob_bias") is not None
            else getattr(getattr(model_instance, "metadata", lambda: None)(), "params", {}).get("winprob_bias", 0.0)
        )
    except Exception:
        winprob_bias = 0.0
    base_total = float(metrics.get("base_total", 0.0)) or fallback_total
    margin_std = metrics.get("margin_std")
    total_std = metrics.get("total_std")
    conditional_sd_intercept = metrics.get("conditional_sd_intercept")
    conditional_sd_slope = metrics.get("conditional_sd_slope")
    played_records = played.to_dict(orient="records")
    total_intercept, total_slope = fit_total_model(played_records, ratings)
    scoring_averages = team_scoring_averages(played_records)
    sd_sample_size, sd_residual_min, sd_residual_max = _margin_sd_fit_stats(
        played, ratings, home_advantage=fallback_home_advantage
    )
    projection_context = {
        "ratings": ratings,
        "base_total": base_total,
        "scoring_averages": scoring_averages,
        "total_intercept": total_intercept,
        "total_slope": total_slope,
        "margin_std": margin_std,
        "total_std": total_std,
        "conditional_sd_intercept": conditional_sd_intercept,
        "conditional_sd_slope": conditional_sd_slope,
        "win_prob_k": win_prob_k,
        "winprob_bias": winprob_bias,
        "sport": sport,
        "sd_sample_size": sd_sample_size,
        "sd_residual_min": sd_residual_min,
        "sd_residual_max": sd_residual_max,
        "rating_units": "points",
    }
    if model == "poisson" and model_params and "n_simulations" in model_params:
        projection_context["n_simulations"] = model_params["n_simulations"]

    schedule_rows: List[Dict[str, Any]] = []
    if not upcoming_only:
        for _, row in played.iterrows():
            home = str(row.get("home_team", "")).strip()
            schedule_rows.append(
                _project_row(
                    row,
                    ratings=ratings,
                    status="final",
                    home_advantage=home_advantages.get(home, fallback_home_advantage),
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
                    model_instance=model_instance,
                    projection_engine=projection_engine,
                    projection_context=projection_context,
                )
            )

    for _, row in upcoming.iterrows():
        home = str(row.get("home_team", "")).strip()
        schedule_rows.append(
            _project_row(
                row,
                ratings=ratings,
                status="scheduled",
                home_advantage=home_advantages.get(home, fallback_home_advantage),
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
                model_instance=model_instance,
                projection_engine=projection_engine,
                projection_context=projection_context,
            )
        )

    schedule_df = pd.DataFrame(schedule_rows)
    if not schedule_df.empty and "date" in schedule_df.columns:
        if "_sort_dt" in schedule_df.columns:
            schedule_df = (
                schedule_df.sort_values(
                    ["_sort_dt", "game_id", "away_team", "home_team"]
                )
                .drop(columns=["_sort_dt"], errors="ignore")
            )
        else:
            schedule_df = (
                schedule_df.assign(
                    _dt=pd.to_datetime(schedule_df["date"], errors="coerce")
                )
                .sort_values(["_dt", "game_id", "away_team", "home_team"])
                .drop(columns=["_dt"], errors="ignore")
            )

    for col in (
        "home_win_prob_raw",
        "away_win_prob_raw",
        "home_win_prob_calibrated",
        "away_win_prob_calibrated",
    ):
        if col not in schedule_df.columns:
            schedule_df[col] = pd.NA

    return schedule_df


def build_forecasts_df(
    *,
    db_path: str | Path,
    sport: str,
    season: str,
    model: str,
    games_df: pd.DataFrame | None = None,
    date: str | date | None = None,
    include_upcoming: bool = True,
    include_played: bool = True,
    model_params: dict[str, float] | None = None,
    params_source: str = "default",
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
    if games_df is None:
        rows = load_games(db_path, sport=sport, season=season)
        games = normalize_games(rows)
    else:
        games = games_df.copy(deep=True)

    if games.empty:
        return pd.DataFrame()

    legacy_df = _build_forecasts_df_legacy(
        games,
        db_path=db_path,
        sport=sport,
        season=season,
        model=model,
        upcoming_only=not include_played,
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

    filtered = legacy_df
    if not include_played:
        filtered = filtered[filtered.get("status") != "final"]
    if not include_upcoming:
        filtered = filtered[filtered.get("status") != "scheduled"]
    if date is not None and not filtered.empty:
        parsed = pd.to_datetime(date, errors="coerce")
        if not pd.isna(parsed):
            target_date = parsed.date()
            filtered = filtered.assign(
                _date=pd.to_datetime(filtered["date"], errors="coerce").dt.date
            )
            filtered = filtered[filtered["_date"] == target_date].drop(
                columns=["_date"], errors="ignore"
            )

    return filtered.reset_index(drop=True)
