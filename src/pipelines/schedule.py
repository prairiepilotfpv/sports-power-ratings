"""Schedule export pipeline with projection fields."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import re
import sqlite3

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles.numbers import FORMAT_CURRENCY_USD_SIMPLE

from contracts import SCHEDULE_EXPORT_COLUMNS, validate_schedule_export_frame
from data.paths import processed_path_for
from data.repository import (
    get_active_ensemble_market_weights,
    get_active_ensemble_market_weights_source,
    get_active_model_market_params,
    get_active_model_market_params_source,
    load_games,
    load_model_metrics,
)
from data import teams as team_repo
from forecasting import build_forecasts_df
from forecasting.forecast_service import (
    _completed_games,
    _project_row as _forecast_project_row,
    _rating_lookup as _forecast_rating_lookup,
    _safe_date,
)
from pipelines.model_params import (
    resolve_active_model_market_params,
    resolve_active_ensemble_weights,
    resolve_effective_params,
    resolve_model_market_params_with_metadata,
)
resolve_model_params_with_metadata = resolve_model_market_params_with_metadata
from pipelines.common import normalize_games, resolve_output_path
from ensemble.config import load_ensemble_config
from ensemble.io import load_market_weights
from pipelines.excel_bets_format import apply_bets_sheet_formatting
from pipelines.excel_formulas import (
    apply_ev_formulas,
    apply_model_prob_formulas_for_bets_sheet,
    validate_bets_formulas,
    validate_no_ellipsis_formulas,
)
from pipelines.market_utils import _metric_name_for_market
from pipelines.matchups import team_home_advantages
from pipelines.projection_engines import get_projection_engine
from pipelines.projections import (
    average_total_points,
    fit_total_model,
    team_scoring_averages,
)
from pipelines.metadata import prediction_hash
from calibration.io import load_latest_calibrator
from models.base import resolve_model_identity
from models.registry import get_model, list_models, normalize_model_name
from markets.base import Market


_project_row = _forecast_project_row
_rating_lookup = _forecast_rating_lookup

EXPORT_MARKETS: List[Market] = [Market.ML, Market.SPREAD, Market.TOTAL]
MARKET_ORDER: Dict[str, int] = {market.name: idx for idx, market in enumerate(EXPORT_MARKETS)}

DASHBOARD_COLUMNS: List[str] = [
    "model",
    "model_version",
    "params_source",
    "params_source_label",
    "params_source_run_id",
    "tuned_metric_used",
    "params_metric_optimized",
    "params_best_score",
    "params_fingerprint",
    "params_nonempty",
    "tuning_run_id",
    "run_timestamp_utc",
    "prediction_hash",
    "date",
    "params_market",
    "game",
    "projected_home_score",
    "projected_away_score",
    "total",
    "projected_winner",
    "projected_spread",
    "margin_mean",
    "margin_sd",
    "model_p_home_win",
    "normal_p_home_win",
    "home_win_prob",
    "away_win_prob",
    "winner_win_prob",
    "logistic_home_win_prob",
    "win_prob_source",
    "margin_dist_assumption",
    "total_sd",
]

MODEL_METADATA_DATA_START_ROW = 50  # Allow multi-market metadata; bump if new metadata keys appear


def _finalize_schedule_export(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the export includes all expected columns before validation."""
    finalized = schedule_df.copy()
    required_columns = ["date", "game_id"]
    missing_required = [col for col in required_columns if col not in finalized.columns]
    if missing_required:
        raise ValueError(
            f"Missing required schedule export columns: {missing_required}"
        )
    for column in SCHEDULE_EXPORT_COLUMNS:
        if column not in finalized.columns:
            finalized[column] = pd.NA
    return finalized.loc[:, SCHEDULE_EXPORT_COLUMNS]


def _order_schedule_export(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a consistent column order for downstream reporting."""
    finalized = _finalize_schedule_export(schedule_df)
    return validate_schedule_export_frame(
        finalized,
        expected_columns=SCHEDULE_EXPORT_COLUMNS,
        context="Schedule export",
    )


def _resolve_models(model: str | None) -> list[str]:
    if model is None:
        return list_models()
    normalized = normalize_model_name(model)
    if normalized in {"all", "*"}:
        return list_models()
    return [normalized]


def _resolve_workbook_path(
    output_path: str | Path | None,
    *,
    sport: str,
    season: str,
    default_name: str,
) -> Path:
    default_path = processed_path_for(sport, season, default_name)
    resolved = resolve_output_path(
        output_path,
        default_path=default_path,
    )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _build_schedule_dataframe(
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
    schedule_df = build_forecasts_df(
        db_path=db_path,
        sport=sport,
        season=season,
        model=model,
        games_df=df,
        include_upcoming=True,
        include_played=not upcoming_only,
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
    schedule_df = _apply_calibration_to_schedule_df(
        schedule_df,
        sport=sport,
        season=season,
        model=model,
    )
    for col in (
        "home_win_prob_raw",
        "away_win_prob_raw",
        "home_win_prob_calibrated",
        "away_win_prob_calibrated",
    ):
        if col not in schedule_df.columns:
            schedule_df[col] = pd.NA
    return _order_schedule_export(schedule_df)


def _apply_calibration_to_schedule_df(
    schedule_df: pd.DataFrame,
    *,
    sport: str,
    season: str,
    model: str,
) -> pd.DataFrame:
    """Apply the latest ML calibrator, if available, to schedule win probabilities."""
    if schedule_df.empty or "home_win_prob" not in schedule_df.columns:
        return schedule_df

    calibrator = load_latest_calibrator(
        sport=sport,
        season=season,
        model=model,
        market=Market.ML,
    )
    if calibrator is None:
        return schedule_df

    df = schedule_df.copy()
    raw_probs = pd.to_numeric(df["home_win_prob"], errors="coerce")
    valid_mask = raw_probs.notna()
    if not valid_mask.any():
        return df

    # Calibrators should be stored under outputs/calibrators/<sport>/<season>/<model>.
    df["home_win_prob_raw"] = df["home_win_prob"]
    df["away_win_prob_raw"] = df["away_win_prob"]

    try:
        calibrated = pd.Series(pd.NA, index=df.index, dtype="float")
        calibrated_values = calibrator.transform(raw_probs.loc[valid_mask].astype(float))
        calibrated.loc[valid_mask] = pd.to_numeric(calibrated_values, errors="coerce")
        df["home_win_prob_calibrated"] = calibrated
        df["away_win_prob_calibrated"] = 1.0 - calibrated
    except Exception:
        return df

    calibrated_mask = df["home_win_prob_calibrated"].notna()
    df.loc[calibrated_mask, "home_win_prob"] = df.loc[
        calibrated_mask, "home_win_prob_calibrated"
    ]
    df.loc[calibrated_mask, "away_win_prob"] = df.loc[
        calibrated_mask, "away_win_prob_calibrated"
    ]

    if "projected_winner" in df.columns:
        home_wins = df["projected_winner"] == df["home_team"]
        away_wins = df["projected_winner"] == df["away_team"]
        df.loc[home_wins, "winner_win_prob"] = df.loc[home_wins, "home_win_prob"]
        df.loc[away_wins, "winner_win_prob"] = df.loc[away_wins, "away_win_prob"]

    if "win_prob_source" in df.columns:
        def _append_calibrated(value: Any) -> str:
            if value is None:
                return "calibrated"
            text = str(value).strip()
            if not text:
                return "calibrated"
            if "calibrated" in text.lower():
                return text
            return f"{text}+calibrated"

        df["win_prob_source"] = df["win_prob_source"].apply(_append_calibrated)

    return df


def _build_market_forecasts_for_ensembles(
    schedule_df: pd.DataFrame,
    *,
    db_path: str | Path,
    sport: str,
    season: str,
    models: list[str],
    as_of_date: date,
    allowed_models_by_market: dict[str, list[str] | None] | None = None,
    market_metrics: dict[str, str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    rows_by_market: dict[str, list[dict[str, Any]]] = {
        Market.ML.name: [],
        Market.SPREAD.name: [],
        Market.TOTAL.name: [],
    }
    if schedule_df.empty:
        return rows_by_market

    for market in (Market.ML, Market.SPREAD, Market.TOTAL):
        market_allowed = None
        if allowed_models_by_market and market.name in allowed_models_by_market:
            market_allowed = allowed_models_by_market.get(market.name)
        model_iter = market_allowed if market_allowed else models
        metric_for_market = None
        if market_metrics and market.name in market_metrics:
            metric_for_market = market_metrics.get(market.name)
        for model_name in model_iter:
            # Resolve active market params (active -> best run -> defaults)
            resolved = resolve_effective_params(
                db_path=db_path,
                sport=sport,
                season=season,
                model=model_name,
                market=market.name,
            )
            params = resolved.params
            params_source = resolved.params_source_label
            tuned_metric_used = (
                resolved.metric_optimized.replace("backtest_", "", 1)
                if resolved.metric_optimized and str(resolved.metric_optimized).startswith("backtest_")
                else resolved.metric_optimized
            )

            model_df = schedule_df.copy(deep=True)
            market_schedule = _build_schedule_dataframe(
                model_df,
                db_path=db_path,
                sport=sport,
                season=season,
                model=model_name,
                upcoming_only=True,
                model_params=params,
                params_source=params_source,
                params_source_label=resolved.params_source_label,
                params_source_run_id=resolved.source_run_id,
                tuned_metric_used=tuned_metric_used,
                params_metric_optimized=resolved.metric_optimized,
                params_best_score=resolved.best_score,
                params_fingerprint=resolved.params_fingerprint,
                params_nonempty=resolved.params_nonempty,
                params_run_id=resolved.source_run_id,
                params_market=market.name,
            )
            if market == Market.ML:
                market_schedule = _apply_calibration_to_schedule_df(
                    market_schedule,
                    sport=sport,
                    season=season,
                    model=model_name,
                )
            if market_schedule.empty:
                continue
            if "date" in market_schedule.columns and "status" in market_schedule.columns:
                _dt = pd.to_datetime(market_schedule["date"], errors="coerce").dt.date
                mask = (market_schedule["status"] == "scheduled") & (_dt == as_of_date)
                subset = market_schedule.loc[mask]
            else:
                subset = market_schedule
            if subset.empty:
                continue
            for _, r in subset.iterrows():
                if market == Market.ML:
                    p_raw = None
                    if (
                        "home_win_prob_raw" in subset.columns
                        and pd.notna(r.get("home_win_prob_raw"))
                    ):
                        p_raw = r.get("home_win_prob_raw")
                    elif pd.notna(r.get("model_p_home_win")):
                        p_raw = r.get("model_p_home_win")
                    else:
                        p_raw = r.get("home_win_prob")
                    rows_by_market[market.name].append(
                        {
                            "game_id": r.get("game_id"),
                            "model_name": model_name,
                            "p_home_win": p_raw,
                            "margin_mean": r.get("margin_mean"),
                            "margin_sd": r.get("margin_sd"),
                            "total_mean": (
                                r.get("total")
                                if not _is_missing(r.get("total"))
                                else r.get("total_mean")
                            ),
                            "total_sd": r.get("total_sd"),
                        }
                    )
                else:
                    rows_by_market[market.name].append(
                        {
                            "game_id": r.get("game_id"),
                            "model_name": model_name,
                            "p_home_win": r.get("home_win_prob"),
                            "margin_mean": r.get("margin_mean"),
                            "margin_sd": r.get("margin_sd"),
                            "total_mean": (
                                r.get("total")
                                if not _is_missing(r.get("total"))
                                else r.get("total_mean")
                            ),
                            "total_sd": r.get("total_sd"),
                        }
                    )
    return rows_by_market


def _validate_market_tuning_inputs(
    *,
    db_path: str | Path,
    sport: str,
    season: str,
    models: list[str],
    ensemble_ids: dict[str, str],
    strict: bool,
) -> None:
    missing_model_market: list[tuple[str, str]] = []
    missing_ensemble_market: list[tuple[str, str]] = []
    for market in (Market.ML, Market.SPREAD, Market.TOTAL):
        for model_name in models:
            resolved = resolve_effective_params(
                db_path=db_path,
                sport=sport,
                season=season,
                model=model_name,
                market=market.name,
            )
            if resolved.params is None:
                missing_model_market.append((model_name, market.name))
        ensemble_id = ensemble_ids.get(market.name)
        if ensemble_id:
            weights = get_active_ensemble_market_weights(
                db_path,
                sport=sport,
                season=season,
                market=market.name,
                ensemble_id=ensemble_id,
            )
            if weights is None:
                missing_ensemble_market.append((market.name, ensemble_id))
    if missing_model_market or missing_ensemble_market:
        print(f"[tuning] DB: {db_path}")
        for model_name, market_name in missing_model_market:
            print(
                "[tuning] Missing active params for "
                f"model={model_name} market={market_name}; using defaults."
            )
        for market_name, ensemble_id in missing_ensemble_market:
            print(
                "[tuning] Missing active ensemble weights for "
                f"market={market_name} ensemble_id={ensemble_id}; using file or equal weights."
            )
        if missing_model_market:
            model_arg = (
                models[0] if len(models) == 1 else "all"
            )
            suggestion = (
                "python -m src.cli.pipeline bootstrap-market-actives "
                f"--sport {sport} --season {season} --model {model_arg}"
            )
            print(f"[tuning] To bootstrap missing model market actives, run: {suggestion}")
        if strict:
            missing_count = len(missing_model_market) + len(missing_ensemble_market)
            raise ValueError(
                "Missing active market tuning inputs "
                f"({missing_count}). Run bootstrap-market-actives or disable --strict."
            )


def _collect_market_param_sources(
    *,
    db_path: str | Path,
    sport: str,
    season: str,
    models: list[str],
) -> dict[str, dict[str, str | None]]:
    sources: dict[str, dict[str, str | None]] = {}
    for market in (Market.ML, Market.SPREAD, Market.TOTAL):
        market_sources: dict[str, str | None] = {}
        for model_name in models:
            resolved = resolve_effective_params(
                db_path=db_path,
                sport=sport,
                season=season,
                model=model_name,
                market=market.name,
            )
            if resolved.params is not None:
                metric_display = (
                    resolved.metric_optimized.replace("backtest_", "", 1)
                    if resolved.metric_optimized and str(resolved.metric_optimized).startswith("backtest_")
                    else resolved.metric_optimized
                )
                # Include tuned metric in reported source when available
                market_sources[model_name] = (
                    f"{resolved.params_source_label}/{metric_display}"
                    if metric_display
                    else resolved.params_source_label
                )
            else:
                market_sources[model_name] = None
        sources[market.name] = market_sources
    return sources


def _collect_ensemble_weight_sources(
    *,
    db_path: str | Path,
    sport: str,
    season: str,
    ensemble_ids: dict[str, str],
) -> dict[str, str | None]:
    sources: dict[str, str | None] = {}
    for market in (Market.ML, Market.SPREAD, Market.TOTAL):
        ensemble_id = ensemble_ids.get(market.name)
        if not ensemble_id:
            continue
        sources[market.name] = get_active_ensemble_market_weights_source(
            db_path,
            sport=sport,
            season=season,
            market=market.name,
            ensemble_id=ensemble_id,
        )
    return sources


def _format_game_name(away_team: Any, home_team: Any) -> str:
    """Render a simple matchup label."""
    away = str(away_team or "").strip()
    home = str(home_team or "").strip()
    if away and home:
        return f"{away} @ {home}"
    return away or home


def _dashboard_rows_for_today(
    schedule_df: pd.DataFrame,
    model_name: str,
    as_of_date: date | None = None,
    *,
    model_metadata: dict[str, Any] | None = None,
) -> list[Dict[str, Any]]:
    """Collect scheduled games for the current day for the dashboard sheet."""
    if schedule_df.empty:
        return []

    today = as_of_date or pd.Timestamp.today().date()
    df = schedule_df.assign(
        _date=pd.to_datetime(schedule_df["date"], errors="coerce").dt.date
    )
    df = df[df["status"] == "scheduled"]
    if df.empty:
        return []

    if as_of_date is not None:
        df = df[df["_date"] == today]
    else:
        df = df[df["_date"] == today]

    if df.empty:
        return []

    rows: list[Dict[str, Any]] = []
    metadata = model_metadata or {}
    for _, row in df.iterrows():
        projected_home_score = row.get("projected_home_score")
        projected_away_score = row.get("projected_away_score")
        projected_total = row.get("projected_total")
        total_mean = row.get("total_mean")
        if not _is_missing(total_mean):
            total = total_mean
        elif projected_home_score is not None and projected_away_score is not None:
            total = float(projected_home_score) + float(projected_away_score)
        else:
            total = projected_total
        rows.append(
            {
                "_sort_dt": pd.to_datetime(row.get("date"), errors="coerce"),
                "_sort_game_id": row.get("game_id"),
                "model": model_name,
                "model_version": metadata.get("model_version"),
                "params_source": metadata.get("params_source"),
                "params_source_label": metadata.get("params_source_label"),
                "params_source_run_id": metadata.get("params_source_run_id"),
                "tuned_metric_used": metadata.get("tuned_metric_used"),
                "params_metric_optimized": metadata.get("params_metric_optimized"),
                "params_best_score": metadata.get("params_best_score"),
                "params_fingerprint": metadata.get("params_fingerprint"),
                "params_nonempty": metadata.get("params_nonempty"),
                "tuning_run_id": metadata.get("tuning_run_id"),
                "params_market": metadata.get("params_market"),
                "run_timestamp_utc": metadata.get("run_timestamp_utc"),
                "prediction_hash": metadata.get("prediction_hash"),
                "date": row.get("date"),
                "game": _format_game_name(row.get("away_team"), row.get("home_team")),
                "projected_home_score": projected_home_score,
                "projected_away_score": projected_away_score,
                "total": total,
                "projected_winner": row.get("projected_winner"),
                "projected_spread": row.get("projected_spread"),
                "margin_mean": row.get("margin_mean"),
                "margin_sd": row.get("margin_sd"),
                "model_p_home_win": row.get("model_p_home_win"),
                "normal_p_home_win": row.get("normal_p_home_win"),
                "home_win_prob": row.get("home_win_prob"),
                "away_win_prob": row.get("away_win_prob"),
                "winner_win_prob": row.get("winner_win_prob"),
                "logistic_home_win_prob": row.get("logistic_home_win_prob"),
                "win_prob_source": row.get("win_prob_source"),
                "margin_dist_assumption": row.get("margin_dist_assumption"),
                "total_sd": row.get("total_sd"),
            }
        )
    return rows


def _resolve_as_of_date(value: str | date | None) -> date:
    if value is None:
        return pd.Timestamp.today().date()
    if isinstance(value, date):
        return value
    try:
        return pd.to_datetime(value).date()
    except Exception as exc:
        raise ValueError(f"Invalid as_of_date: {value}") from exc


def _resolve_bets_model(models: list[str], bets_model: str | None) -> str:
    if bets_model:
        normalized = normalize_model_name(bets_model)
        if normalized not in models:
            raise ValueError(
                f"bets_model={bets_model!r} not in schedule models: {', '.join(models)}"
            )
        return normalized
    if len(models) == 1:
        return models[0]
    return models[0]


def _sanitize_source_id(value: Any, *, default: str = "direct") -> str:
    text = _normalize_source_label(value, default=default).lower()
    cleaned = re.sub(r"[^a-z0-9._-]+", "-", text).strip("-")
    return cleaned or default


def _normalize_source_label(value: Any, *, default: str = "direct") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    return text if text else default


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return isinstance(value, str) and not value.strip()


def _first_nonempty_source(
    schedule_df: pd.DataFrame, column: str, *, as_of_date: date
) -> str | None:
    if schedule_df.empty or column not in schedule_df.columns:
        return None
    subset = schedule_df
    try:
        if "status" in schedule_df.columns and "date" in schedule_df.columns:
            _dt = pd.to_datetime(schedule_df["date"], errors="coerce").dt.date
            mask = (schedule_df["status"] == "scheduled") & (_dt == as_of_date)
            subset = schedule_df.loc[mask]
    except Exception:
        subset = schedule_df
    for value in subset[column]:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        text = str(value).strip()
        if text:
            return text
    return None


def _deterministic_review_run_id(
    *,
    sport: str,
    season: str,
    as_of_date: date,
    ml_source_id: str,
    spread_source_id: str,
    total_source_id: str,
) -> str:
    ml_id = _sanitize_source_id(ml_source_id)
    spread_id = _sanitize_source_id(spread_source_id)
    total_id = _sanitize_source_id(total_source_id)
    return (
        f"schedule-{sport}-{season}-{as_of_date.isoformat()}"
        f"-ml_{ml_id}-spread_{spread_id}-total_{total_id}"
    )


def _build_bets_dataframe(
    schedule_df: pd.DataFrame,
    *,
    model_name: str,
    as_of_date: date,
    review_run_id: str,
    db_path: str | Path | None = None,
    sport: str | None = None,
    season: str | None = None,
) -> pd.DataFrame:
    include_calibrated = (
        "home_win_prob_calibrated" in schedule_df.columns
        and schedule_df["home_win_prob_calibrated"].notna().any()
    )
    bets_columns = [
        "review_run_id",
        "game_id",
        "date",
        "away_team",
        "home_team",
        "market_type",
        "selection",
        "line",
        "odds",
        "price",
        "implied_prob",
        "model_prob",
        "edge",
        "ev",
        "stake",
        "book",
        "bet_id",
        "logged_at",
        "log_status",
        "notes",
        "source_market_snapshot_id",
        "opportunity_id",
        "home_win_prob",
        "away_win_prob",
        "win_prob_source",
    ]
    if include_calibrated:
        bets_columns.extend(
            [
                "home_win_prob_raw",
                "away_win_prob_raw",
                "home_win_prob_calibrated",
                "away_win_prob_calibrated",
            ]
        )
    bets_columns.extend(
        [
            "margin_mean",
            "margin_sd",
            "total",
            "total_sd",
            "model",
            "market_forecast_source",
            "ml_ensemble_components_json",
            "spread_source",
            "spread_ensemble_components_json",
            "total_source",
            "total_ensemble_components_json",
        ]
    )

    if schedule_df.empty:
        return pd.DataFrame(columns=bets_columns)

    df = schedule_df.assign(
        _date=pd.to_datetime(schedule_df["date"], errors="coerce").dt.date
    )
    df = df[df["status"] == "scheduled"]
    df = df[df["_date"] == as_of_date]
    # Ensure deterministic ordering so selection rows list the away team above
    # the home team (matches schedule sheet presentation). Sort by date,
    # game_id, and away_team to guarantee away-first ordering per game.
    sort_cols: list[str] = []
    if "date" in df.columns:
        sort_cols.append("date")
    if "game_id" in df.columns:
        sort_cols.append("game_id")
    # final tie-breaker: away_team first
    if "away_team" in df.columns:
        sort_cols.append("away_team")
    if sort_cols:
        try:
            df = df.sort_values(by=sort_cols, kind="mergesort")
        except Exception:
            # best-effort: ignore sorting failures and continue
            pass
    if df.empty:
        return pd.DataFrame(columns=bets_columns)

    team_id_cache: dict[tuple[str | None, str | None, str], int | None] = {}

    def _resolve_team_id_for(name: str | None) -> int | None:
        if not name or db_path is None or not sport or not season:
            return None
        key = (sport, season, name)
        if key in team_id_cache:
            return team_id_cache[key]
        team_id = None
        try:
            with sqlite3.connect(Path(db_path)) as team_conn:
                team_id = team_repo.get_team_id(
                    team_conn,
                    sport=sport,
                    season=season,
                    canonical_name=name,
                )
        except Exception:
            team_id = None
        team_id_cache[key] = team_id
        return team_id

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        projected_home_score = row.get("projected_home_score")
        projected_away_score = row.get("projected_away_score")
        projected_total = row.get("projected_total")
        total = row.get("total")
        total_mean = row.get("total_mean")
        if _is_missing(total):
            if not _is_missing(total_mean):
                total = total_mean
            elif projected_home_score is not None and projected_away_score is not None:
                total = float(projected_home_score) + float(projected_away_score)
            else:
                total = projected_total

        base = {
            "review_run_id": review_run_id,
            "game_id": row.get("game_id"),
            "date": _safe_date(row.get("date")),
            "away_team": row.get("away_team"),
            "home_team": row.get("home_team"),
            "line": "",
            "odds": "",
            "price": "",
            "implied_prob": "",
            "model_prob": "",
            "edge": "",
            "ev": "",
            "stake": "",
            "book": "",
            "bet_id": "",
            "logged_at": "",
            "log_status": "",
            "notes": "",
            "source_market_snapshot_id": "",
            "opportunity_id": "",
            "model": model_name,
            "market_forecast_source": "",
        }
        forecast_blanks = {
            "home_win_prob": "",
            "away_win_prob": "",
            "win_prob_source": "",
            "margin_mean": "",
            "margin_sd": "",
            "total": "",
            "total_sd": "",
            "ml_ensemble_components_json": "",
            "spread_source": "",
            "spread_ensemble_components_json": "",
            "total_source": "",
            "total_ensemble_components_json": "",
        }
        ml_source_label = _normalize_source_label(
            row.get("win_prob_source") or row.get("ml_source")
        )
        spread_source_label = _normalize_source_label(row.get("spread_source"))
        total_source_label = _normalize_source_label(row.get("total_source"))

        win_prob_source = row.get("win_prob_source")
        if _is_missing(win_prob_source):
            win_prob_source = ml_source_label

        spread_source = row.get("spread_source")
        if _is_missing(spread_source):
            spread_source = spread_source_label

        total_source = row.get("total_source")
        if _is_missing(total_source):
            total_source = total_source_label

        ml_fields = {
            "home_win_prob": row.get("home_win_prob"),
            "away_win_prob": row.get("away_win_prob"),
            "win_prob_source": win_prob_source,
            "ml_ensemble_components_json": row.get("ml_ensemble_components_json"),
        }
        spread_fields = {
            "margin_mean": row.get("margin_mean"),
            "margin_sd": row.get("margin_sd"),
            "spread_source": spread_source,
            "spread_ensemble_components_json": row.get("spread_ensemble_components_json"),
        }
        total_fields = {
            "total": total,
            "total_sd": row.get("total_sd"),
            "total_source": total_source,
            "total_ensemble_components_json": row.get("total_ensemble_components_json"),
        }
        if not include_calibrated:
            for key in (
                "home_win_prob_raw",
                "away_win_prob_raw",
                "home_win_prob_calibrated",
                "away_win_prob_calibrated",
            ):
                base.pop(key, None)
        else:
            forecast_blanks.update(
                {
                    "home_win_prob_raw": "",
                    "away_win_prob_raw": "",
                    "home_win_prob_calibrated": "",
                    "away_win_prob_calibrated": "",
                }
            )
            ml_fields.update(
                {
                    "home_win_prob_raw": row.get("home_win_prob_raw"),
                    "away_win_prob_raw": row.get("away_win_prob_raw"),
                    "home_win_prob_calibrated": row.get("home_win_prob_calibrated"),
                    "away_win_prob_calibrated": row.get("away_win_prob_calibrated"),
                }
            )

        home_team = row.get("home_team")
        away_team = row.get("away_team")
        home_team_id = _resolve_team_id_for(home_team)
        away_team_id = _resolve_team_id_for(away_team)

        # Lookup latest market snapshot for each market/selection as-of the workbook date
        # only when a db_path is provided; otherwise leave line/odds blank.
        br = None
        staging_rows: list[dict] | None = None
        if db_path is not None:
            try:
                from src.data import betting_repository as br
                # load staging rows once to allow permissive population of BETS
                try:
                    staging_rows = br.list_staging_rows(db_path) or []
                except Exception:
                    staging_rows = []

                # Attempt to resolve staging rows to games in-memory (do not persist).
                # Use resolve_staging_to_game when sport/season available to improve matching accuracy.
                try:
                    if sport and season:
                        for s in staging_rows:
                            try:
                                if s.get("game_id"):
                                    continue
                                res = br.resolve_staging_to_game(
                                    db_path,
                                    sport=sport,
                                    season=season,
                                    team_home_raw=s.get("team_home_raw"),
                                    team_away_raw=s.get("team_away_raw"),
                                    game_date=s.get("game_date"),
                                )
                                if res and res.get("match_confidence") is not None and float(res.get("match_confidence")) >= 0.75:
                                    s["game_id"] = res.get("game_id")
                                    s["match_confidence"] = res.get("match_confidence")
                                    s["match_status"] = res.get("match_status")
                            except Exception:
                                continue
                except Exception:
                    pass
            except Exception:
                br = None

        for mtype, sel, mlabel in [
            ("ML", away_team, ml_source_label),
            ("ML", home_team, ml_source_label),
            ("spread", away_team, spread_source_label),
            ("spread", home_team, spread_source_label),
            ("total", "Over", total_source_label),
            ("total", "Under", total_source_label),
        ]:
            snap = None
            if br is not None:
                try:
                    selection_team_id = None
                    selection_value = None
                    if mtype in ("ML", "spread"):
                        if sel == home_team:
                            selection_team_id = home_team_id
                        elif sel == away_team:
                            selection_team_id = away_team_id
                    else:
                        selection_value = sel
                    if sport and season and row.get("game_id"):
                        if selection_team_id is not None:
                            snap = br.get_latest_market_line(
                                db_path,
                                sport=sport,
                                season=season,
                                game_id=str(row.get("game_id")),
                                market_type=mtype,
                                selection_team_id=selection_team_id,
                                as_of=as_of_date.isoformat(),
                            )
                        elif selection_value is not None:
                            snap = br.get_latest_market_line(
                                db_path,
                                sport=sport,
                                season=season,
                                game_id=str(row.get("game_id")),
                                market_type=mtype,
                                selection=selection_value,
                                as_of=as_of_date.isoformat(),
                            )
                except Exception:
                    snap = None

            # If no committed snapshot, try permissive match from staging rows (parser imports).
            staged_match = None
            if snap is None and staging_rows is not None:
                try:
                    # determine iso date for matching
                    gdate = base.get("date")
                    if hasattr(gdate, "isoformat"):
                        gdate_str = gdate.isoformat()
                    else:
                        gdate_str = str(row.get("date"))[:10]

                    sel_norm = (str(sel) or "").lower()
                    home_norm = (str(home_team) or "").lower()
                    away_norm = (str(away_team) or "").lower()

                    for s in staging_rows:
                        try:
                            if not s:
                                continue
                            if s.get("market_type") != mtype:
                                continue
                            # date must match
                            if not s.get("game_date") or str(s.get("game_date"))[:10] != gdate_str:
                                continue
                            # (fallback) prefer staging rows that already resolved to this game_id
                            # NOTE: do not accept a generic game_id match for ML/spread before trying selection-side matching
                            # direct selection match for non-total markets (team names etc.)
                            if mtype != "total" and str(s.get("selection") or "").lower() == sel_norm:
                                staged_match = s
                                break
                            # for team-based markets (ML/spread) prefer matching the correct side (home vs away)
                            if mtype in ("ML", "spread"):
                                th = str(s.get("team_home_raw") or "").lower()
                                ta = str(s.get("team_away_raw") or "").lower()
                                s_sel = str(s.get("selection") or "").lower()

                                def _matches_team(team_norm: str) -> bool:
                                    if not team_norm:
                                        return False
                                    if s_sel and (
                                        s_sel == team_norm
                                        or s_sel in team_norm
                                        or team_norm in s_sel
                                    ):
                                        return True
                                    return team_norm in th or team_norm in ta

                                # If the schedule selection is the home team, prefer staging rows that reference the home team
                                if sel_norm == home_norm and _matches_team(home_norm):
                                    staged_match = s
                                    break
                                # If the schedule selection is the away team, prefer staging rows that reference the away team
                                if sel_norm == away_norm and _matches_team(away_norm):
                                    staged_match = s
                                    break
                                # fallback: exact selection match
                                if s_sel == sel_norm:
                                    staged_match = s
                                    break
                            # totals: ensure the staging row references the same pairing (home+away)
                            if mtype == "total":
                                th = str(s.get("team_home_raw") or "").lower()
                                ta = str(s.get("team_away_raw") or "").lower()
                                # require both teams to appear in the staging row (order-insensitive)
                                if ((home_norm in th or home_norm in ta) and (away_norm in th or away_norm in ta)):
                                    staged_match = s
                                    break

                            # fallback: if the staging row is already resolved to this game, accept for totals or unspecified selections
                            s_sel = str(s.get("selection") or "").lower()
                            if s.get("game_id") and str(s.get("game_id")) == str(row.get("game_id")):
                                if mtype == "total" or s_sel in ("", "over", "under"):
                                    staged_match = s
                                    break
                        except Exception:
                            continue
                except Exception:
                    staged_match = None

            rows.append(
                {
                    **base,
                    **forecast_blanks,
                    **(ml_fields if mtype == "ML" else (spread_fields if mtype == "spread" else total_fields)),
                    "market_type": mtype,
                    "selection": sel,
                    "model": mlabel,
                    "market_forecast_source": mlabel,
                    "line": (
                        snap.get("line")
                        if snap and snap.get("line") is not None
                        else (staged_match.get("line") if staged_match and staged_match.get("line") is not None else "")
                    ),
                    "odds": (
                        int(snap.get("odds"))
                        if snap and snap.get("odds") is not None
                        else (int(staged_match.get("odds")) if staged_match and staged_match.get("odds") is not None else "")
                    ),
                    "source_market_snapshot_id": (
                        snap.get("id")
                        if snap and snap.get("id") is not None
                        else (staged_match.get("id") if staged_match and staged_match.get("id") is not None else "")
                    ),
                }
            )

    return pd.DataFrame(rows, columns=bets_columns)


def _training_date_range(df: pd.DataFrame) -> str:
    if df.empty or "date" not in df.columns:
        return ""
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        return ""
    return f"{dates.min().date().isoformat()} to {dates.max().date().isoformat()}"


def _serialize_params(params: Any) -> str:
    try:
        return json.dumps(params, sort_keys=True, default=str)
    except TypeError:
        return str(params)


def _build_model_metadata(
    *,
    model_name: str,
    played: pd.DataFrame,
    schedule_df: pd.DataFrame,
    params_source: str,
    tuned_metric_used: str | None,
    params_source_label: str | None,
    params_source_run_id: str | None,
    params_metric_optimized: str | None,
    params_best_score: float | None,
    params_fingerprint: str | None,
    params_nonempty: bool | None,
    params_run_id: str | None,
    params_market: str | None = None,
) -> dict[str, Any]:
    model_instance = get_model(model_name)()
    identity = resolve_model_identity(model_instance)
    metadata = {
        "model_id": identity["model_id"],
        "model_version": identity["model_version"],
        "prediction_hash": prediction_hash(schedule_df, SCHEDULE_EXPORT_COLUMNS),
        "params": _serialize_params(identity["params"]),
        "params_source": params_source,
        "params_source_label": params_source_label or params_source,
        "params_source_run_id": params_source_run_id or params_run_id,
        "tuned_metric_used": tuned_metric_used,
        "params_metric_optimized": params_metric_optimized,
        "params_best_score": params_best_score,
        "params_fingerprint": params_fingerprint,
        "params_nonempty": params_nonempty,
        "tuning_run_id": params_run_id,
        "params_market": params_market or Market.ML.name,
        "trained_on_date_range": _training_date_range(played),
        "n_games_train": int(len(played)),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    return metadata


def _write_metadata_section(
    writer: pd.ExcelWriter,
    sheet_name: str,
    metadata: dict[str, Any],
) -> int:
    """Write model metadata above the schedule sheet and return the data start row."""
    metadata_df = pd.DataFrame(
        [{"metadata_key": key, "metadata_value": value} for key, value in metadata.items()]
    )
    metadata_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)

    # Always start schedule data at a fixed row below the metadata block.
    return MODEL_METADATA_DATA_START_ROW - 1


def _build_schedule_for_model(
    df: pd.DataFrame,
    *,
    db_path: str | Path,
    sport: str,
    season: str,
    model: str,
    output_path: str | Path | None,
    upcoming_only: bool,
    add_prefix: bool,
    model_params: dict[str, float] | None,
    model_params_file: str | Path | None,
    tuned_metric: str | None,
) -> Path:
    resolution = resolve_model_market_params_with_metadata(
        model,
        params=model_params,
        params_file=model_params_file,
        db_path=db_path,
        sport=sport,
        season=season,
        tuned_metric=tuned_metric,
        market=Market.ML,
    )
    resolved_params = resolution.params
    params_run_id = resolution.source_run_id
    params_source_label = resolution.params_source_label or resolution.params_source
    params_source_run_id = resolution.source_run_id
    params_metric_optimized = resolution.metric_optimized
    params_best_score = resolution.best_score
    params_fingerprint = resolution.params_fingerprint
    params_nonempty = resolution.params_nonempty
    schedule_df = _build_schedule_dataframe(
        df,
        db_path=db_path,
        sport=sport,
        season=season,
        model=model,
        upcoming_only=upcoming_only,
        model_params=resolved_params,
        params_source=resolution.params_source,
        params_source_label=params_source_label,
        params_source_run_id=params_source_run_id,
        tuned_metric_used=resolution.tuned_metric_used,
        params_metric_optimized=params_metric_optimized,
        params_best_score=params_best_score,
        params_fingerprint=params_fingerprint,
        params_nonempty=params_nonempty,
        params_run_id=params_run_id,
        params_market=resolution.market,
    )
    schedule_df = _apply_calibration_to_schedule_df(
        schedule_df,
        sport=sport,
        season=season,
        model=model,
    )
    schedule_df = _order_schedule_export(schedule_df)
    default_path = processed_path_for(sport, season, "schedule_with_projections.csv")
    resolved_output = resolve_output_path(
        output_path,
        default_path=default_path,
        model=model,
        add_prefix=add_prefix,
    )
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    schedule_df.to_csv(resolved_output, index=False)
    return resolved_output


def build_schedule_with_projections(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    division: str | None = None,
    conference: str | None = None,
    model: str | None = None,
    output_path: str | Path | None = None,
    upcoming_only: bool = False,
    model_params: dict[str, float] | None = None,
    model_params_file: str | Path | None = None,
    tuned_metric: str | None = None,
) -> Path | list[Path]:
    """Build a schedule export containing projections for upcoming games."""
    rows = load_games(
        db_path,
        sport=sport,
        season=season,
        division=division,
        conference=conference,
    )
    df = normalize_games(rows)
    if df.empty:
        raise ValueError(f"No games found for sport={sport!r}, season={season!r}")

    models = _resolve_models(model)
    multiple = len(models) > 1
    results = [
        _build_schedule_for_model(
            df.copy(deep=True),
            db_path=db_path,
            sport=sport,
            season=season,
            model=model_name,
            output_path=output_path,
            upcoming_only=upcoming_only,
            add_prefix=multiple,
            model_params=model_params,
            model_params_file=model_params_file,
            tuned_metric=tuned_metric,
        )
        for model_name in models
    ]
    return results[0] if len(results) == 1 else results


def build_schedule_excel_report(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    division: str | None = None,
    conference: str | None = None,
    model: str | None = None,
    output_path: str | Path | None = None,
    upcoming_only: bool = False,
    model_params: dict[str, float] | None = None,
    model_params_file: str | Path | None = None,
    tuned_metric: str | None = None,
    as_of_date: str | date | None = None,
    bets_model: str | None = None,
    strict: bool = False,
) -> Path:
    """Build an Excel workbook with schedule projections (one sheet per model)."""
    rows = load_games(
        db_path,
        sport=sport,
        season=season,
        division=division,
        conference=conference,
    )
    df = normalize_games(rows)
    if df.empty:
        raise ValueError(f"No games found for sport={sport!r}, season={season!r}")

    models = _resolve_models(model)
    bets_model_name = _resolve_bets_model(models, bets_model)
    resolved_as_of_date = _resolve_as_of_date(as_of_date)
    report_path = _resolve_workbook_path(
        output_path,
        sport=sport,
        season=season,
        default_name="schedule_with_projections.xlsx",
    )
    dashboard_rows: list[Dict[str, Any]] = []
    bets_schedule_df: pd.DataFrame | None = None

    ensemble_config = load_ensemble_config(
        sport,
        season,
        available_models=list_models(),
    )
    ensemble_config_meta = (ensemble_config or {}).get("_meta", {})
    market_configs = (ensemble_config or {}).get("markets", {})
    market_config_meta = ensemble_config_meta.get("markets", {})
    config_warnings: list[str] = list(ensemble_config_meta.get("warnings", []) or [])

    def _market_config_value(market_name: str, field: str, default: Any = None) -> Any:
        cfg = market_configs.get(market_name) or {}
        return cfg.get(field, default)

    ensemble_ids = {
        Market.ML.name: _market_config_value(Market.ML.name, "ensemble_id", "ensemble_ml_v1"),
        Market.SPREAD.name: _market_config_value(
            Market.SPREAD.name, "ensemble_id", "ensemble_spread_v1"
        ),
        Market.TOTAL.name: _market_config_value(
            Market.TOTAL.name, "ensemble_id", "ensemble_total_v1"
        ),
    }
    allowed_models_map = {
        Market.ML.name: _market_config_value(Market.ML.name, "models"),
        Market.SPREAD.name: _market_config_value(Market.SPREAD.name, "models"),
        Market.TOTAL.name: _market_config_value(Market.TOTAL.name, "models"),
    }
    market_metrics = {
        Market.ML.name: _market_config_value(Market.ML.name, "metric_slot"),
        Market.SPREAD.name: _market_config_value(Market.SPREAD.name, "metric_slot"),
        Market.TOTAL.name: _market_config_value(Market.TOTAL.name, "metric_slot"),
    }
    config_weights_map = {
        Market.ML.name: _market_config_value(Market.ML.name, "weights"),
        Market.SPREAD.name: _market_config_value(Market.SPREAD.name, "weights"),
        Market.TOTAL.name: _market_config_value(Market.TOTAL.name, "weights"),
    }
    resolved_ensemble_meta: dict[str, dict[str, Any]] = {}

    with pd.ExcelWriter(report_path) as writer:
        for model_name in models:
            model_df = df.copy(deep=True)
            market_frames: list[pd.DataFrame] = []
            market_metadatas: list[tuple[Market, dict[str, Any]]] = []
            params_sources: set[str] = set()
            tuned_metrics: set[str] = set()
            params_run_ids: set[str] = set()

            for market in EXPORT_MARKETS:
                resolution = resolve_model_market_params_with_metadata(
                    model_name,
                    params=model_params,
                    params_file=model_params_file,
                    db_path=db_path,
                    sport=sport,
                    season=season,
                    tuned_metric=tuned_metric,
                    market=market,
                )
                resolved_params = resolution.params
                params_source = resolution.params_source
                tuned_metric_used = resolution.tuned_metric_used
                params_source_label = resolution.params_source_label or params_source
                params_source_run_id = resolution.source_run_id
                params_metric_optimized = resolution.metric_optimized
                params_best_score = resolution.best_score
                params_fingerprint = resolution.params_fingerprint
                params_nonempty = resolution.params_nonempty
                schedule_df = _build_schedule_dataframe(
                    model_df,
                    db_path=db_path,
                    sport=sport,
                    season=season,
                    model=model_name,
                    upcoming_only=upcoming_only,
                    model_params=resolved_params,
                    params_source=params_source,
                    params_source_label=params_source_label,
                    params_source_run_id=params_source_run_id,
                    tuned_metric_used=tuned_metric_used,
                    params_metric_optimized=params_metric_optimized,
                    params_best_score=params_best_score,
                    params_fingerprint=params_fingerprint,
                    params_nonempty=params_nonempty,
                    params_run_id=resolution.source_run_id,
                    params_market=market.name,
                )
                schedule_df = _apply_calibration_to_schedule_df(
                    schedule_df,
                    sport=sport,
                    season=season,
                    model=model_name,
                )
                schedule_df = _order_schedule_export(schedule_df)
                market_frames.append(schedule_df)

                metadata = _build_model_metadata(
                    model_name=model_name,
                    played=_completed_games(model_df),
                    schedule_df=schedule_df,
                    params_source=params_source,
                    params_source_label=params_source_label,
                    params_source_run_id=params_source_run_id,
                    tuned_metric_used=tuned_metric_used,
                    params_metric_optimized=params_metric_optimized,
                    params_best_score=params_best_score,
                    params_fingerprint=params_fingerprint,
                    params_nonempty=params_nonempty,
                    params_run_id=resolution.source_run_id,
                    params_market=market.name,
                )
                market_metadatas.append((market, metadata))
                if params_source_label:
                    params_sources.add(str(params_source_label))
                if tuned_metric_used:
                    tuned_metrics.add(str(tuned_metric_used))
                if params_source_run_id:
                    params_run_ids.add(str(params_source_run_id))

                dashboard_rows.extend(
                    _dashboard_rows_for_today(
                        schedule_df,
                        model_name,
                        resolved_as_of_date,
                        model_metadata=metadata,
                    )
                )
                if model_name == bets_model_name and market == Market.ML:
                    bets_schedule_df = schedule_df

            model_df_all_markets = (
                pd.concat(market_frames, ignore_index=True)
                if market_frames
                else pd.DataFrame(columns=SCHEDULE_EXPORT_COLUMNS)
            )
            if not model_df_all_markets.empty:
                model_df_all_markets = (
                    model_df_all_markets.assign(
                        _sort_dt=pd.to_datetime(model_df_all_markets["date"], errors="coerce"),
                        _market_order=model_df_all_markets["params_market"].map(MARKET_ORDER).fillna(len(MARKET_ORDER)),
                    )
                    .sort_values(
                        ["_sort_dt", "game_id", "_market_order", "away_team", "home_team"]
                    )
                    .drop(columns=["_sort_dt", "_market_order"], errors="ignore")
                )

            params_source_joined = ",".join(sorted(params_sources)) if params_sources else "multi_market"
            params_source_labels = {md.get("params_source_label") for _, md in market_metadatas if md.get("params_source_label")}
            params_metric_opts = {md.get("params_metric_optimized") for _, md in market_metadatas if md.get("params_metric_optimized")}
            params_fingerprints = {md.get("params_fingerprint") for _, md in market_metadatas if md.get("params_fingerprint")}
            params_nonempty_flags = {str(md.get("params_nonempty")) for _, md in market_metadatas if md.get("params_nonempty") is not None}

            combined_metadata = _build_model_metadata(
                model_name=model_name,
                played=_completed_games(model_df),
                schedule_df=model_df_all_markets,
                params_source=params_source_joined,
                params_source_label=",".join(sorted(params_source_labels)) if params_source_labels else params_source_joined,
                params_source_run_id=",".join(sorted(params_run_ids)) if params_run_ids else None,
                tuned_metric_used=",".join(sorted(tuned_metrics)) if tuned_metrics else None,
                params_metric_optimized=",".join(sorted(params_metric_opts)) if params_metric_opts else None,
                params_best_score=None,
                params_fingerprint=",".join(sorted(params_fingerprints)) if params_fingerprints else None,
                params_nonempty=",".join(sorted(params_nonempty_flags)) if params_nonempty_flags else None,
                params_run_id=",".join(sorted(params_run_ids)) if params_run_ids else None,
                params_market="MULTI",
            )
            combined_metadata["export_markets"] = ",".join(market.name for market in EXPORT_MARKETS)
            combined_metadata[
                "note"
            ] = "Multi-market schedule; rows are duplicated per market. See params_market for grouping and ordering."
            for market, metadata in market_metadatas:
                for key in (
                    "params_market",
                    "params_source",
                    "params_source_label",
                    "params_source_run_id",
                    "tuned_metric_used",
                    "params_metric_optimized",
                    "params_best_score",
                    "params_fingerprint",
                    "params_nonempty",
                    "tuning_run_id",
                    "run_timestamp_utc",
                    "prediction_hash",
                ):
                    combined_metadata[f"{market.name}.{key}"] = metadata.get(key)

            start_row = _write_metadata_section(writer, model_name, combined_metadata)
            model_df_all_markets.to_excel(
                writer,
                sheet_name=model_name,
                index=False,
                startrow=start_row,
            )

        dashboard_df = pd.DataFrame(dashboard_rows)
        if not dashboard_df.empty:
            model_order = {name: idx for idx, name in enumerate(models)}
            sort_dt = (
                dashboard_df["_sort_dt"]
                if "_sort_dt" in dashboard_df.columns
                else pd.to_datetime(dashboard_df["date"], errors="coerce")
            )
            sort_game_id = (
                dashboard_df["_sort_game_id"]
                if "_sort_game_id" in dashboard_df.columns
                else pd.NA
            )
            dashboard_df = dashboard_df.assign(
                _model_order=dashboard_df["model"]
                .map(model_order)
                .fillna(len(model_order)),
                _sort_dt=sort_dt,
                _sort_game_id=sort_game_id,
                _market_order=dashboard_df["params_market"].map(MARKET_ORDER).fillna(len(MARKET_ORDER)),
            ).sort_values(
                ["_sort_dt", "_sort_game_id", "_market_order", "_model_order", "model"]
            )
            dashboard_df = dashboard_df.drop(
                columns=["_model_order", "_sort_dt", "_sort_game_id", "_market_order"],
                errors="ignore",
            )
        dashboard_df = dashboard_df.reindex(columns=DASHBOARD_COLUMNS)
        dashboard_df.to_excel(writer, sheet_name="dashboard", index=False)

        # If multiple models were produced, try to build an ML ensemble and apply
        # the combined probabilities to the bets_schedule_df (best-effort).
        ensemble_applied = False
        market_forecast_rows: dict[str, list[dict[str, Any]]] = {
            Market.ML.name: [],
            Market.SPREAD.name: [],
            Market.TOTAL.name: [],
        }
        if bets_schedule_df is not None and len(models) > 1:
            _validate_market_tuning_inputs(
                db_path=db_path,
                sport=sport,
                season=season,
                models=models,
                ensemble_ids=ensemble_ids,
                strict=strict,
            )
            market_forecast_rows = _build_market_forecasts_for_ensembles(
                df,
                db_path=db_path,
                sport=sport,
                season=season,
                models=models,
                as_of_date=resolved_as_of_date,
                allowed_models_by_market=allowed_models_map,
                market_metrics=market_metrics,
            )
        try:
            ml_rows = market_forecast_rows.get(Market.ML.name, [])
            if bets_schedule_df is not None and len(models) > 1 and ml_rows:
                forecast_df = pd.DataFrame(ml_rows)
                if not forecast_df.empty:
                    allowed_models = allowed_models_map.get(Market.ML.name)
                    if allowed_models:
                        forecast_df = forecast_df[forecast_df["model_name"].isin(set(allowed_models))]
                        if forecast_df.empty:
                            config_warnings.append(
                                f"No ML forecasts matched configured models {allowed_models}; ensemble skipped"
                            )
                            raise Exception("No ML forecasts after model filter")
                    weights = config_weights_map.get(Market.ML.name)
                    weight_source = "config" if weights else None
                    if weights is None:
                        weights = get_active_ensemble_market_weights(
                            db_path,
                            sport=sport,
                            season=season,
                            market=Market.ML.name,
                            ensemble_id=ensemble_ids[Market.ML.name],
                        )
                        weight_source = "db" if weights else weight_source
                    if weights is None:
                        weights = load_market_weights(
                            sport,
                            season,
                            Market.ML.name,
                            ensemble_ids[Market.ML.name],
                        )
                        weight_source = "file" if weights else weight_source
                    ensemble = MLWeightedAverageEnsemble(
                        sport,
                        season,
                        ensemble_id=ensemble_ids[Market.ML.name],
                        weights=weights,
                    )
                    used_models = sorted(set(forecast_df.get("model_name", [])))
                    cfg_meta = market_config_meta.get(Market.ML.name, {}) if isinstance(market_config_meta, dict) else {}
                    resolved_ensemble_meta[Market.ML.name] = {
                        "ensemble_id": ensemble.ensemble_id,
                        "metric_slot": market_metrics.get(Market.ML.name),
                        "configured_models": allowed_models,
                        "configured_weights": config_weights_map.get(Market.ML.name),
                        "weights_source": weight_source or "equal",
                        "weights": weights,
                        "used_models": used_models,
                        "config_source": cfg_meta.get("source"),
                        "config_path": cfg_meta.get("path"),
                    }
                    # Load calibrator for the ensemble source, if present
                    try:
                        ensemble_cal = load_latest_calibrator(
                            sport=sport,
                            season=season,
                            model=ensemble.ensemble_id,
                            market=Market.ML,
                        )
                    except Exception:
                        ensemble_cal = None

                    # Apply per-game ensemble
                    for gid in pd.unique(bets_schedule_df["game_id"]):
                        try:
                            subset = forecast_df[forecast_df["game_id"] == gid]
                            if subset.empty:
                                continue
                            raw_p, components_json = ensemble.combine(subset)
                            if raw_p is None:
                                continue
                            final_p = raw_p
                            if ensemble_cal is not None:
                                try:
                                    transformed = ensemble_cal.transform([raw_p])
                                    if transformed is not None and len(transformed) > 0:
                                        final_p = float(transformed[0])
                                except Exception:
                                    pass

                            mask = bets_schedule_df["game_id"] == gid
                            bets_schedule_df.loc[mask, "home_win_prob"] = final_p
                            bets_schedule_df.loc[mask, "away_win_prob"] = 1.0 - final_p

                            def _append_ens(val: Any) -> str:
                                try:
                                    if val is None or (isinstance(val, float) and pd.isna(val)):
                                        return ensemble.ensemble_id
                                    text = str(val).strip()
                                    if not text:
                                        return ensemble.ensemble_id
                                    if ensemble.ensemble_id in text.lower():
                                        return text
                                    return f"{text}+{ensemble.ensemble_id}"
                                except Exception:
                                    return ensemble.ensemble_id

                            if "win_prob_source" in bets_schedule_df.columns:
                                bets_schedule_df.loc[mask, "win_prob_source"] = (
                                    bets_schedule_df.loc[mask, "win_prob_source"].apply(_append_ens)
                                )
                            else:
                                bets_schedule_df.loc[mask, "win_prob_source"] = ensemble.ensemble_id
                            bets_schedule_df.loc[mask, "ml_ensemble_components_json"] = components_json
                            ensemble_applied = True
                        except Exception:
                            # Best-effort per-game; continue on errors.
                            continue
        except Exception:
            # Do not fail report generation for ensemble errors.
            pass

        # Apply a spread ensemble to margin fields (best-effort).
        spread_ensemble_applied = False
        try:
            spread_rows = market_forecast_rows.get(Market.SPREAD.name, [])
            if bets_schedule_df is not None and len(models) > 1 and spread_rows:
                forecast_df = pd.DataFrame(spread_rows)
                if not forecast_df.empty:
                    allowed_models = allowed_models_map.get(Market.SPREAD.name)
                    if allowed_models:
                        forecast_df = forecast_df[forecast_df["model_name"].isin(set(allowed_models))]
                        if forecast_df.empty:
                            config_warnings.append(
                                f"No SPREAD forecasts matched configured models {allowed_models}; ensemble skipped"
                            )
                            raise Exception("No SPREAD forecasts after model filter")
                    weights = config_weights_map.get(Market.SPREAD.name)
                    weight_source = "config" if weights else None
                    if weights is None:
                        weights = get_active_ensemble_market_weights(
                            db_path,
                            sport=sport,
                            season=season,
                            market=Market.SPREAD.name,
                            ensemble_id=ensemble_ids[Market.SPREAD.name],
                        )
                        weight_source = "db" if weights else weight_source
                    if weights is None:
                        weights = load_market_weights(
                            sport,
                            season,
                            Market.SPREAD.name,
                            ensemble_ids[Market.SPREAD.name],
                        )
                        weight_source = "file" if weights else weight_source
                    spread_ensemble = SpreadWeightedAverageEnsemble(
                        sport,
                        season,
                        ensemble_id=ensemble_ids[Market.SPREAD.name],
                        weights=weights,
                    )
                    used_models = sorted(set(forecast_df.get("model_name", [])))
                    cfg_meta = market_config_meta.get(Market.SPREAD.name, {}) if isinstance(market_config_meta, dict) else {}
                    resolved_ensemble_meta[Market.SPREAD.name] = {
                        "ensemble_id": spread_ensemble.ensemble_id,
                        "metric_slot": market_metrics.get(Market.SPREAD.name),
                        "configured_models": allowed_models,
                        "configured_weights": config_weights_map.get(Market.SPREAD.name),
                        "weights_source": weight_source or "equal",
                        "weights": weights,
                        "used_models": used_models,
                        "config_source": cfg_meta.get("source"),
                        "config_path": cfg_meta.get("path"),
                    }
                    for gid in pd.unique(bets_schedule_df["game_id"]):
                        try:
                            subset = forecast_df[forecast_df["game_id"] == gid]
                            if subset.empty:
                                continue
                            margin_mean_raw, margin_sd_raw, components_json = (
                                spread_ensemble.combine(subset)
                            )
                            if margin_mean_raw is None:
                                continue
                            mask = bets_schedule_df["game_id"] == gid
                            bets_schedule_df.loc[mask, "margin_mean"] = margin_mean_raw
                            if margin_sd_raw is not None:
                                bets_schedule_df.loc[mask, "margin_sd"] = margin_sd_raw
                            bets_schedule_df.loc[
                                mask, "spread_source"
                            ] = spread_ensemble.ensemble_id
                            bets_schedule_df.loc[
                                mask, "spread_ensemble_components_json"
                            ] = components_json
                            spread_ensemble_applied = True
                        except Exception:
                            continue
        except Exception:
            pass

        # Apply a total ensemble to total fields (best-effort).
        total_ensemble_applied = False
        try:
            total_rows = market_forecast_rows.get(Market.TOTAL.name, [])
            if bets_schedule_df is not None and len(models) > 1 and total_rows:
                forecast_df = pd.DataFrame(total_rows)
                if not forecast_df.empty:
                    allowed_models = allowed_models_map.get(Market.TOTAL.name)
                    if allowed_models:
                        forecast_df = forecast_df[forecast_df["model_name"].isin(set(allowed_models))]
                        if forecast_df.empty:
                            config_warnings.append(
                                f"No TOTAL forecasts matched configured models {allowed_models}; ensemble skipped"
                            )
                            raise Exception("No TOTAL forecasts after model filter")
                    weights = config_weights_map.get(Market.TOTAL.name)
                    weight_source = "config" if weights else None
                    if weights is None:
                        weights = get_active_ensemble_market_weights(
                            db_path,
                            sport=sport,
                            season=season,
                            market=Market.TOTAL.name,
                            ensemble_id=ensemble_ids[Market.TOTAL.name],
                        )
                        weight_source = "db" if weights else weight_source
                    if weights is None:
                        weights = load_market_weights(
                            sport,
                            season,
                            Market.TOTAL.name,
                            ensemble_ids[Market.TOTAL.name],
                        )
                        weight_source = "file" if weights else weight_source
                    total_ensemble = TotalWeightedAverageEnsemble(
                        sport,
                        season,
                        ensemble_id=ensemble_ids[Market.TOTAL.name],
                        weights=weights,
                    )
                    used_models = sorted(set(forecast_df.get("model_name", [])))
                    cfg_meta = market_config_meta.get(Market.TOTAL.name, {}) if isinstance(market_config_meta, dict) else {}
                    resolved_ensemble_meta[Market.TOTAL.name] = {
                        "ensemble_id": total_ensemble.ensemble_id,
                        "metric_slot": market_metrics.get(Market.TOTAL.name),
                        "configured_models": allowed_models,
                        "configured_weights": config_weights_map.get(Market.TOTAL.name),
                        "weights_source": weight_source or "equal",
                        "weights": weights,
                        "used_models": used_models,
                        "config_source": cfg_meta.get("source"),
                        "config_path": cfg_meta.get("path"),
                    }
                    for gid in pd.unique(bets_schedule_df["game_id"]):
                        try:
                            subset = forecast_df[forecast_df["game_id"] == gid]
                            if subset.empty:
                                continue
                            total_mean_raw, total_sd_raw, components_json = (
                                total_ensemble.combine(subset)
                            )
                            if total_mean_raw is None:
                                continue
                            mask = bets_schedule_df["game_id"] == gid
                            bets_schedule_df.loc[mask, "total"] = total_mean_raw
                            if total_sd_raw is not None:
                                bets_schedule_df.loc[mask, "total_sd"] = total_sd_raw
                            bets_schedule_df.loc[
                                mask, "total_source"
                            ] = total_ensemble.ensemble_id
                            bets_schedule_df.loc[
                                mask, "total_ensemble_components_json"
                            ] = components_json
                            total_ensemble_applied = True
                        except Exception:
                            continue
        except Exception:
            pass

        # If the ensemble was applied successfully to any games, set the BETS model label
        # and ensure META reflects the ensemble as the bets model.
        if ensemble_applied and bets_schedule_df is not None:
            try:
                bets_schedule_df["model"] = ensemble.ensemble_id
                bets_model_name = ensemble.ensemble_id
            except Exception:
                # Best-effort assignment; ignore failures.
                pass

        ml_ensemble_id = ensemble_ids.get(Market.ML.name, "ensemble_ml_v1")
        spread_ensemble_id = ensemble_ids.get(Market.SPREAD.name, "ensemble_spread_v1")
        total_ensemble_id = ensemble_ids.get(Market.TOTAL.name, "ensemble_total_v1")

        spread_source_id = (
            spread_ensemble_id
            if spread_ensemble_applied
            else _first_nonempty_source(
                bets_schedule_df if bets_schedule_df is not None else pd.DataFrame(),
                "spread_source",
                as_of_date=resolved_as_of_date,
            )
            or "direct"
        )
        total_source_id = (
            total_ensemble_id
            if total_ensemble_applied
            else _first_nonempty_source(
                bets_schedule_df if bets_schedule_df is not None else pd.DataFrame(),
                "total_source",
                as_of_date=resolved_as_of_date,
            )
            or "direct"
        )
        ml_source_id = (
            ml_ensemble_id if ensemble_applied else (bets_model_name or "direct")
        )
        review_run_id = _deterministic_review_run_id(
            sport=sport,
            season=season,
            as_of_date=resolved_as_of_date,
            ml_source_id=ml_source_id,
            spread_source_id=spread_source_id,
            total_source_id=total_source_id,
        )
        bets_df = _build_bets_dataframe(
            bets_schedule_df if bets_schedule_df is not None else pd.DataFrame(),
            model_name=bets_model_name,
            as_of_date=resolved_as_of_date,
            review_run_id=review_run_id,
            db_path=db_path,
            sport=sport,
            season=season,
        )
        bets_df.to_excel(writer, sheet_name="BETS", index=False)

        param_sources = _collect_market_param_sources(
            db_path=db_path,
            sport=sport,
            season=season,
            models=models,
        )
        ensemble_sources = _collect_ensemble_weight_sources(
            db_path=db_path,
            sport=sport,
            season=season,
            ensemble_ids=ensemble_ids,
        )
        config_path = next(
            (meta.get("path") for meta in market_config_meta.values() if isinstance(meta, dict) and meta.get("path")),
            None,
        )
        config_sha = next(
            (meta.get("sha256") for meta in market_config_meta.values() if isinstance(meta, dict) and meta.get("sha256")),
            None,
        )
        config_sources_json = json.dumps(market_config_meta, sort_keys=True, default=str)
        meta_rows = [
            {"key": "review_run_id", "value": review_run_id},
            {"key": "sport", "value": sport},
            {"key": "season", "value": season},
            {"key": "as_of_date", "value": resolved_as_of_date.isoformat()},
            {"key": "bets_model", "value": bets_model_name},
            {"key": "workbook_kind", "value": "schedule_with_bets"},
            {"key": "created_at_utc", "value": datetime.now(timezone.utc).isoformat()},
            {
                "key": "ensemble_config_path",
                "value": config_path,
            },
            {
                "key": "ensemble_config_sha256",
                "value": config_sha,
            },
            {
                "key": "ensemble_config_warnings",
                "value": ", ".join(config_warnings) if config_warnings else None,
            },
            {
                "key": "ensemble_config_sources_json",
                "value": config_sources_json,
            },
            {
                "key": "active_model_market_params_source_json",
                "value": json.dumps(param_sources, sort_keys=True),
            },
            {
                "key": "active_ensemble_market_weights_source_json",
                "value": json.dumps(ensemble_sources, sort_keys=True),
            },
            {
                "key": "resolved_ensemble_membership_json",
                "value": json.dumps(resolved_ensemble_meta, sort_keys=True, default=str),
            },
        ]
        meta_df = pd.DataFrame(meta_rows)
        meta_df.to_excel(writer, sheet_name="META", index=False)

    wb = load_workbook(report_path)
    if "BETS" in wb.sheetnames:
        ws = wb["BETS"]
        apply_ev_formulas(ws, use_price=True)
        apply_model_prob_formulas_for_bets_sheet(ws)
        validate_bets_formulas(ws)
        # Apply UX and helper formatting (add-only, best-effort)
        try:
            apply_bets_sheet_formatting(ws)
        except Exception:
            pass
        validate_no_ellipsis_formulas(wb)
        # Format the `stake` column as US dollars (column header: "stake").
        try:
            header = next(ws.iter_rows(min_row=1, max_row=1))
            stake_col = None
            for cell in header:
                if cell.value is not None and str(cell.value).strip().lower() == "stake":
                    stake_col = cell.column_letter
                    break
            if stake_col:
                for r in range(2, ws.max_row + 1):
                    cell = ws[f"{stake_col}{r}"]
                    cell.number_format = FORMAT_CURRENCY_USD_SIMPLE
        except Exception:
            # Best-effort formatting; do not fail the report generation on errors.
            pass

        ws.protection.sheet = False
    if "META" in wb.sheetnames:
        ws = wb["META"]
        ws.sheet_state = "hidden"
    wb.save(report_path)
    return report_path
