from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import json
from pathlib import Path
from typing import Callable, Iterable
import os
import time

import numpy as np
import pandas as pd

from contracts import validate_model_input, validate_predictions
from data.validation import validate_dataset
from models.base import BaseModel, GamePrediction, resolve_model_identity
from pipelines.guardrails import apply_prediction_validation
from pipelines.metadata import prediction_hash
from eval.validation import get_validation_config
from models.calibration import guardrail_margin_sd
from backtest.eval_schema import (
    REQUIRED_ACTUAL_COLS,
    REQUIRED_EVAL_COLUMNS,
    REQUIRED_PRED_COLS,
)

DEFAULT_BUCKET_EDGES = np.linspace(0.0, 1.0, 11)
REQUIRED_BACKTEST_PREDICTION_COLUMNS = [
    "game_id",
    "date",
    "home_team",
    "away_team",
    "p_home_win",
    "win_prob_samples",
    "win_prob_dist",
    "pred_margin",
    "pred_total",
    "margin_mean",
    "margin_sd",
    "total_mean",
    "total_sd",
    "model_win_prob",
    "model_id",
]

logger = logging.getLogger(__name__)


@dataclass
class BacktestOutputs:
    predictions: pd.DataFrame
    metrics_by_date: pd.DataFrame
    metrics_overall: pd.DataFrame
    calibration: pd.DataFrame
    output_dir: Path


@dataclass(frozen=True)
class BacktestSlice:
    date: pd.Timestamp
    train_idx: np.ndarray
    eval_idx: np.ndarray


def load_games_df_from_db(
    db_path: str | Path,
    *,
    sport: str | None = None,
    season: str | None = None,
) -> pd.DataFrame:
    from data.repository import load_games

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. Provide a valid DB path or ingest CSV."
        )

    games = load_games(db_path, sport=sport, season=season)
    if not games:
        filter_label = f"sport={sport}, season={season}"
        raise ValueError(
            "No games found for backtest "
            f"({filter_label}). Ingest historical CSV data into the database "
            "or provide a database with historical games."
        )
    rows = [game.model_dump() for game in games]
    return pd.DataFrame(rows)


def load_games_df_from_csv(
    csv_path: str | Path,
    *,
    sport: str | None = None,
    season: str | None = None,
) -> pd.DataFrame:
    """Load a backtest dataset from CSV, accepting common Sports-Reference headers."""
    from ingest.sports_reference import load_sr_csv_lenient, parse_sr_csv

    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Lowercase/strip headers and map common aliases to expected names."""
        rename_map: dict[str, str] = {}
        normalized = {c.strip().lower(): c for c in df.columns}

        def _find(*names: str) -> str | None:
            for name in names:
                if name in normalized:
                    return normalized[name]
            return None

        date_col = _find("date", "game date")
        away_col = _find("away_team", "visitor/neutral", "visitor", "away", "away/neutral", "road", "road team")
        home_col = _find("home_team", "home/neutral", "home", "home team")
        away_score_col = _find(
            "away_score",
            "pts_away",
            "visitor pts",
            "visitor_pts",
            "away pts",
            "pts_visitor",
            "ptsaway",
            "pts away",
            "visitor g",
            "visitor_g",
            "away g",
            "away_g",
            "g_away",
            "g away",
        )
        home_score_col = _find(
            "home_score",
            "pts_home",
            "home pts",
            "home_pts",
            "ptshome",
            "pts home",
            "home g",
            "home_g",
            "g_home",
            "g home",
        )
        neutral_col = _find("neutral", "neutral_site")
        overtime_col = _find("overtime", "ot")
        game_id_col = _find("game_id", "box score", "boxscore", "box")

        for src, target in [
            (date_col, "date"),
            (away_col, "away_team"),
            (home_col, "home_team"),
            (away_score_col, "away_score"),
            (home_score_col, "home_score"),
            (neutral_col, "neutral"),
            (overtime_col, "overtime"),
            (game_id_col, "game_id"),
        ]:
            if src:
                rename_map[src] = target

        normalized_df = df.rename(columns=rename_map)
        # Promote a numeric neutral/overtime if present; leave validation to downstream checks.
        return normalized_df

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found at {csv_path}. Provide a valid CSV path."
        )
    text = csv_path.read_text(encoding="utf-8", errors="ignore")
    df = load_sr_csv_lenient(text)
    if df.empty:
        raise ValueError(f"No rows found in CSV at {csv_path}.")

    normalized = _normalize_columns(df)
    normalized = _scrub_game_id_column(normalized)
    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    if required.issubset(set(normalized.columns)) and _has_required_data(normalized):
        return normalized

    # Fallback: parse as a Sports-Reference export and convert to a DataFrame the backtester understands.
    parsed = parse_sr_csv(csv_path, sport=sport, season=season)
    if not parsed:
        raise ValueError(
            "CSV is missing required columns for backtesting "
            "(date, home_team, away_team, home_score, away_score), "
            "and parsing as a Sports-Reference export also failed."
        )
    return pd.DataFrame([game.model_dump() for game in parsed])


def _scrub_game_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unusable game_id values (e.g., tip times) so we can rebuild IDs."""
    from ingest.sports_reference import looks_like_tip_time

    if "game_id" not in df.columns:
        return df
    cleaned = df.copy()
    series = cleaned["game_id"].astype(str).str.strip()
    lowered = series.str.lower()
    invalid_mask = lowered.isin({"", "nan", "none"}) | series.map(looks_like_tip_time)
    if invalid_mask.any():
        cleaned.loc[invalid_mask, "game_id"] = pd.NA
    duplicate_mask = cleaned["game_id"].notna() & cleaned["game_id"].duplicated(keep=False)
    if duplicate_mask.any():
        cleaned.loc[duplicate_mask, "game_id"] = pd.NA
    return cleaned


def _has_required_data(df: pd.DataFrame) -> bool:
    """Ensure required columns contain usable values before trusting normalization."""
    required_cols = ["date", "home_team", "away_team"]
    if not all(col in df.columns for col in required_cols):
        return False
    total = len(df)
    if total == 0:
        return False
    for col in required_cols:
        series = df[col]
        non_empty = series.notna() & (series.astype(str).str.strip() != "")
        if non_empty.sum() / total < 0.5:
            return False
    return True


def _precompute_actual_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute actual outcome columns once for the full dataset."""
    result = df.copy()
    index = result.index
    if "home_score" not in result.columns:
        result["home_score"] = pd.Series(pd.NA, index=index, dtype="Float64")
    if "away_score" not in result.columns:
        result["away_score"] = pd.Series(pd.NA, index=index, dtype="Float64")

    home = pd.to_numeric(result.get("home_score"), errors="coerce")
    away = pd.to_numeric(result.get("away_score"), errors="coerce")

    if "actual_margin" not in result.columns:
        result["actual_margin"] = (home - away).astype("Float64")
    else:
        actual_margin = pd.to_numeric(result.get("actual_margin"), errors="coerce")
        missing = actual_margin.isna()
        if missing.any():
            actual_margin.loc[missing] = (home - away)[missing]
        result["actual_margin"] = actual_margin.astype("Float64")

    if "actual_total" not in result.columns:
        result["actual_total"] = (home + away).astype("Float64")
    else:
        actual_total = pd.to_numeric(result.get("actual_total"), errors="coerce")
        missing = actual_total.isna()
        if missing.any():
            actual_total.loc[missing] = (home + away)[missing]
        result["actual_total"] = actual_total.astype("Float64")

    if "home_win" not in result.columns:
        home_win = pd.Series(pd.NA, index=index, dtype="Float64")
    else:
        home_win = pd.to_numeric(result.get("home_win"), errors="coerce")
    missing = home_win.isna()
    score_mask = home.notna() & away.notna() & missing
    if score_mask.any():
        home_win.loc[score_mask] = np.where(
            home[score_mask] > away[score_mask],
            1.0,
            np.where(home[score_mask] < away[score_mask], 0.0, 0.5),
        )
    result["home_win"] = home_win.astype("Float64")
    return result


def _prepare_backtest_slices(
    games: pd.DataFrame,
    *,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    window: str,
    rolling_days: int | None,
    rolling_games: int | None,
) -> list[BacktestSlice]:
    date_values = pd.to_datetime(games["date"], errors="coerce").dt.normalize()
    evaluation_mask = (date_values >= start_dt) & (date_values <= end_dt)
    evaluation_dates = sorted(date_values[evaluation_mask].dropna().unique())
    if not evaluation_dates:
        raise ValueError(
            "No evaluation dates found for backtest window "
            f"{start_dt.date().isoformat()} to {end_dt.date().isoformat()}. "
            "Confirm the --start/--end dates overlap the dataset."
        )

    slices: list[BacktestSlice] = []
    index_values = games.index.to_numpy()
    for current_date in evaluation_dates:
        eval_mask = date_values == current_date
        eval_idx = index_values[eval_mask.to_numpy()]

        train_mask = date_values < current_date
        if window == "rolling" and rolling_days is not None:
            cutoff = pd.Timestamp(current_date) - pd.Timedelta(days=rolling_days)
            train_mask &= date_values >= cutoff
        train_idx = index_values[train_mask.to_numpy()]
        if window == "rolling" and rolling_games is not None and len(train_idx) > rolling_games:
            train_idx = train_idx[-rolling_games:]

        slices.append(
            BacktestSlice(
                date=pd.Timestamp(current_date),
                train_idx=train_idx,
                eval_idx=eval_idx,
            )
        )
    return slices


def run_backtest(
    model_factory: Callable[[], BaseModel],
    games_df: pd.DataFrame,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    window: str = "expanding",
    rolling_days: int | None = None,
    rolling_games: int | None = None,
    output_dir: str | Path | None = None,
    model_name: str | None = None,
    db_path: str | Path | None = None,
    sport: str | None = None,
    season: str | None = None,
    calibrate: bool = False,
    calib_dir: str | Path | None = None,
    calibrator_override: str | None = None,
) -> BacktestOutputs:
    if window not in {"expanding", "rolling"}:
        raise ValueError("window must be 'expanding' or 'rolling'.")
    if window == "rolling" and rolling_days is None and rolling_games is None:
        raise ValueError("Provide rolling_days or rolling_games for rolling backtests.")

    games = validate_dataset(games_df)
    if "neutral" not in games.columns:
        games["neutral"] = False
    games = games.copy()
    games["date"] = pd.to_datetime(games["date"], errors="coerce").dt.normalize()

    start_dt = (
        pd.to_datetime(start_date).normalize() if start_date else games["date"].min()
    )
    end_dt = pd.to_datetime(end_date).normalize() if end_date else games["date"].max()

    precompute_start = time.perf_counter()
    games = _precompute_actual_columns(games)
    precompute_actuals_time = time.perf_counter() - precompute_start

    slice_start = time.perf_counter()
    slices = _prepare_backtest_slices(
        games,
        start_dt=start_dt,
        end_dt=end_dt,
        window=window,
        rolling_days=rolling_days,
        rolling_games=rolling_games,
    )
    slice_prep_time = time.perf_counter() - slice_start

    model_instance = model_factory()
    meta = None
    try:
        meta = model_instance.metadata()
    except Exception:
        meta = None

    if (
        bool(getattr(meta, "supports_streaming_backtest", False))
        and window == "expanding"
        and rolling_days is None
        and rolling_games is None
    ):
        return run_backtest_streaming(
            model_instance=model_instance,
            games=games,
            slices=slices,
            start_dt=start_dt,
            end_dt=end_dt,
            output_dir=output_dir,
            model_name=model_name,
            db_path=db_path,
            sport=sport,
            season=season,
            calibrate=calibrate,
            calib_dir=calib_dir,
            calibrator_override=calibrator_override,
            window=window,
            rolling_days=rolling_days,
            rolling_games=rolling_games,
            precompute_actuals_time=precompute_actuals_time,
            slice_prep_time=slice_prep_time,
        )

    prediction_frames: list[pd.DataFrame] = []
    calibration_eval_rows: list[dict] = []
    model_identity: dict[str, str] | None = None
    # Profiling timers (seconds)
    profile_enabled = os.getenv("TUNE_PROFILE", "0").strip() == "1" or logging.getLogger().isEnabledFor(logging.DEBUG)
    fit_time_total = 0.0
    predict_time_total = 0.0
    pred_to_frame_time_total = 0.0
    merge_time_total = 0.0
    ensure_schema_time_total = 0.0
    for slice_info in slices:
        current_date = slice_info.date
        train_data = games.loc[slice_info.train_idx]
        if train_data.empty:
            continue

        model = model_factory()
        if model_identity is None:
            model_identity = resolve_model_identity(model)
        # Time model.fit
        t0 = time.perf_counter()
        # Centralize as-of date: pass explicit fit_end_date for recency semantics
        fit_end_date = pd.to_datetime(train_data["date"], errors="coerce").dropna().max()
        try:
            model.fit(train_data, fit_end_date=fit_end_date)
        except TypeError:
            # Backward compatibility: some models may not accept fit_end_date
            model.fit(train_data)
        t1 = time.perf_counter()
        fit_time_total += (t1 - t0)

        day_games = games.loc[slice_info.eval_idx]
        predict_input = day_games.drop(
            columns=["home_score", "away_score"], errors="ignore"
        )
        predict_input = validate_model_input(predict_input, context="Backtest model input", sport=sport, season=season)
        t0 = time.perf_counter()
        predictions = model.predict(predict_input)
        t1 = time.perf_counter()
        predict_time_total += (t1 - t0)
        predictions = validate_predictions(predictions, context="Backtest model output")
        _attach_prediction_metadata(predictions, model=model, train_data=train_data)
        t0 = time.perf_counter()
        pred_df = _predictions_to_frame(predictions)
        t1 = time.perf_counter()
        pred_to_frame_time_total += (t1 - t0)
        # Ensure an empty predictions return value yields a DataFrame with the
        # expected columns so downstream normalization and merges don't fail.
        if pred_df.empty:
            pred_df = pd.DataFrame(columns=REQUIRED_BACKTEST_PREDICTION_COLUMNS)
            _log_backtest_issue(model_name, current_date, pred_df, note="empty_predictions")

        # If the frame is missing expected columns, record details for debugging.
        missing = [c for c in (REQUIRED_BACKTEST_PREDICTION_COLUMNS) if c not in pred_df.columns]
        if missing:
            _log_backtest_issue(model_name, current_date, pred_df, note=f"missing_columns:{missing}")

        # Normalize date column for merging; keep wrapped to log unexpected errors.
        try:
            pred_df["date"] = pd.to_datetime(pred_df["date"]).dt.normalize()
        except Exception as exc:
            _log_backtest_issue(model_name, current_date, pred_df, note=f"date_normalize_error:{type(exc).__name__}")
            # Re-raise so upstream behavior remains unchanged for visible failures.
            raise

        _apply_guardrails(pred_df, sport=sport)

        t0 = time.perf_counter()
        merged = day_games.merge(
            pred_df,
            on=["date", "home_team", "away_team"],
            how="left",
            suffixes=("", "_pred"),
        )
        t1 = time.perf_counter()
        merge_time_total += (t1 - t0)
        t0 = time.perf_counter()
        merged = ensure_eval_schema(merged, precomputed_actuals=True)
        t1 = time.perf_counter()
        ensure_schema_time_total += (t1 - t0)

        # Optional calibration: use previously produced out-of-sample predictions
        # (prediction_frames) as training data to fit a calibrator and transform
        # today's predictions. This avoids leakage because training folds are
        # previous test folds in the walk-forward loop.
        _apply_optional_calibration(
            merged,
            prediction_frames,
            calibration_eval_rows=calibration_eval_rows,
            calibrate=calibrate,
            calib_dir=calib_dir,
            calibrator_override=calibrator_override,
            sport=sport,
            model_name=model_name,
            current_date=current_date,
        )

        prediction_frames.append(merged)

    outputs, run_id, apply_validation_time, metrics_agg_time = _finalize_backtest_outputs(
        prediction_frames,
        calibration_eval_rows,
        model_identity=model_identity,
        start_dt=start_dt,
        end_dt=end_dt,
        window=window,
        rolling_days=rolling_days,
        rolling_games=rolling_games,
        output_dir=output_dir,
        model_name=model_name,
        db_path=db_path,
        sport=sport,
        season=season,
    )

    # Print profiling summary when enabled. Use candidate identification when available.
    if profile_enabled:
        model_id = model_identity.get("model_id") if model_identity else (model_name or "model")
        date_range = f"{start_dt.date().isoformat()}_to_{end_dt.date().isoformat()}"
        summary = (
            f"PROFILE model={model_id} range={date_range} "
            f"fit={fit_time_total:.3f}s predict={predict_time_total:.3f}s "
            f"pred_to_frame={pred_to_frame_time_total:.3f}s merge={merge_time_total:.3f}s "
            f"ensure_schema={ensure_schema_time_total:.3f}s apply_validation={apply_validation_time:.3f}s "
            f"metrics_agg={metrics_agg_time:.3f}s precompute_actuals={precompute_actuals_time:.3f}s "
            f"slice_prep={slice_prep_time:.3f}s"
        )
        try:
            print(summary)
        except Exception:
            logger.debug("Failed to print tuning profile summary")

    return outputs


def _finalize_backtest_outputs(
    prediction_frames: list[pd.DataFrame],
    calibration_eval_rows: list[dict],
    *,
    model_identity: dict[str, str] | None,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    window: str,
    rolling_days: int | None,
    rolling_games: int | None,
    output_dir: str | Path | None,
    model_name: str | None,
    db_path: str | Path | None,
    sport: str | None,
    season: str | None,
) -> tuple[BacktestOutputs, str, float, float]:
    def _has_data(frame: pd.DataFrame) -> bool:
        """Return True when the frame has rows, columns, and at least one non-NA value."""
        return (
            isinstance(frame, pd.DataFrame)
            and frame.shape[0] > 0
            and frame.shape[1] > 0
            and frame.notna().any().any()
        )

    valid_frames = []
    for _frame in prediction_frames:
        if not _has_data(_frame):
            continue
        # Preserve required schema columns even when all-NA; allow optional columns
        # to be dropped if completely empty to avoid dtype inference warnings.
        required_cols = set(REQUIRED_BACKTEST_PREDICTION_COLUMNS) | REQUIRED_EVAL_COLUMNS
        keep_columns = _frame.columns.isin(required_cols)
        cleaned = _frame.loc[:, _frame.notna().any(axis=0) | keep_columns]

        # Drop only optional columns that are entirely NA; retain required schema
        # columns even when empty to keep downstream concatenations stable.
        drop_candidates = [
            col
            for col in cleaned.columns
            if col not in required_cols and cleaned[col].isna().all()
        ]
        if drop_candidates:
            cleaned = cleaned.drop(columns=drop_candidates)
        if _has_data(cleaned):
            valid_frames.append(cleaned)
    if valid_frames:
        # Filter out empty/all-NA frames to avoid dtype inference changes in future pandas versions.
        filtered = []
        for frame in valid_frames:
            if not isinstance(frame, pd.DataFrame):
                continue
            non_empty_cols = frame.columns[frame.notna().any(axis=0)]
            if len(non_empty_cols) == 0:
                continue
            filtered.append(frame.loc[:, non_empty_cols])
        if not filtered:
            raise ValueError(
                "Backtest produced no usable prediction frames after cleaning."
            )
        predictions_df = pd.concat(filtered, ignore_index=True)
    else:
        raise ValueError(
            "Backtest produced no predictions. "
            "This happens when each evaluation date has no training data "
            "(games must exist before each evaluation date)."
        )

    for column in REQUIRED_BACKTEST_PREDICTION_COLUMNS:
        if column not in predictions_df.columns:
            predictions_df[column] = pd.NA

    for column in REQUIRED_EVAL_COLUMNS:
        if column not in predictions_df.columns:
            predictions_df[column] = pd.Series(
                pd.NA, index=predictions_df.index, dtype="Float64"
            )

    # Evaluation-only exclusion: validate predictions but keep rows for scoring
    # so outcomes do not change the evaluation set. Extreme corruption may still
    # be dropped by the validator.
    validation_reason_counts: dict[str, int] | None = None
    validation_drop_counts: dict[str, int] | None = None
    try:
        t0 = time.perf_counter()
        eval_df, validation_exclusions = apply_prediction_validation(
            predictions_df,
            sport=sport,
            drop_invalid=False,
            include_reasons=True,
            log_reasons_per_row=False,
        )
        t1 = time.perf_counter()
        apply_validation_time = t1 - t0
    except Exception:
        eval_df = predictions_df
        validation_exclusions = []
        apply_validation_time = 0.0

    if "__invalid_reasons" in eval_df.columns:
        validation_reason_counts = {}
        for reasons in eval_df["__invalid_reasons"]:
            if not reasons:
                continue
            for reason in reasons:
                validation_reason_counts[reason] = (
                    validation_reason_counts.get(reason, 0) + 1
                )

    if validation_exclusions:
        validation_drop_counts = {}
        for _, _, reasons in validation_exclusions:
            for reason in reasons:
                validation_drop_counts[reason] = (
                    validation_drop_counts.get(reason, 0) + 1
                )

    if validation_reason_counts == {}:
        validation_reason_counts = None
    if validation_drop_counts == {}:
        validation_drop_counts = None

    # Emit a single summary of validation counts for backtests. Log at DEBUG
    # when debug is enabled, otherwise emit at INFO when verbose logging is
    # requested. Avoid per-row warnings during bulk backtests.
    if logger.isEnabledFor(logging.DEBUG):
        if validation_reason_counts:
            logger.debug(
                "Backtest validation reasons (kept rows): %s",
                validation_reason_counts,
            )
        if validation_drop_counts:
            logger.debug(
                "Backtest validation drops: %s",
                validation_drop_counts,
            )
    elif logger.isEnabledFor(logging.INFO):
        if validation_reason_counts:
            logger.info(
                "Backtest validation reasons (kept rows): %s",
                validation_reason_counts,
            )
        if validation_drop_counts:
            logger.info(
                "Backtest validation drops: %s",
                validation_drop_counts,
            )

    t0 = time.perf_counter()
    metrics_by_date = _aggregate_metrics_by_date(eval_df)
    metrics_overall = _aggregate_overall_metrics(eval_df)
    t1 = time.perf_counter()
    metrics_agg_time = t1 - t0
    calibration = _calibration_table(eval_df)
    if validation_reason_counts is not None:
        metrics_overall["validation_reason_counts"] = [validation_reason_counts]
    if validation_drop_counts is not None:
        metrics_overall["validation_drop_counts"] = [validation_drop_counts]
    model_id = model_identity.get("model_id") if model_identity else None
    for frame in (metrics_by_date, metrics_overall, calibration):
        frame["model_id"] = model_id

    resolved_model_name = model_name or "model"
    target_dir = (
        Path(output_dir)
        if output_dir
        else Path("outputs/backtests") / resolved_model_name
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    run_id = _build_run_id(start_dt, end_dt, window, rolling_days, rolling_games)
    outputs = BacktestOutputs(
        predictions=predictions_df,
        metrics_by_date=metrics_by_date,
        metrics_overall=metrics_overall,
        calibration=calibration,
        output_dir=target_dir,
    )

    # Persist per-fold calibration evaluation rows (if any) to outputs/calibrators
    try:
        if calibration_eval_rows:
            calib_df = pd.DataFrame(calibration_eval_rows)
            # Determine base path for calibrator eval outputs
            if sport and season:
                calib_out_dir = (
                    Path("outputs/calibrators")
                    / sport
                    / season
                    / (model_name or "model")
                )
            elif sport:
                calib_out_dir = (
                    Path("outputs/calibrators") / sport / (model_name or "model")
                )
            else:
                calib_out_dir = Path("outputs/calibrators") / (model_name or "model")
            calib_out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = calib_out_dir / f"calibration_eval_{run_id}.csv"
            json_path = calib_out_dir / f"calibration_eval_{run_id}.json"
            calib_df.to_csv(csv_path, index=False)
            json_path.write_text(
                calib_df.to_json(orient="records", indent=2), encoding="utf-8"
            )
    except Exception:
        # best-effort persistence; do not fail the backtest on I/O errors
        pass

    export_backtest_outputs(outputs, run_id=run_id)
    _persist_backtest_metrics(
        outputs,
        run_id=run_id,
        db_path=db_path,
        sport=sport,
        season=season,
        model_name=model_name,
    )
    return outputs, run_id, apply_validation_time, metrics_agg_time


def _supports_streaming_backtest(model: BaseModel) -> bool:
    try:
        meta = model.metadata()
    except Exception:
        meta = None
    return bool(
        getattr(meta, "supports_streaming_backtest", False)
        and getattr(model, "supports_incremental_update", False)
        and hasattr(model, "predict_one")
        and hasattr(model, "update_with_result")
    )


def _apply_guardrails(pred_df: pd.DataFrame, *, sport: str | None) -> None:
    # Apply sport-specific guardrails to predicted SDs so downstream
    # evaluation and calibration see clamped, sensible variances.
    try:
        cfg = get_validation_config(sport)
        if "margin_sd" in pred_df.columns:
            pred_df["margin_sd"] = pred_df["margin_sd"].apply(
                lambda v: guardrail_margin_sd(
                    float(v) if v is not None and not pd.isna(v) else None,
                    guardrail_min=cfg.margin_sd_min,
                    guardrail_max=cfg.margin_sd_max,
                )[0]
            )
        if "total_mean" in pred_df.columns and "total_sd" in pred_df.columns:
            pred_df["total_sd"] = pred_df["total_sd"].apply(
                lambda v: guardrail_margin_sd(
                    float(v) if v is not None and not pd.isna(v) else None,
                    guardrail_min=cfg.total_sd_min,
                    guardrail_max=cfg.total_sd_max,
                )[0]
            )
    except Exception:
        # Best-effort; do not fail backtests on guardrail application errors.
        pass


def _apply_optional_calibration(
    merged: pd.DataFrame,
    prediction_frames: list[pd.DataFrame],
    *,
    calibration_eval_rows: list[dict],
    calibrate: bool,
    calib_dir: str | Path | None,
    calibrator_override: str | None,
    sport: str | None,
    model_name: str | None,
    current_date: object,
) -> None:
    if not calibrate:
        return
    try:
        from src.calibration.registry import get_calibrator
        from src.calibration.platt import PlattScalingCalibrator
        from src.calibration.isotonic import IsotonicCalibrator
        from src.calibration.eval import brier_score, log_loss
        import time

        past = (
            pd.concat(prediction_frames, ignore_index=True)
            if prediction_frames
            else pd.DataFrame()
        )
        # Build training set: past out-of-sample predictions with p_home_win and home_win
        if not past.empty and "p_home_win" in past.columns and "home_win" in past.columns:
            train_df = past.dropna(subset=["p_home_win", "home_win"]).copy()
        else:
            train_df = pd.DataFrame()

        calib = None
        method = None
        calibration_id = None
        if not train_df.empty:
            n_train = len(train_df)
            # Allow explicit override from CLI: 'platt', 'isotonic', or 'auto'
            override = (calibrator_override or "").strip().lower()
            if override in {"platt", "isotonic"}:
                if override == "isotonic":
                    calib = IsotonicCalibrator()
                    method = "isotonic"
                else:
                    calib = PlattScalingCalibrator()
                    method = "platt"
            else:
                # check registry for a registered calibrator for this sport/model
                try:
                    reg = get_calibrator(sport or "", model_name or "", "ML")
                except Exception:
                    reg = None
                if reg:
                    # registry entries may be classes or factory callables
                    if callable(reg):
                        calib = reg() if isinstance(reg, type) else reg()
                    else:
                        calib = reg
                    method = getattr(calib, "metadata", {}).get("method", "registered")
                else:
                    # default threshold: use isotonic when n >= 500, else Platt
                    N_THRESH = 500
                    if n_train >= N_THRESH:
                        calib = IsotonicCalibrator()
                        method = "isotonic"
                    else:
                        calib = PlattScalingCalibrator()
                        method = "platt"
            calib.fit(
                train_df.rename(
                    columns={"p_home_win": "p_home_win", "home_win": "home_win"}
                )
            )
            # generate id and persist if requested
            ts = int(time.time())
            model_name = model_name or "model"
            calibration_id = f"{model_name}-ML-{ts}"
            if calib_dir:
                p_out = Path(calib_dir) / model_name
                p_out.mkdir(parents=True, exist_ok=True)
                calib_path = p_out / f"{calibration_id}.joblib"
                calib.save(calib_path)

            # transform today's predictions if available
            if "p_home_win" in merged.columns:
                merged["p_home_win_calibrated"] = (
                    calib.transform(merged["p_home_win"])
                    if not merged["p_home_win"].isna().all()
                    else pd.NA
                )
                merged["calibration_id"] = calibration_id
                merged["calibration_method"] = method

                # evaluation on this day's test fold
                test_mask = merged["p_home_win"].notna() & merged["home_win"].notna()
                if test_mask.any():
                    raw_brier = brier_score(
                        merged.loc[test_mask, "p_home_win"],
                        merged.loc[test_mask, "home_win"],
                    )
                    cal_brier = brier_score(
                        merged.loc[test_mask, "p_home_win_calibrated"],
                        merged.loc[test_mask, "home_win"],
                    )
                    raw_ll = log_loss(
                        merged.loc[test_mask, "p_home_win"],
                        merged.loc[test_mask, "home_win"],
                    )
                    cal_ll = log_loss(
                        merged.loc[test_mask, "p_home_win_calibrated"],
                        merged.loc[test_mask, "home_win"],
                    )
                    calibration_eval_rows.append(
                        {
                            "date": current_date,
                            "n_test": int(test_mask.sum()),
                            "method": method,
                            "calibration_id": calibration_id,
                            "brier_raw": raw_brier,
                            "brier_calibrated": cal_brier,
                            "logloss_raw": raw_ll,
                            "logloss_calibrated": cal_ll,
                        }
                    )
        else:
            if "p_home_win" in merged.columns:
                merged["p_home_win_calibrated"] = merged["p_home_win"]
                merged["calibration_id"] = pd.NA
                merged["calibration_method"] = "identity"
    except Exception:
        # best-effort: if calibration fails, continue without it
        pass


def run_backtest_streaming(
    *,
    model_instance: BaseModel,
    games: pd.DataFrame,
    slices: list[BacktestSlice],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    output_dir: str | Path | None,
    model_name: str | None,
    db_path: str | Path | None,
    sport: str | None,
    season: str | None,
    calibrate: bool,
    calib_dir: str | Path | None,
    calibrator_override: str | None,
    window: str,
    rolling_days: int | None,
    rolling_games: int | None,
    precompute_actuals_time: float,
    slice_prep_time: float,
) -> BacktestOutputs:
    # Guard: only models that explicitly opt-in to streaming should be run here.
    try:
        meta = model_instance.metadata()
    except Exception:
        meta = None
    if not bool(getattr(meta, "supports_streaming_backtest", False)):
        raise ValueError(
            "Model does not support streaming backtest. This runner should not be invoked."
        )
    # Validate required streaming API methods exist on the instance.
    if not hasattr(model_instance, "predict_one") or not hasattr(
        model_instance, "update_with_result"
    ):
        raise ValueError(
            "Model instance is missing required streaming methods: predict_one and/or update_with_result."
        )
    prediction_frames: list[pd.DataFrame] = []
    calibration_eval_rows: list[dict] = []
    model_identity = resolve_model_identity(model_instance)

    profile_enabled = os.getenv("TUNE_PROFILE", "0").strip() == "1" or logging.getLogger().isEnabledFor(logging.DEBUG)
    fit_time_total = 0.0
    predict_time_total = 0.0
    pred_to_frame_time_total = 0.0
    merge_time_total = 0.0
    ensure_schema_time_total = 0.0
    update_time_total = 0.0
    refit_time_total = 0.0

    refit_days = os.getenv("ELO_STREAM_REFIT_DAYS")
    refit_games = os.getenv("ELO_STREAM_REFIT_GAMES")
    refit_days_val = int(refit_days) if refit_days and refit_days.isdigit() else None
    refit_games_val = int(refit_games) if refit_games and refit_games.isdigit() else None

    # Initial fit on first available training slice.
    first_train_slice = next((s for s in slices if len(s.train_idx) > 0), None)
    if first_train_slice is None:
        raise ValueError(
            "Backtest produced no predictions. "
            "This happens when each evaluation date has no training data "
            "(games must exist before each evaluation date)."
        )
    initial_train = games.loc[first_train_slice.train_idx]
    t0 = time.perf_counter()
    model_instance.fit(initial_train)
    t1 = time.perf_counter()
    fit_time_total += (t1 - t0)
    last_refit_date = first_train_slice.date
    last_refit_games = int(len(first_train_slice.train_idx))

    for slice_info in slices:
        current_date = slice_info.date
        train_data = games.loc[slice_info.train_idx]
        if train_data.empty:
            continue

        if refit_days_val is not None:
            if (pd.Timestamp(current_date) - pd.Timestamp(last_refit_date)).days >= refit_days_val:
                t0 = time.perf_counter()
                model_instance.fit(train_data)
                t1 = time.perf_counter()
                refit_time_total += (t1 - t0)
                last_refit_date = current_date
                last_refit_games = int(len(slice_info.train_idx))
        elif refit_games_val is not None:
            if int(len(slice_info.train_idx)) - last_refit_games >= refit_games_val:
                t0 = time.perf_counter()
                model_instance.fit(train_data)
                t1 = time.perf_counter()
                refit_time_total += (t1 - t0)
                last_refit_date = current_date
                last_refit_games = int(len(slice_info.train_idx))

        day_games = games.loc[slice_info.eval_idx]
        predict_input = day_games.drop(
            columns=["home_score", "away_score"], errors="ignore"
        )
        predict_input = validate_model_input(predict_input, context="Backtest model input", sport=sport, season=season)
        t0 = time.perf_counter()
        predictions = model_instance.predict(predict_input)
        t1 = time.perf_counter()
        predict_time_total += (t1 - t0)
        predictions = validate_predictions(predictions, context="Backtest model output")
        _attach_prediction_metadata(predictions, model=model_instance, train_data=train_data)
        t0 = time.perf_counter()
        pred_df = _predictions_to_frame(predictions)
        t1 = time.perf_counter()
        pred_to_frame_time_total += (t1 - t0)

        if pred_df.empty:
            pred_df = pd.DataFrame(columns=REQUIRED_BACKTEST_PREDICTION_COLUMNS)
            _log_backtest_issue(model_name, current_date, pred_df, note="empty_predictions")

        missing = [c for c in (REQUIRED_BACKTEST_PREDICTION_COLUMNS) if c not in pred_df.columns]
        if missing:
            _log_backtest_issue(model_name, current_date, pred_df, note=f"missing_columns:{missing}")

        try:
            pred_df["date"] = pd.to_datetime(pred_df["date"]).dt.normalize()
        except Exception as exc:
            _log_backtest_issue(model_name, current_date, pred_df, note=f"date_normalize_error:{type(exc).__name__}")
            raise

        _apply_guardrails(pred_df, sport=sport)

        t0 = time.perf_counter()
        merged = day_games.merge(
            pred_df,
            on=["date", "home_team", "away_team"],
            how="left",
            suffixes=("", "_pred"),
        )
        t1 = time.perf_counter()
        merge_time_total += (t1 - t0)
        t0 = time.perf_counter()
        merged = ensure_eval_schema(merged, precomputed_actuals=True)
        t1 = time.perf_counter()
        ensure_schema_time_total += (t1 - t0)

        _apply_optional_calibration(
            merged,
            prediction_frames,
            calibration_eval_rows=calibration_eval_rows,
            calibrate=calibrate,
            calib_dir=calib_dir,
            calibrator_override=calibrator_override,
            sport=sport,
            model_name=model_name,
            current_date=current_date,
        )

        prediction_frames.append(merged)

        # Update model state after all predictions for the date.
        t0 = time.perf_counter()
        for row in day_games.to_dict(orient="records"):
            model_instance.update_with_result(row)
        t1 = time.perf_counter()
        update_time_total += (t1 - t0)

    outputs, run_id, apply_validation_time, metrics_agg_time = _finalize_backtest_outputs(
        prediction_frames,
        calibration_eval_rows,
        model_identity=model_identity,
        start_dt=start_dt,
        end_dt=end_dt,
        window=window,
        rolling_days=rolling_days,
        rolling_games=rolling_games,
        output_dir=output_dir,
        model_name=model_name,
        db_path=db_path,
        sport=sport,
        season=season,
    )

    if profile_enabled:
        model_id = model_identity.get("model_id") if model_identity else (model_name or "model")
        date_range = f"{start_dt.date().isoformat()}_to_{end_dt.date().isoformat()}"
        summary = (
            f"PROFILE model={model_id} range={date_range} streaming=1 "
            f"fit={fit_time_total:.3f}s refit={refit_time_total:.3f}s "
            f"predict={predict_time_total:.3f}s update={update_time_total:.3f}s "
            f"pred_to_frame={pred_to_frame_time_total:.3f}s merge={merge_time_total:.3f}s "
            f"ensure_schema={ensure_schema_time_total:.3f}s apply_validation={apply_validation_time:.3f}s "
            f"metrics_agg={metrics_agg_time:.3f}s precompute_actuals={precompute_actuals_time:.3f}s "
            f"slice_prep={slice_prep_time:.3f}s"
        )
        try:
            print(summary)
        except Exception:
            logger.debug("Failed to print tuning profile summary")

    return outputs


def export_backtest_outputs(outputs: BacktestOutputs, *, run_id: str) -> None:
    from backtest.export import export_backtest_outputs_excel

    if not outputs.predictions.empty:
        outputs.predictions.to_csv(
            outputs.output_dir / f"predictions_{run_id}.csv",
            index=False,
        )
    outputs.metrics_by_date.to_csv(
        outputs.output_dir / f"metrics_by_date_{run_id}.csv",
        index=False,
    )
    outputs.metrics_overall.to_csv(
        outputs.output_dir / f"metrics_overall_{run_id}.csv",
        index=False,
    )
    outputs.calibration.to_csv(
        outputs.output_dir / f"calibration_{run_id}.csv",
        index=False,
    )
    export_backtest_outputs_excel(outputs, run_id=run_id)


def _persist_backtest_metrics(
    outputs: BacktestOutputs,
    *,
    run_id: str,
    db_path: str | Path | None,
    sport: str | None,
    season: str | None,
    model_name: str | None,
) -> None:
    if db_path is None or sport is None or season is None or model_name is None:
        return
    from data.repository import save_backtest_metrics

    metrics = (
        outputs.metrics_overall.iloc[0].to_dict()
        if not outputs.metrics_overall.empty
        else {}
    )
    log_loss = metrics.get("log_loss")
    brier_score = metrics.get("brier_score")
    mae_margin = metrics.get("mae_margin")
    mae_total = metrics.get("mae_total")
    win_prob_k = _extract_backtest_win_prob_k(outputs.predictions)

    save_backtest_metrics(
        db_path,
        sport=sport,
        season=season,
        model=model_name,
        log_loss=log_loss,
        brier_score=brier_score,
        mae_margin=mae_margin,
        mae_total=mae_total,
        win_prob_k=win_prob_k,
        run_id=run_id,
    )


def _build_run_id(
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    window: str,
    rolling_days: int | None,
    rolling_games: int | None,
) -> str:
    start_label = start_dt.date().isoformat()
    end_label = end_dt.date().isoformat()
    if window == "expanding":
        return f"{start_label}_to_{end_label}_expanding"
    details: list[str] = []
    if rolling_days is not None:
        details.append(f"{rolling_days}d")
    if rolling_games is not None:
        details.append(f"{rolling_games}g")
    detail_label = "_".join(details) if details else "rolling"
    return f"{start_label}_to_{end_label}_rolling_{detail_label}"


def _predictions_to_frame(predictions: Iterable[GamePrediction]) -> pd.DataFrame:
    rows = []
    for pred in predictions:
        row = {
            "game_id": pred.game_id,
            "date": pred.date,
            "home_team": pred.home_team,
            "away_team": pred.away_team,
            "p_home_win": pred.p_home_win,
            "win_prob_samples": (
                json.dumps(pred.win_prob_samples)
                if pred.win_prob_samples is not None
                else None
            ),
            "win_prob_dist": (
                json.dumps(pred.win_prob_samples)
                if pred.win_prob_samples is not None
                else None
            ),
            "pred_margin": pred.pred_margin,
            "pred_total": pred.pred_total,
            "margin_mean": pred.margin_mean,
            "margin_sd": pred.margin_sd,
            "total_mean": pred.total_mean,
            "total_sd": pred.total_sd,
            "model_win_prob": getattr(pred, "model_win_prob", None),
            "model_id": pred.metadata.get("model_id"),
        }
        if pred.extra:
            row["extra"] = pred.extra
            # If the model placed projection fields inside `extra` (dict),
            # promote common projection keys to top-level columns so downstream
            # consumers and CSV exports have consistent fields.
            if isinstance(pred.extra, dict):
                for key in (
                    "projected_home_score",
                    "projected_away_score",
                    "projected_total",
                    "projected_spread",
                    "model_p_home_win",
                    "normal_p_home_win",
                    "win_prob_source",
                    "margin_dist_assumption",
                    "margin_mean",
                    "margin_sd",
                    "total_mean",
                    "total_sd",
                ):
                    if key not in row or row.get(key) is None:
                        val = pred.extra.get(key)
                        if val is not None:
                            row[key] = val
        rows.append(row)
    return pd.DataFrame(rows)


def ensure_eval_schema(df: pd.DataFrame, *, precomputed_actuals: bool = False) -> pd.DataFrame:
    """Ensure evaluation frames contain required columns with safe defaults.

    Existing non-null values are preserved; derived values are only populated
    where missing to avoid overwriting model outputs or precomputed outcomes.
    """
    if df is None:
        return pd.DataFrame(
            {col: pd.Series(dtype="Float64") for col in sorted(REQUIRED_EVAL_COLUMNS)}
        )

    result = df.copy()
    index = result.index

    # Ensure actual-side columns exist before deriving margins/wins.
    for col in REQUIRED_ACTUAL_COLS:
        if col not in result.columns:
            result[col] = pd.Series(pd.NA, index=index, dtype="Float64")

    if not precomputed_actuals:
        home = pd.to_numeric(result.get("home_score"), errors="coerce")
        away = pd.to_numeric(result.get("away_score"), errors="coerce")

        actual_margin = pd.to_numeric(result.get("actual_margin"), errors="coerce")
        margin_missing = actual_margin.isna()
        computed_margin = home - away
        actual_margin.loc[margin_missing] = computed_margin[margin_missing]
        result["actual_margin"] = actual_margin.astype("Float64")

        actual_total = pd.to_numeric(result.get("actual_total"), errors="coerce")
        total_missing = actual_total.isna()
        computed_total = home + away
        actual_total.loc[total_missing] = computed_total[total_missing]
        result["actual_total"] = actual_total.astype("Float64")

        home_win = pd.to_numeric(result.get("home_win"), errors="coerce")
        home_win_missing = home_win.isna()
        score_mask = home.notna() & away.notna() & home_win_missing
        if score_mask.any():
            home_win.loc[score_mask] = np.where(
                home[score_mask] > away[score_mask],
                1.0,
                np.where(home[score_mask] < away[score_mask], 0.0, 0.5),
            )
        result["home_win"] = home_win.astype("Float64")

    # Ensure prediction-side columns exist without overwriting existing values.
    for col in REQUIRED_PRED_COLS:
        if col not in result.columns:
            result[col] = pd.Series(pd.NA, index=index, dtype="Float64")

    return result


def _log_backtest_issue(model_name: str | None, current_date: object, pred_df: pd.DataFrame, note: str | None = None) -> None:
    try:
        out_dir = Path("outputs") / "logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "backtest_pred_issues.log"
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "date": str(current_date),
            "note": note,
            "n_rows": int(len(pred_df)) if isinstance(pred_df, pd.DataFrame) else None,
            "columns": list(pred_df.columns) if isinstance(pred_df, pd.DataFrame) else None,
        }
        with out_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        # Best-effort logging only; do not raise from instrumentation.
        pass


def _prediction_hash_columns(pred_df: pd.DataFrame) -> list[str]:
    columns = [
        "game_id",
        "date",
        "home_team",
        "away_team",
        "p_home_win",
        "win_prob_samples",
        "win_prob_dist",
        "pred_margin",
        "pred_total",
        "margin_mean",
        "margin_sd",
        "total_mean",
        "total_sd",
        "model_win_prob",
    ]
    if "extra" in pred_df.columns:
        columns.append("extra")
    return columns


def _extract_backtest_win_prob_k(predictions_df: pd.DataFrame) -> float | None:
    if predictions_df.empty or "extra" not in predictions_df.columns:
        return None
    values: list[float] = []
    for extra in predictions_df["extra"]:
        if isinstance(extra, dict):
            value = extra.get("win_prob_k")
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if value > 0:
                values.append(value)
    if not values:
        return None
    return float(np.median(values))


def _training_date_range(train_data: pd.DataFrame) -> str:
    if train_data.empty or "date" not in train_data.columns:
        return ""
    dates = pd.to_datetime(train_data["date"], errors="coerce").dropna()
    if dates.empty:
        return ""
    return f"{dates.min().date().isoformat()} to {dates.max().date().isoformat()}"


def _attach_prediction_metadata(
    predictions: list[GamePrediction],
    *,
    model: BaseModel,
    train_data: pd.DataFrame,
) -> None:
    if not predictions:
        return
    model_identity = resolve_model_identity(model)
    trained_on_date_range = _training_date_range(train_data)
    n_games_train = int(len(train_data))
    run_timestamp_utc = datetime.now(timezone.utc).isoformat()

    pred_df = _predictions_to_frame(predictions)
    pred_hash = prediction_hash(pred_df, _prediction_hash_columns(pred_df))

    metadata = {
        **model_identity,
        "trained_on_date_range": trained_on_date_range,
        "n_games_train": n_games_train,
        "run_timestamp_utc": run_timestamp_utc,
        "prediction_hash": pred_hash,
    }
    for prediction in predictions:
        prediction.metadata.update(metadata)


def _home_win_flag(df: pd.DataFrame) -> pd.Series:
    return np.where(
        df["home_score"] > df["away_score"],
        1.0,
        np.where(df["home_score"] < df["away_score"], 0.0, 0.5),
    )


def _aggregate_metrics_by_date(predictions_df: pd.DataFrame) -> pd.DataFrame:
    if predictions_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "games",
                "ml_games",
                "log_loss",
                "brier_score",
                "mae_margin",
                "mae_total",
                "margin_games",
                "total_games",
                "calibration_intercept",
                "calibration_slope",
                "ece",
            ]
        )
    metrics = (
        predictions_df.groupby("date")
        .apply(lambda group: pd.Series(_compute_metrics(group)), include_groups=False)
        .reset_index()
    )
    return metrics


def _aggregate_overall_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    metrics = _compute_metrics(predictions_df) if not predictions_df.empty else {}
    return pd.DataFrame([metrics])


def _compute_metrics(df: pd.DataFrame) -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {
        "games": int(len(df)),
        "ml_games": 0,
        "log_loss": None,
        "brier_score": None,
        "mae_margin": None,
        "mae_total": None,
        "margin_games": 0,
        "total_games": 0,
        "calibration_intercept": None,
        "calibration_slope": None,
        "ece": None,
    }

    prob_df = (
        df.dropna(subset=["p_home_win", "home_win"])
        if {"p_home_win", "home_win"}.issubset(df.columns)
        else pd.DataFrame()
    )
    metrics["ml_games"] = int(len(prob_df))
    if metrics["ml_games"] > 0:
        probs = np.clip(prob_df["p_home_win"].astype(float), 1e-6, 1 - 1e-6)
        actuals = prob_df["home_win"].astype(float)
        metrics["log_loss"] = float(
            -np.mean(actuals * np.log(probs) + (1 - actuals) * np.log(1 - probs))
        )
        metrics["brier_score"] = float(np.mean((probs - actuals) ** 2))
        design = np.column_stack([np.ones(len(probs)), probs])
        coeffs, *_ = np.linalg.lstsq(design, actuals, rcond=None)
        metrics["calibration_intercept"] = float(coeffs[0])
        metrics["calibration_slope"] = float(coeffs[1])
        metrics["ece"] = float(_expected_calibration_error(probs, actuals))

    margin_df = (
        df.dropna(subset=["pred_margin", "actual_margin"])
        if {"pred_margin", "actual_margin"}.issubset(df.columns)
        else pd.DataFrame()
    )
    metrics["margin_games"] = int(len(margin_df))
    if metrics["margin_games"] > 0:
        metrics["mae_margin"] = float(
            np.mean(np.abs(margin_df["pred_margin"] - margin_df["actual_margin"]))
        )

    total_df = (
        df.dropna(subset=["pred_total", "actual_total"])
        if {"pred_total", "actual_total"}.issubset(df.columns)
        else pd.DataFrame()
    )
    metrics["total_games"] = int(len(total_df))
    if metrics["total_games"] > 0:
        actual_total = total_df["actual_total"]
        metrics["mae_total"] = float(
            np.mean(np.abs(total_df["pred_total"] - actual_total))
        )

    # Explicitly return NaN for metrics with zero scorable games to keep
    # downstream summaries informative and avoid confusing None/0 mixes.
    for key in ("log_loss", "brier_score", "calibration_intercept", "calibration_slope", "ece"):
        if metrics[key] is None and metrics["ml_games"] == 0:
            metrics[key] = np.nan
    if metrics["mae_margin"] is None and metrics["margin_games"] == 0:
        metrics["mae_margin"] = np.nan
    if metrics["mae_total"] is None and metrics["total_games"] == 0:
        metrics["mae_total"] = np.nan

    return metrics


def _expected_calibration_error(
    probs: pd.Series | np.ndarray,
    actuals: pd.Series | np.ndarray,
    *,
    bins: int = 10,
) -> float:
    if bins <= 0:
        raise ValueError("bins must be positive.")
    probs_arr = np.asarray(probs, dtype=float)
    actuals_arr = np.asarray(actuals, dtype=float)
    if probs_arr.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(probs_arr, edges, right=True) - 1
    total = probs_arr.size
    ece = 0.0
    for idx in range(bins):
        mask = bin_ids == idx
        if not np.any(mask):
            continue
        avg_pred = float(np.mean(probs_arr[mask]))
        avg_actual = float(np.mean(actuals_arr[mask]))
        ece += (np.sum(mask) / total) * abs(avg_pred - avg_actual)
    return float(ece)


def _calibration_table(predictions_df: pd.DataFrame) -> pd.DataFrame:
    if predictions_df.empty:
        return pd.DataFrame(columns=["bucket", "count", "avg_pred", "avg_actual"])

    if not {"p_home_win", "home_win"}.issubset(predictions_df.columns):
        return pd.DataFrame(columns=["bucket", "count", "avg_pred", "avg_actual"])

    prob_df = predictions_df.dropna(subset=["p_home_win", "home_win"]).copy()
    prob_df["bucket"] = pd.cut(
        prob_df["p_home_win"].clip(0.0, 1.0),
        bins=DEFAULT_BUCKET_EDGES,
        include_lowest=True,
        right=True,
    )

    summary = (
        prob_df.groupby("bucket", observed=True)
        .agg(
            count=("p_home_win", "size"),
            avg_pred=("p_home_win", "mean"),
            avg_actual=("home_win", "mean"),
        )
        .reset_index()
    )
    summary["bucket"] = summary["bucket"].astype(str)
    return summary
