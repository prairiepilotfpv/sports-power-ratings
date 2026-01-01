from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from contracts import validate_model_input, validate_predictions
from data.validation import validate_dataset
from models.base import BaseModel, GamePrediction, resolve_model_identity
from pipelines.metadata import prediction_hash

DEFAULT_BUCKET_EDGES = np.linspace(0.0, 1.0, 11)


@dataclass
class BacktestOutputs:
    predictions: pd.DataFrame
    metrics_by_date: pd.DataFrame
    metrics_overall: pd.DataFrame
    calibration: pd.DataFrame
    output_dir: Path


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


def load_games_df_from_csv(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found at {csv_path}. Provide a valid CSV path."
        )
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No rows found in CSV at {csv_path}.")
    return df


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
) -> BacktestOutputs:
    if window not in {"expanding", "rolling"}:
        raise ValueError("window must be 'expanding' or 'rolling'.")
    if window == "rolling" and rolling_days is None and rolling_games is None:
        raise ValueError("Provide rolling_days or rolling_games for rolling backtests.")

    games = validate_dataset(games_df)
    if "neutral" not in games.columns:
        games["neutral"] = False

    start_dt = (
        pd.to_datetime(start_date).normalize() if start_date else games["date"].min()
    )
    end_dt = pd.to_datetime(end_date).normalize() if end_date else games["date"].max()

    evaluation = games[(games["date"] >= start_dt) & (games["date"] <= end_dt)]
    evaluation_dates = sorted(evaluation["date"].unique())
    if not evaluation_dates:
        raise ValueError(
            "No evaluation dates found for backtest window "
            f"{start_dt.date().isoformat()} to {end_dt.date().isoformat()}. "
            "Confirm the --start/--end dates overlap the dataset."
        )

    prediction_frames: list[pd.DataFrame] = []
    model_identity: dict[str, str] | None = None
    for current_date in evaluation_dates:
        train_data = games[games["date"] < current_date]
        if window == "rolling":
            if rolling_days is not None:
                cutoff = current_date - pd.Timedelta(days=rolling_days)
                train_data = train_data[train_data["date"] >= cutoff]
            if rolling_games is not None:
                train_data = train_data.tail(rolling_games)

        if train_data.empty:
            continue

        model = model_factory()
        if model_identity is None:
            model_identity = resolve_model_identity(model)
        model.fit(train_data)

        day_games = evaluation[evaluation["date"] == current_date]
        predict_input = day_games.drop(
            columns=["home_score", "away_score"], errors="ignore"
        )
        predict_input = validate_model_input(predict_input, context="Backtest model input")
        predictions = model.predict(predict_input)
        predictions = validate_predictions(predictions, context="Backtest model output")
        _attach_prediction_metadata(predictions, model=model, train_data=train_data)
        pred_df = _predictions_to_frame(predictions)
        pred_df["date"] = pd.to_datetime(pred_df["date"]).dt.normalize()

        merged = day_games.merge(
            pred_df,
            on=["date", "home_team", "away_team"],
            how="left",
            suffixes=("", "_pred"),
        )
        merged["home_win"] = _home_win_flag(merged)
        merged["actual_margin"] = merged["home_score"] - merged["away_score"]
        prediction_frames.append(merged)

    if prediction_frames:
        predictions_df = pd.concat(prediction_frames, ignore_index=True)
    else:
        raise ValueError(
            "Backtest produced no predictions. "
            "This happens when each evaluation date has no training data "
            "(games must exist before each evaluation date)."
        )

    metrics_by_date = _aggregate_metrics_by_date(predictions_df)
    metrics_overall = _aggregate_overall_metrics(predictions_df)
    calibration = _calibration_table(predictions_df)
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
    export_backtest_outputs(outputs, run_id=run_id)
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
            "pred_margin": pred.pred_margin,
            "pred_total": pred.pred_total,
            "model_id": pred.metadata.get("model_id"),
        }
        if pred.extra:
            row["extra"] = pred.extra
        rows.append(row)
    return pd.DataFrame(rows)


def _prediction_hash_columns(pred_df: pd.DataFrame) -> list[str]:
    columns = [
        "game_id",
        "date",
        "home_team",
        "away_team",
        "p_home_win",
        "pred_margin",
        "pred_total",
    ]
    if "extra" in pred_df.columns:
        columns.append("extra")
    return columns


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
                "log_loss",
                "brier_score",
                "mae_margin",
                "margin_games",
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
        "log_loss": None,
        "brier_score": None,
        "mae_margin": None,
        "margin_games": 0,
    }

    prob_df = df.dropna(subset=["p_home_win", "home_win"])
    if not prob_df.empty:
        probs = np.clip(prob_df["p_home_win"].astype(float), 1e-6, 1 - 1e-6)
        actuals = prob_df["home_win"].astype(float)
        metrics["log_loss"] = float(
            -np.mean(actuals * np.log(probs) + (1 - actuals) * np.log(1 - probs))
        )
        metrics["brier_score"] = float(np.mean((probs - actuals) ** 2))

    margin_df = df.dropna(subset=["pred_margin", "actual_margin"])
    if not margin_df.empty:
        metrics["mae_margin"] = float(
            np.mean(np.abs(margin_df["pred_margin"] - margin_df["actual_margin"]))
        )
        metrics["margin_games"] = int(len(margin_df))

    return metrics


def _calibration_table(predictions_df: pd.DataFrame) -> pd.DataFrame:
    if predictions_df.empty:
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
