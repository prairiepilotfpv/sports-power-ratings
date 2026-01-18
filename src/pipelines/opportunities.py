"""Pipeline to evaluate market snapshots into betting opportunities."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import logging
import sqlite3

import pandas as pd

from data import betting_repository as br
from data.repository import load_games
from eval.evaluator import evaluate_market_rows
from eval.validation import get_validation_config
from markets.base import Market
from pipelines import guardrails
from pipelines import schedule as schedule_pipeline
from pipelines.model_params import resolve_model_market_params_with_metadata
from utils.normalization import normalize_evaluation_market_type

logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_market_type(value: str | None) -> str | None:
    return normalize_evaluation_market_type(value) or (
        str(value).strip().lower() if value is not None else None
    )


def _clean_value(value: object) -> object | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _model_ref_from_row(row: pd.Series | dict, default_model: str | None = None) -> str | None:
    if isinstance(row, pd.Series):
        model_ref = row.get("model") or row.get("model_id")
    else:
        model_ref = row.get("model") or row.get("model_id")
    return model_ref or default_model


def _format_validation_exclusions(
    exclusions: Iterable[tuple[str | None, str | None, list[str]]],
    *,
    default_model: str | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for model_ref, game_id, reasons in exclusions:
        reason = ", ".join(reasons) if reasons else "validation_failed"
        rows.append(
            {
                "game_id": game_id,
                "model": model_ref or default_model,
                "excluded_reason": reason,
            }
        )
    return rows


def load_market_snapshots(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    snapshot_run_id: str | None = None,
    snapshot_date: str | None = None,
) -> pd.DataFrame:
    """Load the latest market_lines per game/market/selection for the requested filters."""
    if snapshot_run_id is None and snapshot_date is None:
        raise ValueError("Provide snapshot_run_id or snapshot_date to load market snapshots.")

    br.init_db(db_path)
    filters = ["sport = ?", "season = ?"]
    params: list[object] = [sport, season]
    if snapshot_date is not None:
        filters.append("date(game_date) = date(?)")
        params.append(snapshot_date)
    filter_clause = " AND ".join(filters)

    query = f"""
        WITH latest_lines AS (
            SELECT *
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY sport, season, game_id, market_type,
                                        COALESCE(selection_team_id, -1),
                                        COALESCE(selection, ''),
                                        COALESCE(line, -9999),
                                        COALESCE(book, '')
                           ORDER BY datetime(imported_at) DESC, id DESC
                       ) AS rn
                FROM market_lines
                WHERE {filter_clause}
            )
            WHERE rn = 1
        )
        SELECT
            latest_lines.id AS source_market_snapshot_id,
            '' AS snapshot_run_id,
            latest_lines.imported_at AS captured_at,
            latest_lines.book,
            latest_lines.market_type,
            COALESCE(latest_lines.selection, '') AS selection,
            latest_lines.line,
            latest_lines.odds,
            latest_lines.game_id,
            g.home_team,
            g.away_team,
            g.sport,
            g.season
        FROM latest_lines
        LEFT JOIN games g ON latest_lines.game_id = g.game_id
        ORDER BY latest_lines.imported_at DESC, latest_lines.game_id
    """
    with closing(sqlite3.connect(Path(db_path))) as conn:
        cur = conn.execute(query, params)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]

    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    df["market_type"] = df["market_type"].apply(_normalize_market_type)
    return df


def load_schedule_predictions(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str,
    game_ids: Iterable[str],
) -> pd.DataFrame:
    """Load schedule projections for the provided game ids."""
    rows = load_games(db_path, sport=sport, season=season)
    games_df = schedule_pipeline.normalize_games(rows)
    if games_df.empty:
        return games_df
    resolution = resolve_model_market_params_with_metadata(
        model,
        db_path=db_path,
        sport=sport,
        season=season,
        market=Market.ML,
    )
    schedule_df = schedule_pipeline._build_schedule_dataframe(
        games_df,
        db_path=db_path,
        sport=sport,
        season=season,
        model=model,
        upcoming_only=False,
        model_params=resolution.params,
        params_source=resolution.params_source,
        tuned_metric_used=resolution.tuned_metric_used,
        params_run_id=resolution.source_run_id,
        params_market=resolution.market,
    )
    if schedule_df.empty:
        return schedule_df

    game_ids_set = {gid for gid in game_ids if gid}
    filtered = schedule_df[schedule_df["game_id"].isin(game_ids_set)].copy()
    if filtered.empty:
        return filtered
    filtered["model"] = model
    filtered["sport"] = sport
    return filtered


def build_opportunities(
    db_path: str | Path,
    *,
    review_run_id: str,
    sport: str,
    season: str,
    model: str,
    snapshot_run_id: str | None = None,
    snapshot_date: str | None = None,
) -> dict:
    """Evaluate market snapshots into opportunities and persist them."""
    market_df = load_market_snapshots(
        db_path,
        sport=sport,
        season=season,
        snapshot_run_id=snapshot_run_id,
        snapshot_date=snapshot_date,
    )
    if market_df.empty:
        return {"inserted": 0, "review_run_id": review_run_id}

    game_ids = market_df["game_id"].dropna().unique().tolist()
    predictions = load_schedule_predictions(
        db_path,
        sport=sport,
        season=season,
        model=model,
        game_ids=game_ids,
    )
    if predictions.empty:
        logger.warning("No predictions found for review_run_id=%s", review_run_id)
        return {"inserted": 0, "review_run_id": review_run_id}

    exclusions: list[dict] = []
    pre_guardrail = predictions.copy()
    validation_config = get_validation_config(sport)
    predictions = guardrails.enforce_margin_sd_guardrail(predictions, sport=sport)
    if not pre_guardrail.empty:
        dropped_idx = pre_guardrail.index.difference(predictions.index)
        if len(dropped_idx) > 0:
            dropped = pre_guardrail.loc[dropped_idx]
            for _, row in dropped.iterrows():
                exclusions.append(
                    {
                        "game_id": row.get("game_id"),
                        "model": _model_ref_from_row(row, model),
                        "excluded_reason": "margin_sd_guardrail",
                    }
                )

    predictions, validation_exclusions = guardrails.apply_prediction_validation(
        predictions, sport=sport, validation_config=validation_config
    )
    exclusions.extend(
        _format_validation_exclusions(validation_exclusions, default_model=model)
    )

    opportunities_df, _ = evaluate_market_rows(
        predictions,
        market_df,
        include_excluded_reason=True,
        validation_config=validation_config,
    )
    if not opportunities_df.empty and "excluded_reason" in opportunities_df.columns:
        excluded_markets = opportunities_df[opportunities_df["excluded_reason"].notna()]
        for _, row in excluded_markets.iterrows():
            exclusions.append(
                {
                    "game_id": row.get("game_id"),
                    "model": model,
                    "excluded_reason": row.get("excluded_reason"),
                }
            )
    if exclusions:
        br.add_prediction_exclusions(
            db_path,
            review_run_id=review_run_id,
            exclusions=exclusions,
        )
    if opportunities_df.empty:
        return {"inserted": 0, "review_run_id": review_run_id}

    join_cols = ["game_id", "market_type", "selection", "line", "odds"]
    source_lookup = market_df[join_cols + ["source_market_snapshot_id"]]
    merged = opportunities_df.merge(source_lookup, on=join_cols, how="left")

    now = _utcnow_iso()
    inserted = 0
    with closing(sqlite3.connect(Path(db_path))) as conn:
        cur = conn.cursor()
        for _, row in merged.iterrows():
            cur.execute(
                """
                INSERT INTO opportunities (
                    review_run_id,
                    game_id,
                    market_type,
                    selection,
                    line,
                    odds,
                    implied_prob,
                    model_prob,
                    edge,
                    ev,
                    source_market_snapshot_id,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(review_run_id, game_id, market_type, selection) DO UPDATE SET
                    line = excluded.line,
                    odds = excluded.odds,
                    implied_prob = excluded.implied_prob,
                    model_prob = excluded.model_prob,
                    edge = excluded.edge,
                    ev = excluded.ev,
                    source_market_snapshot_id = excluded.source_market_snapshot_id,
                    created_at = excluded.created_at
                """,
                (
                    review_run_id,
                    _clean_value(row.get("game_id")),
                    _clean_value(row.get("market_type")),
                    _clean_value(row.get("selection")),
                    _clean_value(row.get("line")),
                    _clean_value(row.get("odds")),
                    _clean_value(row.get("implied_prob")),
                    _clean_value(row.get("model_prob")),
                    _clean_value(row.get("edge")),
                    _clean_value(row.get("ev")),
                    _clean_value(row.get("source_market_snapshot_id")),
                    now,
                ),
            )
            inserted += 1
        conn.commit()

    return {"inserted": inserted, "review_run_id": review_run_id}


__all__ = [
    "build_opportunities",
    "load_market_snapshots",
    "load_schedule_predictions",
]
