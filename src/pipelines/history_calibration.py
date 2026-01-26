"\"\"\"Helpers for fitting calibrators from historical prediction outcomes.\"\"\""

from __future__ import annotations

import logging
import math
import sqlite3
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from markets.base import Market
from markets.registry import get_market_spec
from pipelines.calibration_utils import select_calibrator

_LOG = logging.getLogger(__name__)


def _iso_date(value: str | None) -> str | None:
    if not value:
        return None
    return date.fromisoformat(value).isoformat()


def _compute_outcome(row: pd.Series, market: Market) -> int | None:
    if market == Market.ML:
        selection = str(row.get("selection") or "").strip()
        if not selection:
            return None
        home = str(row.get("home_team") or "").strip()
        away = str(row.get("away_team") or "").strip()
        home_score = row.get("home_score")
        away_score = row.get("away_score")
        if home_score is None or away_score is None:
            return None
        if selection == home:
            if home_score > away_score:
                return 1
            if home_score < away_score:
                return 0
        if selection == away:
            if away_score > home_score:
                return 1
            if away_score < home_score:
                return 0
        return None
    if market == Market.SPREAD:
        selection = str(row.get("selection") or "").strip()
        home = str(row.get("home_team") or "").strip()
        away = str(row.get("away_team") or "").strip()
        line = row.get("line")
        if not selection or line is None:
            return None
        home_score = row.get("home_score")
        away_score = row.get("away_score")
        if home_score is None or away_score is None:
            return None
        margin = (home_score - away_score) + float(line)
        if selection == home:
            return 1 if margin > 0 else 0 if margin < 0 else None
        if selection == away:
            neg = -margin
            return 1 if neg > 0 else 0 if neg < 0 else None
        return None
    if market == Market.TOTAL:
        line = row.get("line")
        if line is None:
            return None
        total = (row.get("home_score") or 0) + (row.get("away_score") or 0)
        selection = str(row.get("selection") or "").strip().lower()
        if selection == "over":
            if total > float(line):
                return 1
            if total < float(line):
                return 0
        elif selection == "under":
            if total < float(line):
                return 1
            if total > float(line):
                return 0
        return None
    return None


def _filter_probability(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(prob) or math.isinf(prob):
        return None
    return min(max(prob, 1e-6), 1 - 1e-6)


def build_history_calibration_dataset(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    market: Market,
    source: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Collect prediction/outcome pairs from bets_predictions for a market."""
    query = """
        SELECT
            bp.model_prob,
            bp.selection,
            bp.line,
            bp.market_forecast_source,
            g.home_team,
            g.away_team,
            g.home_score,
            g.away_score
        FROM bets_predictions bp
        JOIN games g ON bp.game_id = g.game_id
        WHERE bp.sport = ?
          AND bp.season = ?
          AND bp.market_type = ?
          AND g.home_score IS NOT NULL
          AND g.away_score IS NOT NULL
    """
    params: list[str] = [sport, season, market.value]
    if source:
        query += " AND bp.market_forecast_source = ?"
        params.append(source)
    if start_date:
        query += " AND DATE(bp.prediction_date) >= DATE(?)"
        params.append(_iso_date(start_date))
    if end_date:
        query += " AND DATE(bp.prediction_date) <= DATE(?)"
        params.append(_iso_date(end_date))

    with closing(sqlite3.connect(db_path)) as conn:
        df = pd.read_sql(query, conn, params=params)
    
    _LOG.info(f"[build_history_dataset] Query returned {len(df)} rows for market={market.value}, source={source}")
    if not df.empty:
        source_counts = df.groupby("market_forecast_source").size()
        for src, count in source_counts.items():
            _LOG.info(f"[build_history_dataset]   - {count} rows from source={src}")

    records: list[dict[str, float | int]] = []
    for _, row in df.iterrows():
        raw_prob = _filter_probability(row.get("model_prob"))
        if raw_prob is None:
            continue
        outcome = _compute_outcome(row, market)
        if outcome is None:
            continue
        records.append({"p_home_win": raw_prob, "home_win": outcome})
    if not records:
        _LOG.warning(f"[build_history_dataset] No valid records created for market={market.value}, source={source}")
        return pd.DataFrame(columns=["p_home_win", "home_win"])
    _LOG.info(f"[build_history_dataset] Created {len(records)} valid calibration records for market={market.value}, source={source}")
    return pd.DataFrame(records)


def calibrate_market_from_history(
    *,
    db_path: str | Path,
    sport: str,
    season: str,
    market: Market | str,
    source: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    method: str = "auto",
) -> Path:
    if isinstance(market, str):
        try:
            market_enum = Market[market.strip().upper()]
        except KeyError:
            market_enum = Market(market.strip().lower())
    else:
        market_enum = market

    dataset = build_history_calibration_dataset(
        db_path,
        sport=sport,
        season=season,
        market=market_enum,
        source=source,
        start_date=start_date,
        end_date=end_date,
    )
    _LOG.info(f"[calibrate_market] Dataset built: market={market_enum.value}, source={source}, rows={len(dataset)}")
    if dataset.empty:
        raise ValueError(
            f"No historical predictions found for market={market_enum.value}"
            f"{' source='+source if source else ''} in {db_path}"
        )
    calibrator = select_calibrator(method, len(dataset))
    calibrator.fit(dataset)

    spec = get_market_spec(market_enum)
    identifier = source or market_enum.value
    calib_dir = spec.calibrator_dir(sport, season, identifier)
    calib_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = calib_dir / f"{identifier}_{timestamp}.joblib"
    calibrator.save(out_path)
    _LOG.info(
        "Fitted history calibrator market=%s source=%s rows=%d -> %s",
        market_enum.value,
        identifier,
        len(dataset),
        out_path,
    )
    return out_path
