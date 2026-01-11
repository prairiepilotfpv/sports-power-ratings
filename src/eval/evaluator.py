"""EV aggregation with prediction validation and model weights."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Tuple
import logging
import math
import statistics

import pandas as pd

from eval.validation import ValidationConfig, get_validation_config, validate_prediction_row
from pipelines.projections import _normal_cdf
from utils.odds import american_to_implied, expected_value

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationConfig:
    model_weights: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "spreads": {
                "bradley-terry": 1.0,
                "elo": 1.0,
                "gssd": 1.0,
                "poisson": 0.0,
                "toor": 0.0,
            },
            "moneyline": {
                "bradley-terry": 1.0,
                "elo": 1.0,
                "gssd": 1.0,
                "poisson": 0.0,
                "toor": 0.0,
            },
            "totals": {
                "bradley-terry": 1.0,
                "elo": 1.0,
                "gssd": 1.0,
                "poisson": 0.25,
                "toor": 0.0,
            },
        }
    )
    poisson_divergence_threshold: float = 8.0
    poisson_divergent_weight: float = 0.1


DEFAULT_EVAL_CONFIG = EvaluationConfig()


def _canonical_model_name(model: Any) -> str:
    if model is None:
        return ""
    return str(model).strip().lower().replace("_", "-")


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def _home_win_prob(row: Mapping[str, Any]) -> float | None:
    for key in (
        "p_home_win_calibrated",
        "home_win_prob_calibrated",
        "model_p_home_win_calibrated",
        "model_p_home_win",
        "home_win_prob",
        "normal_p_home_win",
        "logistic_home_win_prob",
    ):
        val = _coerce_float(row.get(key))
        if val is not None:
            return val
    return None


def _margin_params(row: Mapping[str, Any]) -> Tuple[float | None, float | None]:
    return _coerce_float(row.get("margin_mean")), _coerce_float(row.get("margin_sd") or row.get("margin_std"))


def _total_params(row: Mapping[str, Any]) -> Tuple[float | None, float | None]:
    total_mean = _coerce_float(
        row.get("total_mean")
        or row.get("projected_total")
        or row.get("total")
    )
    total_sd = _coerce_float(row.get("total_sd") or row.get("total_std"))
    return total_mean, total_sd


def _is_home_selection(selection: str | None, home_team: str | None, away_team: str | None) -> bool | None:
    if selection is None:
        return None
    sel = selection.strip().lower()
    if home_team and sel == home_team.strip().lower():
        return True
    if away_team and sel == away_team.strip().lower():
        return False
    return None


def _cover_probability(line: float, margin_mean: float, margin_sd: float, is_home: bool) -> float:
    threshold = -line if is_home else line
    # P(home margin > -line) for home, P(home margin < line) for away
    if is_home:
        return 1.0 - _normal_cdf(threshold, mean=margin_mean, sd=margin_sd)
    return _normal_cdf(threshold, mean=margin_mean, sd=margin_sd)


def _over_probability(line: float, total_mean: float, total_sd: float) -> float:
    return 1.0 - _normal_cdf(line, mean=total_mean, sd=total_sd)


def _validation_frame(predictions: pd.DataFrame, config: ValidationConfig, *, require_score_bounds: bool = True) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    results = predictions.copy(deep=True)
    validity: list[bool] = []
    reasons: list[list[str]] = []
    for _, row in results.iterrows():
        ok, why = validate_prediction_row(
            row.to_dict(), config=config, require_score_bounds=require_score_bounds
        )
        validity.append(ok)
        reasons.append(why)
    results["__is_valid"] = validity
    results["__invalid_reasons"] = reasons
    return results


def _poisson_weight(model: str, market_type: str, base_weight: float, *, poisson_total: float | None, other_totals: Iterable[float], eval_config: EvaluationConfig) -> float:
    if model != "poisson" or market_type != "totals" or base_weight <= 0:
        return base_weight
    others = [t for t in other_totals if t is not None]
    if not others or poisson_total is None:
        return base_weight
    median_total = statistics.median(others)
    if abs(poisson_total - median_total) > eval_config.poisson_divergence_threshold:
        return eval_config.poisson_divergent_weight
    return base_weight


def evaluate_market_rows(
    predictions: pd.DataFrame | Iterable[Mapping[str, Any]],
    markets: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    validation_config: ValidationConfig | None = None,
    eval_config: EvaluationConfig = DEFAULT_EVAL_CONFIG,
    include_excluded_reason: bool = False,
    debug: bool = False,
    debug_output_path: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate markets against model predictions with guardrails.

    Returns (opportunities_df, debug_df). debug_df is empty unless debug=True.
    """
    pred_df = predictions if isinstance(predictions, pd.DataFrame) else pd.DataFrame(predictions)
    market_df = markets if isinstance(markets, pd.DataFrame) else pd.DataFrame(markets)
    if pred_df.empty or market_df.empty:
        empty = pd.DataFrame()
        return empty, empty

    # Try to infer sport from predictions or markets to select sport-specific guardrails.
    sport = None
    if "sport" in pred_df.columns:
        non_null = pred_df["sport"].dropna()
        if not non_null.empty:
            sport = str(non_null.iloc[0]).strip().lower()
    if sport is None and "sport" in market_df.columns:
        non_null_m = market_df["sport"].dropna()
        if not non_null_m.empty:
            sport = str(non_null_m.iloc[0]).strip().lower()

    config = validation_config if validation_config is not None else get_validation_config(sport)
    pred_df = _validation_frame(pred_df, config, require_score_bounds=(sport is not None))
    # Canonicalize model name from either `model` or `model_id` to support both fields.
    model_series = pred_df.get("model", pd.Series(dtype=object))
    if (model_series is None or getattr(model_series, "isna", lambda: True)().all()) and "model_id" in pred_df.columns:
        model_series = pred_df.get("model_id", pd.Series(dtype=object))
    # Ensure we have a Series to apply the canonicalizer
    if model_series is None:
        model_series = pd.Series([""] * len(pred_df))
    pred_df["__model"] = model_series.apply(_canonical_model_name)

    invalid = pred_df[~pred_df["__is_valid"]]
    for _, row in invalid.iterrows():
        model_ref = row.get("model") or row.get("model_id")
        logger.warning(
            "Excluding prediction model=%s game_id=%s reasons=%s",
            model_ref,
            row.get("game_id"),
            row.get("__invalid_reasons"),
        )

    valid_preds = pred_df[pred_df["__is_valid"]]
    if valid_preds.empty:
        empty = pd.DataFrame()
        return empty, pd.DataFrame()

    opportunities: list[dict] = []
    debug_rows: list[dict] = []

    valid_by_game = {gid: frame.sort_values("__model") for gid, frame in valid_preds.groupby("game_id")}
    totals_by_game = {
        gid: {
            r["__model"]: _coerce_float(
                r.get("total_mean") or r.get("projected_total") or r.get("total")
            )
            for _, r in frame.iterrows()
        }
        for gid, frame in valid_by_game.items()
    }

    for _, market in market_df.iterrows():
        game_id = market.get("game_id")
        market_type_raw = str(market.get("market_type", "")).strip().lower()
        market_type = market_type_raw
        if market_type in {"ml", "moneyline"}:
            market_type = "moneyline"
            weight_key = "moneyline"
        elif market_type == "spread":
            weight_key = "spreads"
        elif market_type == "total":
            weight_key = "totals"
        else:
            weight_key = market_type
        selection = market.get("selection")
        line = _coerce_float(market.get("line"))
        odds = market.get("odds")
        implied_prob = _coerce_float(market.get("implied_prob"))
        if implied_prob is None and odds is not None:
            try:
                implied_prob = american_to_implied(int(odds))
            except Exception:
                implied_prob = None

        home_team = market.get("home_team")
        away_team = market.get("away_team")
        is_home_sel = _is_home_selection(selection, home_team, away_team)

        game_preds = valid_by_game.get(game_id)
        if game_preds is None or game_preds.empty:
            row = {
                "game_id": game_id,
                "market_type": market_type,
                "selection": selection,
                "line": line,
                "odds": odds,
                "implied_prob": implied_prob,
                "model_prob": None,
                "edge": None,
                "ev": None,
            }
            if include_excluded_reason:
                row["excluded_reason"] = "no_valid_predictions"
            opportunities.append(row)
            if debug:
                debug_rows.append({**row, "per_model_probs": {}, "weights": {}})
            continue

        per_model_probs: dict[str, float] = {}
        weights_used: dict[str, float] = {}
        total_weight = 0.0
        weighted_prob_sum = 0.0

        totals_for_game = totals_by_game.get(game_id, {})
        other_totals = [t for m_name, t in totals_for_game.items() if m_name != "poisson"]
        poisson_total = totals_for_game.get("poisson")

        for _, pred_row in game_preds.iterrows():
            model_name = pred_row["__model"]
            base_weights = eval_config.model_weights.get(weight_key, {})
            base_weight = base_weights.get(model_name, 0.0)
            weight = _poisson_weight(
                model_name,
                market_type,
                base_weight,
                poisson_total=poisson_total,
                other_totals=[t for t in other_totals if t is not None and not math.isnan(t)],
                eval_config=eval_config,
            )
            if weight <= 0:
                continue

            prob = None
            if market_type == "moneyline":
                home_prob = _home_win_prob(pred_row)
                if home_prob is not None and is_home_sel is not None:
                    prob = home_prob if is_home_sel else 1.0 - home_prob
            elif market_type == "spread":
                margin_mean, margin_sd = _margin_params(pred_row)
                if margin_mean is not None and margin_sd is not None and line is not None and is_home_sel is not None:
                    try:
                        prob = _cover_probability(line, margin_mean, margin_sd, is_home_sel)
                    except Exception:
                        prob = None
            elif market_type == "total":
                total_mean, total_sd = _total_params(pred_row)
                if total_mean is not None and total_sd is not None and line is not None and selection:
                    sel_lower = str(selection).strip().lower()
                    try:
                        over_p = _over_probability(line, total_mean, total_sd)
                        if "over" in sel_lower:
                            prob = over_p
                        elif "under" in sel_lower:
                            prob = 1.0 - over_p
                    except Exception:
                        prob = None

            if prob is None:
                continue

            per_model_probs[model_name] = prob
            weights_used[model_name] = weight
            total_weight += weight
            weighted_prob_sum += weight * prob

        model_prob = None
        if total_weight > 0:
            model_prob = weighted_prob_sum / total_weight
        edge = None
        ev = None
        if model_prob is not None and implied_prob is not None:
            edge = model_prob - implied_prob
        if model_prob is not None:
            try:
                ev = expected_value(implied_prob, model_prob, odds=int(odds) if odds is not None else None)
            except Exception:
                ev = None

        row = {
            "game_id": game_id,
            "market_type": market_type,
            "selection": selection,
            "line": line,
            "odds": odds,
            "implied_prob": implied_prob,
            "model_prob": model_prob,
            "edge": edge,
            "ev": ev,
        }
        if include_excluded_reason:
            if model_prob is None:
                row["excluded_reason"] = "no_valid_model_prob"
            else:
                row["excluded_reason"] = None
        opportunities.append(row)

        if debug:
            debug_rows.append(
                {
                    **row,
                    "breakeven_prob": implied_prob,
                    "per_model_probs": per_model_probs,
                    "weights": weights_used,
                    "total_weight": total_weight,
                }
            )

    opp_df = pd.DataFrame(opportunities)
    debug_df = pd.DataFrame(debug_rows)
    if debug and debug_output_path:
        try:
            serializable_debug = debug_df.copy(deep=True)
            for col in ("per_model_probs", "weights"):
                if col in serializable_debug.columns:
                    serializable_debug[col] = serializable_debug[col].apply(lambda v: dict(v) if isinstance(v, dict) else v)
            serializable_debug.to_csv(debug_output_path, index=False)
        except Exception:
            logger.exception("Failed to write debug output to %s", debug_output_path)
    return opp_df, debug_df
