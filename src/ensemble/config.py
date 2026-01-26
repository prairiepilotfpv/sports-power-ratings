"""Helpers for loading and validating ensemble configuration files.

Config resolution order (highest priority first):
1) Custom per-market path: outputs/ensembles/<sport>/<season>/<market>/<ensemble_id>.json
2) Sport/season default per-market: outputs/ensembles/<sport>/<season>/<market>/default.json
3) Global repo default: src/ensemble/default_configs/<market>.json
4) Fallback: generated config using the default allowlist per market.

Schema (single-market files):
{
    "sport": "nba",
    "season": "2025-26",
    "market": "ML",
    "ensemble_id": "ensemble_ml_v1",
    "metric_slot": "log_loss",
    "models": ["elo", "bradley-terry"],
    "weights": {"elo": 0.5, "bradley-terry": 0.5}
}

Multi-market files (legacy) with a `markets` object are also supported and merged.
Rules:
- Market keys must be ML/SPREAD/TOTAL (case-insensitive).
- Missing weights -> equal weights.
- Missing models -> default allowlist for that market.
- Weight keys outside `models` are dropped; weights are renormalized when needed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

from models.registry import normalize_model_name

_MARKET_KEYS = {"ML", "SPREAD", "TOTAL"}
_METRIC_SLOTS = {"log_loss", "mae_margin", "mae_total"}

_FALLBACK_MARKET_MODELS: dict[str, list[str]] = {
    "ML": ["elo", "bradley-terry"],
    "SPREAD": ["elo", "gssd", "toor"],
    "TOTAL": ["poisson", "gssd", "toor"],
}

_FALLBACK_MARKET_METRICS: dict[str, str] = {
    "ML": "log_loss",
    "SPREAD": "mae_margin",
    "TOTAL": "mae_total",
}

_FALLBACK_ENSEMBLE_IDS: dict[str, str] = {
    "ML": "ensemble_ml_v1",
    "SPREAD": "ensemble_spread_v1",
    "TOTAL": "ensemble_total_v1",
}

_GLOBAL_DEFAULT_DIR = Path(__file__).resolve().parent / "default_configs"


def _legacy_config_path(sport: str, season: str) -> Path:
    return Path("outputs") / "ensembles" / sport / season / "ensemble_config.json"


def _market_config_path(sport: str, season: str, market: str, filename: str) -> Path:
    return Path("outputs") / "ensembles" / sport / season / market / filename


def _global_default_config_path(market: str) -> Path:
    return _GLOBAL_DEFAULT_DIR / f"{market}.json"


def _normalize_market_key(key: str) -> str:
    norm = str(key).strip().upper()
    if norm.lower() == "spread":
        norm = "SPREAD"
    if norm not in _MARKET_KEYS:
        raise ValueError(f"Unsupported market key: {key}")
    return norm


def _load_default_market_defaults(
    *,
    fallback_models: dict[str, list[str]],
    fallback_metrics: dict[str, str],
    fallback_ensemble_ids: dict[str, str],
) -> tuple[dict[str, list[str]], dict[str, str], dict[str, str], list[str]]:
    models = {k: list(v) for k, v in fallback_models.items()}
    metrics = dict(fallback_metrics)
    ensemble_ids = dict(fallback_ensemble_ids)
    warnings: list[str] = []

    for market in sorted(_MARKET_KEYS):
        path = _global_default_config_path(market)
        if not path.exists():
            warnings.append(f"default config missing for {market}: {path}")
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.append(f"default config unreadable for {market}: {exc}")
            continue

        entry: dict | None = None
        if isinstance(raw, dict) and "markets" in raw:
            markets = raw.get("markets") or {}
            entry = markets.get(market) if isinstance(markets, dict) else None
        elif isinstance(raw, dict):
            entry = raw

        if not isinstance(entry, dict):
            warnings.append(f"default config invalid for {market}: expected object")
            continue

        raw_market = entry.get("market")
        if raw_market is not None:
            try:
                normalized = _normalize_market_key(raw_market)
                if normalized != market:
                    warnings.append(
                        f"default config market mismatch for {market}: {raw_market}"
                    )
            except ValueError as exc:
                warnings.append(f"default config invalid market for {market}: {exc}")

        raw_models = entry.get("models")
        if isinstance(raw_models, list):
            parsed_models = [str(m).strip() for m in raw_models if str(m).strip()]
            if parsed_models:
                models[market] = list(dict.fromkeys(parsed_models))
            else:
                warnings.append(
                    f"default config models empty for {market}; using fallback"
                )
        else:
            warnings.append(
                f"default config models missing for {market}; using fallback"
            )

        metric_slot = entry.get("metric_slot")
        if metric_slot is not None:
            metric_slot = str(metric_slot).strip().lower()
            if metric_slot in _METRIC_SLOTS:
                metrics[market] = metric_slot
            else:
                warnings.append(
                    f"default config metric_slot invalid for {market}: {metric_slot}"
                )
        else:
            warnings.append(
                f"default config metric_slot missing for {market}; using fallback"
            )

        ensemble_id = entry.get("ensemble_id")
        if ensemble_id is not None:
            ensemble_id = str(ensemble_id).strip()
            if ensemble_id:
                ensemble_ids[market] = ensemble_id
            else:
                warnings.append(
                    f"default config ensemble_id empty for {market}; using fallback"
                )
        else:
            warnings.append(
                f"default config ensemble_id missing for {market}; using fallback"
            )

    return models, metrics, ensemble_ids, warnings


DEFAULT_MARKET_MODELS, DEFAULT_MARKET_METRICS, DEFAULT_ENSEMBLE_IDS, DEFAULT_CONFIG_WARNINGS = (
    _load_default_market_defaults(
        fallback_models=_FALLBACK_MARKET_MODELS,
        fallback_metrics=_FALLBACK_MARKET_METRICS,
        fallback_ensemble_ids=_FALLBACK_ENSEMBLE_IDS,
    )
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _equal_weights(models: Iterable[str]) -> dict[str, float]:
    models = list(models)
    if not models:
        return {}
    weight = 1.0 / len(models)
    return {m: weight for m in models}


def _normalize_models_and_weights(
    *,
    market_key: str,
    raw_models: object,
    raw_weights: object,
    default_models: list[str],
    available_models: Iterable[str] | None = None,
) -> tuple[list[str], dict[str, float] | None, list[str]]:
    warnings: list[str] = []
    models: list[str]
    if raw_models is None:
        normalized_defaults = [normalize_model_name(m) for m in default_models]
        models = [m for m in normalized_defaults if m]
        warnings.append(
            f"models missing for {market_key}; using default allowlist {default_models}"
        )
    else:
        if not isinstance(raw_models, list):
            raise ValueError(f"models for {market_key} must be a list")
        models = []
        for m in raw_models:
            name = str(m).strip()
            if not name:
                continue
            normalized = normalize_model_name(name)
            if normalized:
                models.append(normalized)
        models = list(dict.fromkeys(models))
        if not models:
            normalized_defaults = [normalize_model_name(m) for m in default_models]
            models = [m for m in normalized_defaults if m]
            warnings.append(
                f"models empty for {market_key}; using default allowlist {default_models}"
            )

    weights: dict[str, float] | None = None
    if raw_weights is not None:
        if not isinstance(raw_weights, dict):
            raise ValueError(f"weights for {market_key} must be an object")
        weights = {}
        for mk, mv in raw_weights.items():
            key_raw = str(mk).strip()
            if not key_raw:
                continue
            key_norm = normalize_model_name(key_raw)
            if not key_norm:
                warnings.append(
                    f"Invalid weight key {mk} in {market_key}; dropping entry"
                )
                continue
            try:
                weight_value = float(mv)
            except Exception:
                warnings.append(
                    f"Invalid weight for model {mk} in {market_key}; dropping entry"
                )
                continue
            if key_norm in weights:
                warnings.append(
                    f"Duplicate weight key {key_raw} normalized to {key_norm} in {market_key}; summing"
                )
            weights[key_norm] = weights.get(key_norm, 0.0) + weight_value
        if not weights:
            warnings.append(
                f"weights empty for {market_key}; using equal weights"
            )
            weights = None

    if weights is not None:
        filtered = {m: w for m, w in weights.items() if m in set(models)}
        if len(filtered) != len(weights):
            warnings.append(
                f"weights trimmed for {market_key} to match listed models"
            )
        weights = filtered or None
        if weights is not None and models:
            missing_weights = [m for m in models if m not in weights]
            if missing_weights:
                warnings.append(
                    f"weights missing for {market_key} models {missing_weights}; using equal weights"
                )
                weights = None

    if weights is not None:
        total = sum(weights.values())
        if total <= 0:
            warnings.append(
                f"weights for {market_key} sum to 0; using equal weights"
            )
            weights = None
        elif abs(total - 1.0) > 1e-6:
            weights = {k: v / total for k, v in weights.items()}
            warnings.append(
                f"weights for {market_key} renormalized to sum to 1.0"
            )

    available_order: list[str] | None = None
    available_set: set[str] | None = None
    if available_models is not None:
        available_order = [normalize_model_name(m) for m in available_models]
        available_set = set(available_order)
    if available_set is not None:
        normalized_models = [normalize_model_name(m) for m in models]
        filtered_models = [m for m, norm in zip(models, normalized_models) if norm in available_set]
        missing = [m for m, norm in zip(models, normalized_models) if norm not in available_set]
        if missing:
            warnings.append(
                f"models unavailable for {market_key} skipped: {missing}"
            )
        if not filtered_models:
            fallback_models = available_order or []
            if fallback_models:
                filtered_models = [m for m in models if normalize_model_name(m) in available_set]
                if not filtered_models:
                    filtered_models = fallback_models
            if not filtered_models:
                warnings.append(
                    f"no available models for {market_key}; leaving membership empty"
                )
        models = filtered_models
        if weights is not None:
            weights = {m: w for m, w in weights.items() if normalize_model_name(m) in available_set}
            if not weights:
                warnings.append(
                    f"weights for {market_key} removed due to unavailable models; using equal weights"
                )
                weights = None
            else:
                total = sum(weights.values())
                if total > 0:
                    weights = {k: v / total for k, v in weights.items()}
                else:
                    warnings.append(
                        f"weights for {market_key} sum to 0 after filtering; using equal weights"
                    )
                    weights = None

    if weights is None and models:
        weights = _equal_weights(models)

    return models, weights, warnings


def _normalize_market_entry(
    *,
    market_key: str,
    entry: dict,
    default_models: list[str],
    available_models: Iterable[str] | None,
    default_ensemble_id: str,
) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    if not isinstance(entry, dict):
        raise ValueError(f"Market entry for {market_key} must be an object")

    ensemble_id = str(entry.get("ensemble_id") or default_ensemble_id).strip()
    if not ensemble_id:
        ensemble_id = default_ensemble_id
        warnings.append(
            f"ensemble_id missing for {market_key}; using {default_ensemble_id}"
        )

    metric_slot = entry.get("metric_slot")
    if metric_slot is not None:
        metric_slot = str(metric_slot).strip().lower()
        if metric_slot not in _METRIC_SLOTS:
            raise ValueError(
                f"metric_slot for {market_key} must be one of {_METRIC_SLOTS}"
            )
    else:
        metric_slot = DEFAULT_MARKET_METRICS.get(market_key)

    models, weights, mw_warnings = _normalize_models_and_weights(
        market_key=market_key,
        raw_models=entry.get("models"),
        raw_weights=entry.get("weights"),
        default_models=default_models,
        available_models=available_models,
    )
    warnings.extend(mw_warnings)

    return (
        {
            "ensemble_id": ensemble_id,
            "metric_slot": metric_slot,
            "models": models,
            "weights": weights,
        },
        warnings,
    )


def validate_ensemble_config(
    raw_config: dict,
    *,
    available_models: Iterable[str] | None = None,
    default_models: dict[str, list[str]] | None = None,
    default_ensemble_ids: dict[str, str] | None = None,
) -> Tuple[dict, list[str]]:
    """Validate and normalize a multi-market ensemble config object."""

    if not isinstance(raw_config, dict):
        raise ValueError("Ensemble config must be a JSON object.")

    warnings: list[str] = []
    sport = raw_config.get("sport")
    season = raw_config.get("season")

    markets_raw = raw_config.get("markets", {}) or {}
    if not isinstance(markets_raw, dict):
        raise ValueError("Ensemble config 'markets' must be an object")

    default_models = default_models or DEFAULT_MARKET_MODELS
    default_ensemble_ids = default_ensemble_ids or DEFAULT_ENSEMBLE_IDS
    markets: Dict[str, dict] = {}

    for key, value in markets_raw.items():
        market_key = _normalize_market_key(key)
        normalized_entry, warnings_for_market = _normalize_market_entry(
            market_key=market_key,
            entry=value,
            default_models=default_models.get(market_key, []),
            available_models=available_models,
            default_ensemble_id=default_ensemble_ids.get(market_key, ""),
        )
        warnings.extend(warnings_for_market)
        markets[market_key] = normalized_entry

    normalized = {
        "sport": sport,
        "season": season,
        "markets": markets,
    }
    return normalized, warnings


def _validate_single_market_config(
    raw_config: dict,
    *,
    market_fallback: str,
    available_models: Iterable[str] | None,
    default_models: dict[str, list[str]] | None,
    default_ensemble_ids: dict[str, str] | None,
) -> Tuple[dict, list[str]]:
    if not isinstance(raw_config, dict):
        raise ValueError("Ensemble config must be a JSON object.")

    warnings: list[str] = []
    default_models = default_models or DEFAULT_MARKET_MODELS
    default_ensemble_ids = default_ensemble_ids or DEFAULT_ENSEMBLE_IDS

    market_key = _normalize_market_key(raw_config.get("market", market_fallback))
    entry = {k: v for k, v in raw_config.items() if k != "market"}
    normalized_entry, warnings_for_market = _normalize_market_entry(
        market_key=market_key,
        entry=entry,
        default_models=default_models.get(market_key, []),
        available_models=available_models,
        default_ensemble_id=default_ensemble_ids.get(market_key, ""),
    )
    warnings.extend(warnings_for_market)
    normalized = {
        "sport": raw_config.get("sport"),
        "season": raw_config.get("season"),
        "markets": {market_key: normalized_entry},
    }
    return normalized, warnings


def _load_config_from_path(
    path: Path,
    *,
    market: str,
    available_models: Iterable[str] | None,
    default_models: dict[str, list[str]] | None,
    default_ensemble_ids: dict[str, str] | None,
) -> tuple[dict | None, list[str], str | None]:
    if not path.exists():
        return None, [], None
    raw_text = path.read_text(encoding="utf-8")
    data = json.loads(raw_text)
    warnings: list[str]
    if isinstance(data, dict) and "markets" in data:
        normalized, warnings = validate_ensemble_config(
            data,
            available_models=available_models,
            default_models=default_models,
            default_ensemble_ids=default_ensemble_ids,
        )
    else:
        normalized, warnings = _validate_single_market_config(
            data,
            market_fallback=market,
            available_models=available_models,
            default_models=default_models,
            default_ensemble_ids=default_ensemble_ids,
        )
    sha = _sha256_text(raw_text)
    return normalized, warnings, sha


def _resolve_market_config(
    *,
    sport: str,
    season: str,
    market: str,
    ensemble_id_hint: str,
    available_models: Iterable[str] | None,
    override_config: dict | None,
    default_models: dict[str, list[str]] | None,
    default_ensemble_ids: dict[str, str] | None,
) -> tuple[dict, dict, list[str]]:
    warnings: list[str] = []
    meta: dict[str, str | None] = {"path": None, "sha256": None, "source": "fallback"}
    default_models = default_models or DEFAULT_MARKET_MODELS
    default_ensemble_ids = default_ensemble_ids or DEFAULT_ENSEMBLE_IDS

    candidates: list[tuple[str, Path | None]] = []
    if override_config is not None and market in (override_config.get("markets") or {}):
        candidates.append(("override", None))
    candidates.append(("custom", _market_config_path(sport, season, market, f"{ensemble_id_hint}.json")))
    candidates.append(("season_default", _market_config_path(sport, season, market, "default.json")))
    legacy_path = _legacy_config_path(sport, season)
    candidates.append(("legacy", legacy_path))
    candidates.append(("global_default", _global_default_config_path(market)))

    normalized: dict | None = None
    sha_used: str | None = None
    for source, path in candidates:
        if source == "override":
            normalized = override_config
            sha_used = override_config.get("_meta", {}).get("sha256") if isinstance(override_config, dict) else None
            warnings.extend(override_config.get("_meta", {}).get("warnings", []) if isinstance(override_config, dict) else [])
        else:
            if path is None:
                continue
            normalized, warnings_for_path, sha_used = _load_config_from_path(
                path,
                market=market,
                available_models=available_models,
                default_models=default_models,
                default_ensemble_ids=default_ensemble_ids,
            )
            warnings.extend(warnings_for_path)
        if normalized and market in (normalized.get("markets") or {}):
            path_value = str(path) if path else None
            if path_value is None and source == "override":
                path_value = (override_config or {}).get("_meta", {}).get("path") if isinstance(override_config, dict) else None
            meta.update({"path": path_value, "sha256": sha_used, "source": source})
            break
        normalized = None

    if normalized is None or market not in (normalized.get("markets") or {}):
        models, weights, mw_warnings = _normalize_models_and_weights(
            market_key=market,
            raw_models=default_models.get(market, []),
            raw_weights=_equal_weights(default_models.get(market, [])),
            default_models=default_models.get(market, []),
            available_models=available_models,
        )
        warnings.append(
            f"No ensemble config found for market={market}; using fallback allowlist {default_models.get(market, [])}"
        )
        warnings.extend(mw_warnings)
        fallback_entry = {
            "ensemble_id": default_ensemble_ids.get(market, ""),
            "metric_slot": DEFAULT_MARKET_METRICS.get(market),
            "models": models,
            "weights": weights,
        }
        normalized = {
            "sport": sport,
            "season": season,
            "markets": {market: fallback_entry},
        }

    return normalized["markets"][market], meta, warnings


def load_ensemble_config(
    sport: str,
    season: str,
    path_override: str | Path | None = None,
    *,
    available_models: Iterable[str] | None = None,
    ensemble_ids: dict[str, str] | None = None,
    default_models: dict[str, list[str]] | None = None,
) -> dict:
    """Resolve ensemble config for all markets with fallbacks.

    Always returns a normalized config with `_meta` describing per-market sources.
    """

    default_models = default_models or DEFAULT_MARKET_MODELS
    default_ensemble_ids = ensemble_ids or DEFAULT_ENSEMBLE_IDS

    override_config: dict | None = None
    override_sha: str | None = None
    if path_override:
        override_path = Path(path_override)
        if override_path.exists():
            override_config, override_warnings, override_sha = _load_config_from_path(
                override_path,
                market="ML",
                available_models=available_models,
                default_models=default_models,
                default_ensemble_ids=default_ensemble_ids,
            )
            if override_config is not None:
                override_config.setdefault("_meta", {})
                override_config["_meta"].update(
                    {"path": str(override_path), "sha256": override_sha, "warnings": override_warnings}
                )

    resolved_markets: dict[str, dict] = {}
    meta_by_market: dict[str, dict] = {}
    aggregate_warnings: list[str] = []
    if DEFAULT_CONFIG_WARNINGS:
        aggregate_warnings.extend(DEFAULT_CONFIG_WARNINGS)

    for market in sorted(_MARKET_KEYS):
        market_config, meta, warnings = _resolve_market_config(
            sport=sport,
            season=season,
            market=market,
            ensemble_id_hint=default_ensemble_ids.get(market, ""),
            available_models=available_models,
            override_config=override_config,
            default_models=default_models,
            default_ensemble_ids=default_ensemble_ids,
        )
        resolved_markets[market] = market_config
        meta_by_market[market] = meta
        aggregate_warnings.extend(warnings)

    normalized = {
        "sport": sport,
        "season": season,
        "markets": resolved_markets,
        "_meta": {
            "warnings": aggregate_warnings,
            "markets": meta_by_market,
        },
    }
    first_path = next(
        (meta.get("path") for meta in meta_by_market.values() if meta.get("path")),
        None,
    )
    first_sha = next(
        (meta.get("sha256") for meta in meta_by_market.values() if meta.get("sha256")),
        None,
    )
    normalized["_meta"].update({"path": first_path, "sha256": first_sha})
    return normalized
