"""Market-specific configuration layer.

This module enforces the "market-specific optimization" contract:
- Each market (ML, SPREAD, TOTAL) has its own optimized set of base models
- Each base model uses its market-tuned active params (NOT global params)
- Ensembles are tuned specifically for that market's metric(s)

Key concepts:
- MarketParamsResolution: result of resolving params for one (model, market) pair
- MarketEnsembleSpec: ensemble configuration for a specific market
- resolve_market_params(): canonical resolver for market-specific model params
- get_market_ensemble_spec(): canonical resolver for market-specific ensemble config

HARD CONTRACT:
- TOTALS ensemble is tuned for totals and ONLY uses models with best TOTALS params
- SPREAD ensemble uses spread-tuned models/params
- ML ensemble uses ML-tuned models/params

Cross-market contamination is prevented by:
1. Always resolving params with explicit market argument
2. Never falling back to a different market's params
3. Logging/tracing all param resolutions for auditability
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from markets.base import Market
from models.registry import normalize_model_name, list_models
from pipelines.market_utils import _resolve_market_metric
from pipelines.model_params import (
    EffectiveParamsResolution,
    resolve_effective_params,
)
from data.repository import (
    get_active_ensemble_market_weights,
    get_active_ensemble_market_weights_source,
    load_best_ensemble_market_tuning_weights_by_optimized_metric,
)

logger = logging.getLogger(__name__)


# Default model allowlists per market (can be overridden by ensemble config)
DEFAULT_MARKET_MODELS: dict[str, list[str]] = {
    "ML": ["elo", "bradley-terry"],
    "SPREAD": ["elo", "gssd", "toor"],
    "TOTAL": ["poisson", "gssd", "toor"],
}

DEFAULT_MARKET_METRICS: dict[str, str] = {
    "ML": "log_loss",
    "SPREAD": "mae_margin",
    "TOTAL": "mae_total",
}

DEFAULT_ENSEMBLE_IDS: dict[str, str] = {
    "ML": "ensemble_ml_v1",
    "SPREAD": "ensemble_spread_v1",
    "TOTAL": "ensemble_total_v1",
}

VALID_MARKETS = {"ML", "SPREAD", "TOTAL"}


@dataclass(frozen=True)
class MarketParamsResolution:
    """Result of resolving parameters for a specific (model, market) pair.
    
    This dataclass captures everything needed to audit the provenance of
    model parameters used for a market-specific prediction.
    """
    model: str
    market: str
    params: dict[str, Any] | None
    params_source_label: str
    """One of: 'tuned_active', 'db_market_active', 'db_market_best_run', 
    'default_active', 'missing_active', 'legacy_active', 'cli', 'file'"""
    metric_optimized: str | None
    """The metric this param set was optimized for, e.g., 'backtest_mae_total'"""
    source_run_id: str | None
    """The tuning run ID that produced these params, if any"""
    best_score: float | None
    """The best score achieved during tuning, if known"""
    params_fingerprint: str
    """SHA256 hash of the params dict for deduplication"""
    params_nonempty: bool
    """True if params dict has at least one key"""
    
    def is_market_optimized(self) -> bool:
        """Return True if these params were explicitly tuned for this market."""
        return self.params_source_label in {
            "tuned_active", 
            "db_market_active", 
            "db_market_best_run",
        } and self.metric_optimized is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "market": self.market,
            "params": self.params,
            "params_source_label": self.params_source_label,
            "metric_optimized": self.metric_optimized,
            "source_run_id": self.source_run_id,
            "best_score": self.best_score,
            "params_fingerprint": self.params_fingerprint,
            "params_nonempty": self.params_nonempty,
            "is_market_optimized": self.is_market_optimized(),
        }


@dataclass(frozen=True)
class MarketEnsembleSpec:
    """Ensemble configuration for a specific market.
    
    This captures the models, weights, and metric used for combining
    predictions in a market-specific ensemble.
    """
    market: str
    ensemble_id: str
    models: list[str]
    weights: dict[str, float] | None
    metric_slot: str
    """Primary metric for this market: 'log_loss', 'mae_margin', or 'mae_total'"""
    weights_source: str
    """One of: 'db_active', 'db_best_run', 'config_file', 'default'"""
    source_run_id: str | None
    config_path: str | None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "ensemble_id": self.ensemble_id,
            "models": self.models,
            "weights": self.weights,
            "metric_slot": self.metric_slot,
            "weights_source": self.weights_source,
            "source_run_id": self.source_run_id,
            "config_path": self.config_path,
        }


@dataclass
class MarketForecastDiagnostics:
    """Diagnostic information for auditing market-specific forecasts."""
    market: str
    game_id: str
    model_resolutions: list[MarketParamsResolution] = field(default_factory=list)
    ensemble_spec: MarketEnsembleSpec | None = None
    warnings: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "game_id": self.game_id,
            "model_resolutions": [r.to_dict() for r in self.model_resolutions],
            "ensemble_spec": self.ensemble_spec.to_dict() if self.ensemble_spec else None,
            "warnings": self.warnings,
        }


def _normalize_market(market: str | Market) -> str:
    """Normalize market to uppercase string."""
    if isinstance(market, Market):
        return market.name
    normalized = str(market).strip().upper()
    if normalized == "SPREAD":
        return "SPREAD"
    if normalized not in VALID_MARKETS:
        raise ValueError(f"Invalid market: {market}. Must be one of {VALID_MARKETS}")
    return normalized


def resolve_market_params(
    *,
    db_path: str | Path | None,
    sport: str | None,
    season: str | None,
    model: str,
    market: str | Market,
) -> MarketParamsResolution:
    """Resolve model parameters for a specific market.
    
    This is the CANONICAL resolver for market-specific model parameters.
    It enforces the market-specific optimization contract by:
    1. Only looking up params keyed by (sport, season, model, market)
    2. Never falling back to params from a different market
    3. Returning clear provenance labels for audit
    
    Args:
        db_path: Path to the SQLite database
        sport: Sport identifier (e.g., 'nba')
        season: Season identifier (e.g., '2025-26')
        model: Model name (e.g., 'elo', 'bradley-terry')
        market: Market to resolve params for (ML, SPREAD, TOTAL)
    
    Returns:
        MarketParamsResolution with params and full provenance metadata
    """
    model_name = normalize_model_name(model)
    market_name = _normalize_market(market)
    
    # Delegate to the existing effective params resolver
    effective = resolve_effective_params(
        db_path=db_path,
        sport=sport,
        season=season,
        model=model_name,
        market=market_name,
    )
    
    logger.debug(
        "resolve_market_params: model=%s market=%s -> source=%s metric=%s",
        model_name,
        market_name,
        effective.params_source_label,
        effective.metric_optimized,
    )
    
    return MarketParamsResolution(
        model=model_name,
        market=market_name,
        params=effective.params,
        params_source_label=effective.params_source_label,
        metric_optimized=effective.metric_optimized,
        source_run_id=effective.source_run_id,
        best_score=effective.best_score,
        params_fingerprint=effective.params_fingerprint,
        params_nonempty=effective.params_nonempty,
    )


def resolve_market_params_batch(
    *,
    db_path: str | Path | None,
    sport: str | None,
    season: str | None,
    models: list[str],
    market: str | Market,
) -> dict[str, MarketParamsResolution]:
    """Resolve params for multiple models in a single market.
    
    Returns a dict mapping model name -> MarketParamsResolution.
    """
    market_name = _normalize_market(market)
    results: dict[str, MarketParamsResolution] = {}
    
    for model in models:
        model_name = normalize_model_name(model)
        results[model_name] = resolve_market_params(
            db_path=db_path,
            sport=sport,
            season=season,
            model=model_name,
            market=market_name,
        )
    
    return results


def get_market_ensemble_spec(
    *,
    db_path: str | Path | None,
    sport: str,
    season: str,
    market: str | Market,
    ensemble_config: dict | None = None,
) -> MarketEnsembleSpec:
    """Get the ensemble specification for a market.
    
    Resolution order:
    1. ensemble_config override (if provided and has this market)
    2. Active ensemble weights in DB
    3. Best ensemble tuning run in DB  
    4. Ensemble config file (outputs/ensembles/<sport>/<season>/<market>/)
    5. Default config
    
    Args:
        db_path: Path to the SQLite database
        sport: Sport identifier
        season: Season identifier
        market: Market to get ensemble for
        ensemble_config: Optional pre-loaded ensemble config dict
        
    Returns:
        MarketEnsembleSpec with models, weights, and provenance
    """
    market_name = _normalize_market(market)
    default_ensemble_id = DEFAULT_ENSEMBLE_IDS.get(market_name, f"ensemble_{market_name.lower()}_v1")
    default_metric = DEFAULT_MARKET_METRICS.get(market_name, "mae_total")
    default_models = DEFAULT_MARKET_MODELS.get(market_name, [])
    
    # 1. Check override config
    if ensemble_config:
        markets = ensemble_config.get("markets", {})
        market_cfg = markets.get(market_name, {})
        if market_cfg:
            models = market_cfg.get("models", default_models)
            weights = market_cfg.get("weights")
            ensemble_id = market_cfg.get("ensemble_id", default_ensemble_id)
            metric_slot = market_cfg.get("metric_slot", default_metric)
            meta = ensemble_config.get("_meta", {}).get("markets", {}).get(market_name, {})
            return MarketEnsembleSpec(
                market=market_name,
                ensemble_id=ensemble_id,
                models=models,
                weights=weights,
                metric_slot=metric_slot,
                weights_source="config_override" if weights else "config_models_only",
                source_run_id=None,
                config_path=meta.get("path"),
            )
    
    # 2. Check active ensemble weights in DB
    if db_path is not None:
        active_weights = get_active_ensemble_market_weights(
            db_path,
            sport=sport,
            season=season,
            market=market_name,
            ensemble_id=default_ensemble_id,
        )
        if active_weights is not None:
            source_run = get_active_ensemble_market_weights_source(
                db_path,
                sport=sport,
                season=season,
                market=market_name,
                ensemble_id=default_ensemble_id,
            )
            models = list(active_weights.keys())
            return MarketEnsembleSpec(
                market=market_name,
                ensemble_id=default_ensemble_id,
                models=models,
                weights=active_weights,
                metric_slot=default_metric,
                weights_source="db_active",
                source_run_id=source_run,
                config_path=None,
            )
        
        # 3. Check best ensemble tuning run
        _, metric_optimized = _resolve_market_metric(market_name, None)
        best_weights, best_run_id = load_best_ensemble_market_tuning_weights_by_optimized_metric(
            db_path,
            sport=sport,
            season=season,
            market=market_name,
            ensemble_id=default_ensemble_id,
            metric_optimized=metric_optimized,
        )
        if best_weights is not None:
            models = list(best_weights.keys())
            return MarketEnsembleSpec(
                market=market_name,
                ensemble_id=default_ensemble_id,
                models=models,
                weights=best_weights,
                metric_slot=default_metric,
                weights_source="db_best_run",
                source_run_id=best_run_id,
                config_path=None,
            )
    
    # 4. Try loading from config file
    try:
        from ensemble.config import load_ensemble_config
        file_config = load_ensemble_config(
            sport,
            season,
            available_models=list_models(),
        )
        if file_config:
            markets = file_config.get("markets", {})
            market_cfg = markets.get(market_name, {})
            if market_cfg:
                models = market_cfg.get("models", default_models)
                weights = market_cfg.get("weights")
                ensemble_id = market_cfg.get("ensemble_id", default_ensemble_id)
                metric_slot = market_cfg.get("metric_slot", default_metric)
                meta = file_config.get("_meta", {}).get("markets", {}).get(market_name, {})
                return MarketEnsembleSpec(
                    market=market_name,
                    ensemble_id=ensemble_id,
                    models=models,
                    weights=weights,
                    metric_slot=metric_slot,
                    weights_source="config_file" if meta.get("path") else "default",
                    source_run_id=None,
                    config_path=meta.get("path"),
                )
    except Exception as e:
        logger.debug("Failed to load ensemble config from file: %s", e)
    
    # 5. Return defaults
    return MarketEnsembleSpec(
        market=market_name,
        ensemble_id=default_ensemble_id,
        models=default_models,
        weights=None,  # Will use equal weights
        metric_slot=default_metric,
        weights_source="default",
        source_run_id=None,
        config_path=None,
    )


def get_all_market_specs(
    *,
    db_path: str | Path | None,
    sport: str,
    season: str,
    ensemble_config: dict | None = None,
) -> dict[str, MarketEnsembleSpec]:
    """Get ensemble specs for all markets.
    
    Returns dict mapping market name -> MarketEnsembleSpec.
    """
    return {
        market: get_market_ensemble_spec(
            db_path=db_path,
            sport=sport,
            season=season,
            market=market,
            ensemble_config=ensemble_config,
        )
        for market in VALID_MARKETS
    }


def validate_market_isolation(
    *,
    db_path: str | Path | None,
    sport: str,
    season: str,
    models: list[str],
) -> dict[str, list[str]]:
    """Validate that each market has properly isolated params.
    
    Returns dict with any warnings/errors found. Empty dict means all clear.
    """
    issues: dict[str, list[str]] = {}
    
    for market in VALID_MARKETS:
        market_issues: list[str] = []
        
        for model in models:
            resolution = resolve_market_params(
                db_path=db_path,
                sport=sport,
                season=season,
                model=model,
                market=market,
            )
            
            # Check if params exist
            if resolution.params_source_label == "missing_active":
                market_issues.append(
                    f"Model {model} has no active params for {market}; using defaults"
                )
            
            # Check if the metric matches the market
            if resolution.metric_optimized:
                expected_prefix = f"backtest_{DEFAULT_MARKET_METRICS[market]}"
                if not resolution.metric_optimized.endswith(DEFAULT_MARKET_METRICS[market]):
                    market_issues.append(
                        f"Model {model} in {market} has metric_optimized={resolution.metric_optimized} "
                        f"but expected metric for {market} is {expected_prefix}"
                    )
        
        if market_issues:
            issues[market] = market_issues
    
    return issues


def log_market_params_summary(
    *,
    db_path: str | Path | None,
    sport: str,
    season: str,
    models: list[str],
    level: int = logging.INFO,
) -> None:
    """Log a summary of market params for debugging."""
    for market in VALID_MARKETS:
        logger.log(level, "=== Market: %s ===", market)
        for model in models:
            resolution = resolve_market_params(
                db_path=db_path,
                sport=sport,
                season=season,
                model=model,
                market=market,
            )
            logger.log(
                level,
                "  %s: source=%s, metric=%s, run_id=%s, nonempty=%s",
                model,
                resolution.params_source_label,
                resolution.metric_optimized,
                resolution.source_run_id,
                resolution.params_nonempty,
            )
        
        spec = get_market_ensemble_spec(
            db_path=db_path,
            sport=sport,
            season=season,
            market=market,
        )
        logger.log(
            level,
            "  Ensemble: id=%s, models=%s, weights_source=%s",
            spec.ensemble_id,
            spec.models,
            spec.weights_source,
        )


def collect_game_diagnostics(
    *,
    db_path: str | Path | None,
    sport: str,
    season: str,
    game_id: str,
    models: list[str],
    ensemble_config: dict | None = None,
) -> dict[str, MarketForecastDiagnostics]:
    """Collect diagnostic information for a single game across all markets.
    
    This is the primary debugging tool for understanding which params/models
    were used for a specific game's predictions.
    
    Returns dict mapping market -> MarketForecastDiagnostics.
    """
    diagnostics: dict[str, MarketForecastDiagnostics] = {}
    
    for market in VALID_MARKETS:
        diag = MarketForecastDiagnostics(market=market, game_id=game_id)
        
        # Get ensemble spec
        spec = get_market_ensemble_spec(
            db_path=db_path,
            sport=sport,
            season=season,
            market=market,
            ensemble_config=ensemble_config,
        )
        diag.ensemble_spec = spec
        
        # Resolve params for each model in this market's ensemble
        market_models = spec.models if spec.models else models
        for model in market_models:
            resolution = resolve_market_params(
                db_path=db_path,
                sport=sport,
                season=season,
                model=model,
                market=market,
            )
            diag.model_resolutions.append(resolution)
            
            # Add warnings for potential issues
            if not resolution.is_market_optimized():
                diag.warnings.append(
                    f"Model {model} params not market-optimized "
                    f"(source={resolution.params_source_label})"
                )
            
            if resolution.metric_optimized:
                expected_metric = DEFAULT_MARKET_METRICS.get(market, "")
                if expected_metric and expected_metric not in resolution.metric_optimized:
                    diag.warnings.append(
                        f"Model {model} metric mismatch: {resolution.metric_optimized} "
                        f"vs expected {expected_metric}"
                    )
        
        diagnostics[market] = diag
    
    return diagnostics


def format_diagnostics_report(
    diagnostics: dict[str, MarketForecastDiagnostics],
) -> str:
    """Format diagnostics as a human-readable report."""
    lines: list[str] = []
    
    for market, diag in sorted(diagnostics.items()):
        lines.append(f"\n{'='*60}")
        lines.append(f"MARKET: {market} | GAME: {diag.game_id}")
        lines.append(f"{'='*60}")
        
        if diag.ensemble_spec:
            spec = diag.ensemble_spec
            lines.append(f"\nEnsemble: {spec.ensemble_id}")
            lines.append(f"  Models: {spec.models}")
            lines.append(f"  Weights: {spec.weights}")
            lines.append(f"  Metric: {spec.metric_slot}")
            lines.append(f"  Weights Source: {spec.weights_source}")
            if spec.source_run_id:
                lines.append(f"  Run ID: {spec.source_run_id}")
            if spec.config_path:
                lines.append(f"  Config: {spec.config_path}")
        
        lines.append("\nModel Parameters:")
        for res in diag.model_resolutions:
            opt_marker = "✓" if res.is_market_optimized() else "✗"
            lines.append(f"  [{opt_marker}] {res.model}:")
            lines.append(f"      Source: {res.params_source_label}")
            lines.append(f"      Metric: {res.metric_optimized}")
            lines.append(f"      Run ID: {res.source_run_id or 'N/A'}")
            lines.append(f"      Non-empty: {res.params_nonempty}")
            if res.params:
                param_str = json.dumps(res.params, default=str)
                if len(param_str) > 60:
                    param_str = param_str[:57] + "..."
                lines.append(f"      Params: {param_str}")
        
        if diag.warnings:
            lines.append("\nWarnings:")
            for warn in diag.warnings:
                lines.append(f"  ⚠ {warn}")
    
    return "\n".join(lines)


def print_market_diagnostics(
    *,
    db_path: str | Path | None,
    sport: str,
    season: str,
    game_id: str,
    models: list[str] | None = None,
    ensemble_config: dict | None = None,
) -> None:
    """Print diagnostics for a game to stdout (for CLI debugging)."""
    from models.registry import list_models as get_all_models
    
    if models is None:
        models = get_all_models()
    
    diagnostics = collect_game_diagnostics(
        db_path=db_path,
        sport=sport,
        season=season,
        game_id=game_id,
        models=models,
        ensemble_config=ensemble_config,
    )
    
    report = format_diagnostics_report(diagnostics)
    print(report)
