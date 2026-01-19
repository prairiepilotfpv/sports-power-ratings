"""Shared utilities for market-specific tuning helpers."""

from __future__ import annotations


def _metric_optimized_label(market: str) -> str:
    if market == "ML":
        return "backtest_log_loss"
    if market == "SPREAD":
        return "backtest_mae_margin"
    return "backtest_mae_total"


def _metric_name_for_market(market: str) -> str:
    if market == "ML":
        return "log_loss"
    if market == "SPREAD":
        return "mae_margin"
    return "mae_total"


def _resolve_market_metric(market: str, metric_override: str | None) -> tuple[str, str]:
    if metric_override:
        metric = metric_override.strip().lower()
        if metric.startswith("backtest_"):
            metric = metric.replace("backtest_", "", 1)
        if metric not in {"log_loss", "brier_score", "mae_margin", "mae_total"}:
            raise ValueError(f"Unsupported metric override: {metric_override}")
        optimized = f"backtest_{metric}"
        return metric, optimized
    return _metric_name_for_market(market), _metric_optimized_label(market)
