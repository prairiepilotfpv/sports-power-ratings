from __future__ import annotations

from typing import Dict

from .base import Market, MarketSpec


_SPECS: Dict[Market, MarketSpec] = {
    Market.ML: MarketSpec(
        market=Market.ML, required_fields=["p_home_win"], primary_metric_name="p_home_win"
    ),
    Market.SPREAD: MarketSpec(
        market=Market.SPREAD,
        required_fields=["margin_mean", "margin_sd"],
        primary_metric_name="margin_mean",
    ),
    Market.TOTAL: MarketSpec(
        market=Market.TOTAL,
        required_fields=["total_mean", "total_sd"],
        primary_metric_name="total_mean",
    ),
}


def get_market_spec(market: Market | str) -> MarketSpec:
    if isinstance(market, str):
        # allow case-insensitive names or enum value
        norm = market.strip()
        for m in Market:
            if norm.lower() == m.value.lower() or norm.lower() == m.name.lower():
                return _SPECS[m]
        raise KeyError(f"Unknown market: {market}")
    return _SPECS[market]
