"""Ensemble weight resolution audit and governance.

This module provides:
- EnsembleAudit dataclass: tracks final weights, source, dropped models, and coverage
- Weight governance: enforces MIN_WEIGHT_EPS clamping and MIN_NEFF thresholds
- Audit logging: single INFO log per market with full audit trail
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

_LOG = logging.getLogger(__name__)


@dataclass(frozen=False)
class EnsembleAudit:
    """Audit object from weight resolution and governance.
    
    Attributes:
        market: Market name (ML, SPREAD, TOTAL).
        final_models: List of models in final ensemble (after filtering).
        final_weights: Dict of model -> weight (post-clamp, post-governance).
        weight_source: Source of weights (db_best_run, db_tuned, db_active, config, file, fallback).
        dropped_models: Dict of model -> reason_for_drop.
        coverage_summary: Dict of model -> description of coverage (games covered, required columns).
        neff: Effective model count (1 / sum(w_i^2)).
        neff_threshold_met: Whether Neff >= MIN_NEFF.
        weight_run_id: Optional run ID from tuning/DB source.
        selection_run_id: Optional run ID from selection source.
        weight_clamped: Whether any weights were clamped to 0.
        fallback_applied: Whether fallback to uniform weights was applied.
        warnings: List of warnings emitted during resolution.
    """

    market: str
    final_models: list[str] = field(default_factory=list)
    final_weights: dict[str, float] = field(default_factory=dict)
    weight_source: str = ""
    dropped_models: dict[str, str] = field(default_factory=dict)
    coverage_summary: dict[str, str] = field(default_factory=dict)
    neff: float = 0.0
    neff_threshold_met: bool = True
    weight_run_id: Optional[str] = None
    selection_run_id: Optional[str] = None
    weight_clamped: bool = False
    fallback_applied: bool = False
    warnings: list[str] = field(default_factory=list)

    def calculate_neff(self) -> float:
        """Calculate effective model count: Neff = 1 / sum(w_i^2)."""
        if not self.final_weights:
            return 0.0
        sum_sq = sum(w ** 2 for w in self.final_weights.values())
        if sum_sq <= 0:
            return 0.0
        return 1.0 / sum_sq

    def to_dict(self) -> dict:
        """Serialize audit to dict."""
        return {
            "market": self.market,
            "final_models": self.final_models,
            "final_weights": self.final_weights,
            "weight_source": self.weight_source,
            "dropped_models": self.dropped_models,
            "coverage_summary": self.coverage_summary,
            "neff": round(self.neff, 2),
            "neff_threshold_met": self.neff_threshold_met,
            "weight_run_id": self.weight_run_id,
            "selection_run_id": self.selection_run_id,
            "weight_clamped": self.weight_clamped,
            "fallback_applied": self.fallback_applied,
            "warnings": self.warnings,
        }

    def emit_log(self) -> None:
        """Emit a single INFO log with full audit trail."""
        info_msg = (
            f"[ensemble audit][{self.market}] "
            f"source={self.weight_source} "
            f"Neff={self.neff:.2f} (threshold={'met' if self.neff_threshold_met else 'UNMET'}) "
            f"models={self.final_models} "
            f"weights={json.dumps(self.final_weights, sort_keys=True)} "
        )
        if self.dropped_models:
            info_msg += f"dropped={json.dumps(self.dropped_models, sort_keys=True)} "
        if self.weight_clamped:
            info_msg += "weight_clamped=true "
        if self.fallback_applied:
            info_msg += "fallback_applied=true "
        if self.warnings:
            info_msg += f"warnings={len(self.warnings)}"

        _LOG.info(info_msg)

        # Emit warnings at WARNING level
        for warning in self.warnings:
            _LOG.warning(f"[ensemble audit][{self.market}] {warning}")


def compute_neff(weights: dict[str, float]) -> float:
    """Compute effective model count: Neff = 1 / sum(w_i^2)."""
    if not weights:
        return 0.0
    sum_sq = sum(w ** 2 for w in weights.values())
    if sum_sq <= 0:
        return 0.0
    return 1.0 / sum_sq
