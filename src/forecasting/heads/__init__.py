"""Projection heads framework for explicit model outputs."""

from forecasting.heads.base import Head, HeadSequence
from forecasting.heads.registry import (
    apply_heads,
    get_model_heads,
    register_model_heads,
)
from forecasting.heads.elo_heads import create_elo_head_sequence
from forecasting.heads.toor_heads import create_toor_head_sequence
from forecasting.heads.gssd_heads import create_gssd_head_sequence
from forecasting.heads.bradley_terry_heads import create_bradley_terry_head_sequence
from forecasting.heads.poisson_heads import create_poisson_head_sequence

__all__ = [
    "Head",
    "HeadSequence",
    "register_model_heads",
    "get_model_heads",
    "apply_heads",
    "create_elo_head_sequence",
    "create_toor_head_sequence",
    "create_gssd_head_sequence",
    "create_bradley_terry_head_sequence",
    "create_poisson_head_sequence",
]
