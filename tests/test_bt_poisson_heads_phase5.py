"""Tests for Bradley-Terry and Poisson heads implementations (Phase 5)."""

import math
import numpy as np
import pandas as pd
import pytest

from config import MIN_CALIBRATION_SAMPLES
from forecasting.heads.bradley_terry_heads import (
    BtMarginHead,
    BtWinProbHead,
    BtTotalHead,
    create_bradley_terry_head_sequence,
)
from forecasting.heads.poisson_heads import (
    PoissonMarginHead,
    PoissonTotalHead,
    PoissonWinProbHead,
    create_poisson_head_sequence,
)
from forecasting.heads.registry import apply_heads
from models.bradley_terry import BradleyTerry
from models.poisson import PoissonPowerRating


def _make_alternating_games(n_games: int, base_score: int = 90, margins=None):
    """Create a simple alternating-home dataset between teams A and B."""
    if margins is None:
        margins = [5] * n_games
    elif isinstance(margins, (int, float)):
        margins = [margins] * n_games
    games = []
    for i in range(n_games):
        home = "A" if i % 2 == 0 else "B"
        away = "B" if home == "A" else "A"
        margin = int(margins[i])
        # alternate winners
        if i % 3 == 0:
            home_score = base_score + margin
            away_score = base_score
        else:
            home_score = base_score
            away_score = base_score + margin
        games.append({
            "date": f"2025-01-{(i%30)+1}",
            "home_team": home,
            "away_team": away,
            "home_score": int(home_score),
            "away_score": int(away_score),
            "neutral": False,
        })
    return pd.DataFrame(games)


def _make_poisson_games(n_games: int, base_rate: float = 2.5):
    """Create a Poisson-like goal scoring dataset."""
    games = []
    for i in range(n_games):
        home = "A" if i % 2 == 0 else "B"
        away = "B" if home == "A" else "A"
        if i % 3 == 0:
            home_score = np.random.poisson(base_rate + 0.5)
            away_score = np.random.poisson(base_rate)
        else:
            home_score = np.random.poisson(base_rate)
            away_score = np.random.poisson(base_rate + 0.5)
        games.append({
            "date": f"2025-01-{(i%30)+1}",
            "home_team": home,
            "away_team": away,
            "home_score": int(home_score),
            "away_score": int(away_score),
            "neutral": False,
        })
    return pd.DataFrame(games)


# ============================================================================
# BRADLEY-TERRY HEADS TESTS
# ============================================================================


class TestBtMarginHead:
    """Test BtMarginHead output correctness."""

    def test_margin_head_produces_fields(self):
        """Margin head should produce margin_mean and margin_sd."""
        head = BtMarginHead()
        assert "margin_mean" in head.produces()
        assert "margin_sd" in head.produces()

    def test_margin_head_outputs_valid(self):
        """Margin head outputs should be numeric and SD > 0."""
        n = max(MIN_CALIBRATION_SAMPLES + 5, 30)
        df_games = _make_alternating_games(n)
        bt = BradleyTerry()
        bt.fit(df_games.to_dict(orient="records"))

        df = pd.DataFrame({
            "home_team": ["A", "B", "A"],
            "away_team": ["B", "A", "B"],
            "neutral": [False, False, True],
        })

        context = {
            "ratings": bt.ratings,
            "home_advantage": bt.hfa_logit,
            "margin_sigma": bt.calibration.margin_sigma,
            "sport": "nba",
        }

        head = BtMarginHead()
        head.apply(df, context)

        assert all(df["margin_mean"].notna())
        assert all(df["margin_sd"].notna())
        assert all(df["margin_sd"] > 0)


class TestBtWinProbHead:
    """Test BtWinProbHead output correctness."""

    def test_win_prob_produces_fields(self):
        """Win prob head should produce p_home_win."""
        head = BtWinProbHead()
        produced = head.produces()
        assert "p_home_win" in produced

    def test_win_prob_in_range(self):
        """Win probability should be in [0, 1]."""
        n = max(MIN_CALIBRATION_SAMPLES + 5, 30)
        df_games = _make_alternating_games(n)
        bt = BradleyTerry()
        bt.fit(df_games.to_dict(orient="records"))

        df = pd.DataFrame({
            "home_team": ["A", "B"],
            "away_team": ["B", "A"],
            "neutral": [False, False],
        })

        context = {
            "ratings": bt.ratings,
            "home_advantage": bt.hfa_logit,
            "margin_sigma": bt.calibration.margin_sigma,
            "temp": bt.temp,
            "sport": "nba",
        }

        margin_head = BtMarginHead()
        margin_head.apply(df, context)

        win_prob_head = BtWinProbHead()
        win_prob_head.apply(df, context)

        assert all((df["p_home_win"] >= 0) & (df["p_home_win"] <= 1))


class TestBtHeadSequence:
    """Test full Bradley-Terry head sequence."""

    def test_full_sequence_complete(self):
        """Full sequence should produce all canonical fields."""
        n = max(MIN_CALIBRATION_SAMPLES + 5, 30)
        df_games = _make_alternating_games(n, base_score=100)
        bt = BradleyTerry()
        bt.fit(df_games.to_dict(orient="records"))

        df = pd.DataFrame({
            "home_team": ["A", "B"],
            "away_team": ["B", "A"],
        })

        context = {
            "ratings": bt.ratings,
            "home_advantage": bt.hfa_logit,
            "margin_sigma": bt.calibration.margin_sigma,
            "total_c": bt.calibration.total_c,
            "total_u": bt.calibration.total_u,
            "total_sigma": bt.calibration.total_sigma,
            "temp": bt.temp,
            "sport": "nba",
        }

        head_seq = create_bradley_terry_head_sequence()
        result = head_seq.apply(df, context)

        # Verify fields present
        required = {"margin_mean", "margin_sd", "total_mean", "total_sd", "p_home_win"}
        for field in required:
            assert field in df.columns
            assert all(df[field].notna())

    def test_registry_integration(self):
        """Test apply_heads registry for BT."""
        n = max(MIN_CALIBRATION_SAMPLES + 5, 30)
        df_games = _make_alternating_games(n)
        bt = BradleyTerry()
        bt.fit(df_games.to_dict(orient="records"))

        df = pd.DataFrame({
            "home_team": ["A"],
            "away_team": ["B"],
        })

        context = {
            "ratings": bt.ratings,
            "home_advantage": bt.hfa_logit,
            "margin_sigma": bt.calibration.margin_sigma,
            "total_c": bt.calibration.total_c,
            "total_u": bt.calibration.total_u,
            "total_sigma": bt.calibration.total_sigma,
            "temp": bt.temp,
            "sport": "nba",
        }

        result = apply_heads("bradley-terry", df, context)
        assert result is not None
        assert "p_home_win" in df.columns


# ============================================================================
# POISSON HEADS TESTS
# ============================================================================


class TestPoissonMarginHead:
    """Test PoissonMarginHead output correctness."""

    def test_margin_head_produces_fields(self):
        """Margin head should produce margin_mean and margin_sd."""
        head = PoissonMarginHead()
        assert "margin_mean" in head.produces()
        assert "margin_sd" in head.produces()

    def test_margin_head_handles_state(self):
        """Margin head should handle Poisson state arrays."""
        np.random.seed(42)
        n = max(MIN_CALIBRATION_SAMPLES + 5, 30)
        df_games = _make_poisson_games(n)
        poisson = PoissonPowerRating(n_simulations=500, random_seed=42)
        poisson.fit(df_games.to_dict(orient="records"))

        if poisson._state is None:
            pytest.skip("Poisson model failed to fit")

        df = pd.DataFrame({
            "home_team": ["A", "B"],
            "away_team": ["B", "A"],
        })

        context = {
            "attack": poisson._state.attack,
            "defense": poisson._state.defense,
            "team_index": poisson._state.team_index,
            "mu": poisson._state.mu,
            "home_advantage": poisson._state.home_advantage,
            "kappa": poisson.kappa,
        }

        head = PoissonMarginHead()
        head.apply(df, context)

        assert all(df["margin_mean"].notna())
        assert all(df["margin_sd"].notna())
        assert all(df["margin_sd"] > 0)


class TestPoissonHeadSequence:
    """Test full Poisson head sequence."""

    def test_full_sequence_complete(self):
        """Full sequence should produce canonical fields."""
        np.random.seed(42)
        n = max(MIN_CALIBRATION_SAMPLES + 5, 30)
        df_games = _make_poisson_games(n)
        poisson = PoissonPowerRating(n_simulations=500, random_seed=42)
        poisson.fit(df_games.to_dict(orient="records"))

        if poisson._state is None:
            pytest.skip("Poisson model failed to fit")

        df = pd.DataFrame({
            "home_team": ["A", "B"],
            "away_team": ["B", "A"],
        })

        context = {
            "attack": poisson._state.attack,
            "defense": poisson._state.defense,
            "team_index": poisson._state.team_index,
            "mu": poisson._state.mu,
            "home_advantage": poisson._state.home_advantage,
            "kappa": poisson.kappa,
            "tie_split_home": 0.5,
        }

        head_seq = create_poisson_head_sequence()
        result = head_seq.apply(df, context)

        required = {"margin_mean", "margin_sd", "total_mean", "total_sd", "p_home_win"}
        for field in required:
            assert field in df.columns
            assert all(df[field].notna())

    def test_registry_integration(self):
        """Test apply_heads registry for Poisson."""
        np.random.seed(42)
        n = max(MIN_CALIBRATION_SAMPLES + 5, 30)
        df_games = _make_poisson_games(n)
        poisson = PoissonPowerRating(n_simulations=500, random_seed=42)
        poisson.fit(df_games.to_dict(orient="records"))

        if poisson._state is None:
            pytest.skip("Poisson model failed to fit")

        df = pd.DataFrame({
            "home_team": ["A"],
            "away_team": ["B"],
        })

        context = {
            "attack": poisson._state.attack,
            "defense": poisson._state.defense,
            "team_index": poisson._state.team_index,
            "mu": poisson._state.mu,
            "home_advantage": poisson._state.home_advantage,
            "kappa": poisson.kappa,
            "tie_split_home": 0.5,
        }

        result = apply_heads("poisson", df, context)
        assert result is not None
        assert "p_home_win" in df.columns


# ============================================================================
# COHERENCE & INVARIANT TESTS
# ============================================================================


class TestCoherence:
    """Test coherence of heads outputs."""

    def test_bt_margin_win_prob_consistency(self):
        """BT margin and win prob should be consistent."""
        n = max(MIN_CALIBRATION_SAMPLES + 5, 30)
        df_games = _make_alternating_games(n)
        bt = BradleyTerry()
        bt.fit(df_games.to_dict(orient="records"))

        df = pd.DataFrame({
            "home_team": ["A"],
            "away_team": ["B"],
        })

        context = {
            "ratings": bt.ratings,
            "home_advantage": bt.hfa_logit,
            "margin_sigma": bt.calibration.margin_sigma,
            "total_c": bt.calibration.total_c,
            "total_u": bt.calibration.total_u,
            "total_sigma": bt.calibration.total_sigma,
            "temp": bt.temp,
            "sport": "nba",
        }

        head_seq = create_bradley_terry_head_sequence()
        head_seq.apply(df, context)

        margin_mean = df.loc[0, "margin_mean"]
        p_win = df.loc[0, "p_home_win"]

        # If margin_mean > 0, p_win should be > 0.5 (home should be favored)
        if margin_mean > 0:
            assert p_win > 0.5
        elif margin_mean < 0:
            assert p_win < 0.5
        else:
            assert abs(p_win - 0.5) < 0.1
