"""Tests for Poisson model OT home-win share logic."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from models.ot_share import get_ot_home_win_share, clear_ot_share_cache
from models.poisson import (
    _skellam_home_win_prob,
    poisson_canonical_from_rates,
    poisson_canonical_from_samples,
)
import numpy as np


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear OT share cache before each test."""
    clear_ot_share_cache()
    yield
    clear_ot_share_cache()


def test_ot_share_returns_fallback_for_non_nhl(tmp_path):
    """Non-NHL sports should always return 0.5."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE games (id INT)")
    conn.close()

    share, source = get_ot_home_win_share(str(db_path), "nba", "2025-26")
    assert share == 0.5
    assert source == "default_non_nhl"


def test_ot_share_fallback_for_missing_db(tmp_path):
    """Missing database should return fallback."""
    db_path = tmp_path / "nonexistent.db"

    share, source = get_ot_home_win_share(str(db_path), "nhl", "2025-26")
    assert share == 0.5
    assert source == "fallback_db_not_found"


def test_ot_share_fallback_insufficient_data(tmp_path):
    """NHL with <20 OT games should return fallback."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE games (
            overtime INTEGER,
            home_score INTEGER,
            away_score INTEGER
        );
    """)
    # Insert only 10 OT games (below threshold of 20)
    for i in range(6):
        conn.execute("INSERT INTO games VALUES (1, 3, 2)")  # home wins
    for i in range(4):
        conn.execute("INSERT INTO games VALUES (1, 2, 3)")  # away wins
    conn.commit()
    conn.close()

    share, source = get_ot_home_win_share(str(db_path), "nhl", "2025-26")
    assert share == 0.5
    assert source == "fallback_insufficient_data"


def test_ot_share_computes_empirical_for_nhl(tmp_path):
    """NHL with sufficient OT games should return empirical share."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE games (
            overtime INTEGER,
            home_score INTEGER,
            away_score INTEGER
        );
    """)
    # Insert 25 OT games: 15 home wins, 10 away wins -> 60% home win rate
    for i in range(15):
        conn.execute("INSERT INTO games VALUES (1, 3, 2)")  # home wins
    for i in range(10):
        conn.execute("INSERT INTO games VALUES (1, 2, 3)")  # away wins
    conn.commit()
    conn.close()

    share, source = get_ot_home_win_share(str(db_path), "nhl", "2025-26")
    assert share == pytest.approx(0.6, rel=1e-6)
    assert source == "empirical_ot"


def test_ot_share_ignores_non_ot_games(tmp_path):
    """Only games with overtime=1 should be counted."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE games (
            overtime INTEGER,
            home_score INTEGER,
            away_score INTEGER
        );
    """)
    # 25 OT games (15-10 split) + 100 regulation games (all away wins)
    for i in range(15):
        conn.execute("INSERT INTO games VALUES (1, 3, 2)")  # OT home wins
    for i in range(10):
        conn.execute("INSERT INTO games VALUES (1, 2, 3)")  # OT away wins
    for i in range(100):
        conn.execute("INSERT INTO games VALUES (0, 2, 3)")  # regulation away wins
    conn.commit()
    conn.close()

    share, source = get_ot_home_win_share(str(db_path), "nhl", "2025-26")
    # Should only count the 25 OT games, not the 100 regulation games
    assert share == pytest.approx(0.6, rel=1e-6)
    assert source == "empirical_ot"


def test_ot_share_caching(tmp_path):
    """Results should be cached per (db_path, sport, season)."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE games (
            overtime INTEGER,
            home_score INTEGER,
            away_score INTEGER
        );
    """)
    for i in range(25):
        conn.execute("INSERT INTO games VALUES (1, 3, 2)")  # all home wins
    conn.commit()
    conn.close()

    # First call
    share1, source1 = get_ot_home_win_share(str(db_path), "nhl", "2025-26")
    assert share1 == 1.0
    assert source1 == "empirical_ot"

    # Add more games to database
    conn = sqlite3.connect(db_path)
    for i in range(25):
        conn.execute("INSERT INTO games VALUES (1, 2, 3)")  # all away wins
    conn.commit()
    conn.close()

    # Second call should return cached result (still 1.0)
    share2, source2 = get_ot_home_win_share(str(db_path), "nhl", "2025-26")
    assert share2 == 1.0  # Cached, not updated
    assert source2 == "empirical_ot"

    # Clear cache and try again
    clear_ot_share_cache()
    share3, source3 = get_ot_home_win_share(str(db_path), "nhl", "2025-26")
    assert share3 == pytest.approx(0.5, rel=1e-6)  # Now reflects 25/50 home wins
    assert source3 == "empirical_ot"


def test_skellam_uses_tie_split_parameter():
    """Verify _skellam_home_win_prob respects tie_split_home parameter."""
    # With equal rates (lambda=3 for both), there's significant tie probability
    # p_home_win = P(H > A) + tie_split_home * P(H = A)
    p_default = _skellam_home_win_prob(3.0, 3.0, fallback_sd=1.5, tie_split_home=0.5)
    p_home_biased = _skellam_home_win_prob(3.0, 3.0, fallback_sd=1.5, tie_split_home=0.6)
    p_away_biased = _skellam_home_win_prob(3.0, 3.0, fallback_sd=1.5, tie_split_home=0.4)

    # Default should be ~0.5 (equal rates, 50-50 tie split)
    assert p_default == pytest.approx(0.5, rel=0.01)

    # Home-biased should be higher than default
    assert p_home_biased > p_default

    # Away-biased should be lower than default
    assert p_away_biased < p_default


def test_canonical_from_rates_includes_tie_split():
    """Verify output includes tie_split_home for traceability."""
    result = poisson_canonical_from_rates(3.0, 2.5, tie_split_home=0.55)
    assert "tie_split_home" in result
    assert result["tie_split_home"] == pytest.approx(0.55)


def test_canonical_from_samples_includes_tie_split():
    """Verify output includes tie_split_home for traceability."""
    rng = np.random.default_rng(42)
    home_samples = rng.poisson(3.0, size=1000)
    away_samples = rng.poisson(2.5, size=1000)

    result = poisson_canonical_from_samples(home_samples, away_samples, tie_split_home=0.55)
    assert "tie_split_home" in result
    assert result["tie_split_home"] == pytest.approx(0.55)


def test_canonical_from_samples_uses_tie_split():
    """Verify samples-based probability computation uses tie_split_home."""
    # Create samples with many ties
    rng = np.random.default_rng(42)
    home_samples = rng.poisson(3.0, size=10000)
    away_samples = rng.poisson(3.0, size=10000)  # Same rate = many ties

    result_default = poisson_canonical_from_samples(home_samples, away_samples, tie_split_home=0.5)
    result_home_biased = poisson_canonical_from_samples(home_samples, away_samples, tie_split_home=0.7)

    # With home bias on tie split, p_home_win should be higher
    assert result_home_biased["p_home_win"] > result_default["p_home_win"]


def test_canonical_from_rates_uses_tie_split():
    """Verify rates-based probability computation uses tie_split_home."""
    # Equal rates = significant tie probability from Skellam
    result_default = poisson_canonical_from_rates(3.0, 3.0, tie_split_home=0.5)
    result_home_biased = poisson_canonical_from_rates(3.0, 3.0, tie_split_home=0.7)

    # With home bias on tie split, p_home_win should be higher
    assert result_home_biased["p_home_win"] > result_default["p_home_win"]

    # Both should be close to 0.5 but differ due to tie split
    assert result_default["p_home_win"] == pytest.approx(0.5, rel=0.01)
