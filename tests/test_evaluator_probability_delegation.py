"""Tests for evaluator probability functions delegating to projections.py."""

from __future__ import annotations

import math

import pytest

from eval.evaluator import _cover_probability, _over_probability
from pipelines.projections import _normal_cdf, cover_prob, over_prob


class TestCoverProbabilityEquivalence:
    """Verify _cover_probability matches the old manual formula."""

    @staticmethod
    def _old_cover_probability(line: float, margin_mean: float, margin_sd: float, is_home: bool) -> float:
        """Original implementation for comparison."""
        threshold = -line if is_home else line
        if is_home:
            return 1.0 - _normal_cdf(threshold, mean=margin_mean, sd=margin_sd)
        return _normal_cdf(threshold, mean=margin_mean, sd=margin_sd)

    def test_home_selection_negative_line(self) -> None:
        """Home selection (is_home=True) with line=-5.5, margin_mean=+6, sd=12."""
        line = -5.5
        margin_mean = 6.0
        margin_sd = 12.0
        is_home = True

        old_result = self._old_cover_probability(line, margin_mean, margin_sd, is_home)
        new_result = _cover_probability(line, margin_mean, margin_sd, is_home)

        assert new_result is not None
        assert new_result == pytest.approx(old_result, rel=1e-9)
        # Home favored by 6 pts, needs to cover 5.5 → should be > 50%
        assert new_result > 0.5

    def test_away_selection_negative_line(self) -> None:
        """Away selection (is_home=False) with line=-5.5, margin_mean=+6, sd=12."""
        line = -5.5
        margin_mean = 6.0
        margin_sd = 12.0
        is_home = False

        old_result = self._old_cover_probability(line, margin_mean, margin_sd, is_home)
        new_result = _cover_probability(line, margin_mean, margin_sd, is_home)

        assert new_result is not None
        assert new_result == pytest.approx(old_result, rel=1e-9)
        # Away with line=-5.5 needs margin < -5.5 (away wins outright by >5.5)
        # Home favored by 6, so away covering -5.5 is rare
        assert new_result < 0.25

    def test_away_selection_positive_line(self) -> None:
        """Away selection (is_home=False) with line=+3.5, margin_mean=-2, sd=12."""
        line = 3.5
        margin_mean = -2.0  # Away favored
        margin_sd = 12.0
        is_home = False

        old_result = self._old_cover_probability(line, margin_mean, margin_sd, is_home)
        new_result = _cover_probability(line, margin_mean, margin_sd, is_home)

        assert new_result is not None
        assert new_result == pytest.approx(old_result, rel=1e-9)
        # Away favored by 2, gets 3.5 points → very likely to cover
        assert new_result > 0.6

    def test_home_selection_positive_line(self) -> None:
        """Home selection with positive line (home is underdog)."""
        line = 4.5
        margin_mean = -3.0  # Away favored by 3
        margin_sd = 10.0
        is_home = True

        old_result = self._old_cover_probability(line, margin_mean, margin_sd, is_home)
        new_result = _cover_probability(line, margin_mean, margin_sd, is_home)

        assert new_result is not None
        assert new_result == pytest.approx(old_result, rel=1e-9)

    def test_returns_none_for_none_sd(self) -> None:
        """Should return None when sd is None (no exception)."""
        result = _cover_probability(-5.5, 6.0, None, True)
        assert result is None

    def test_returns_none_for_zero_sd(self) -> None:
        """Should return None when sd <= 0 (no exception)."""
        result = _cover_probability(-5.5, 6.0, 0.0, True)
        assert result is None

    def test_returns_none_for_negative_sd(self) -> None:
        """Should return None when sd < 0 (no exception)."""
        result = _cover_probability(-5.5, 6.0, -1.0, True)
        assert result is None

    def test_returns_none_for_none_line(self) -> None:
        """Should return None when line is None."""
        result = _cover_probability(None, 6.0, 12.0, True)
        assert result is None

    def test_returns_none_for_none_margin_mean(self) -> None:
        """Should return None when margin_mean is None."""
        result = _cover_probability(-5.5, None, 12.0, True)
        assert result is None


class TestOverProbabilityEquivalence:
    """Verify _over_probability matches the old manual formula."""

    @staticmethod
    def _old_over_probability(line: float, total_mean: float, total_sd: float) -> float:
        """Original implementation for comparison."""
        return 1.0 - _normal_cdf(line, mean=total_mean, sd=total_sd)

    def test_over_probability_standard_case(self) -> None:
        """Standard over probability calculation."""
        line = 220.0
        total_mean = 225.0
        total_sd = 12.0

        old_result = self._old_over_probability(line, total_mean, total_sd)
        new_result = _over_probability(line, total_mean, total_sd)

        assert new_result is not None
        assert new_result == pytest.approx(old_result, rel=1e-9)
        # Mean is 5 points above line → over should be > 50%
        assert new_result > 0.6

    def test_under_probability(self) -> None:
        """Under probability is complement of over."""
        line = 230.0
        total_mean = 225.0
        total_sd = 12.0

        over_p = _over_probability(line, total_mean, total_sd)
        assert over_p is not None

        under_p = 1.0 - over_p
        # Mean is 5 points below line → under should be > 50%
        assert under_p > 0.6

    def test_returns_none_for_none_sd(self) -> None:
        """Should return None when sd is None (no exception)."""
        result = _over_probability(220.0, 225.0, None)
        assert result is None

    def test_returns_none_for_zero_sd(self) -> None:
        """Should return None when sd <= 0 (no exception)."""
        result = _over_probability(220.0, 225.0, 0.0)
        assert result is None

    def test_returns_none_for_negative_sd(self) -> None:
        """Should return None when sd < 0 (no exception)."""
        result = _over_probability(220.0, 225.0, -5.0)
        assert result is None

    def test_returns_none_for_none_line(self) -> None:
        """Should return None when line is None."""
        result = _over_probability(None, 225.0, 12.0)
        assert result is None

    def test_returns_none_for_none_total_mean(self) -> None:
        """Should return None when total_mean is None."""
        result = _over_probability(220.0, None, 12.0)
        assert result is None


class TestProjectionsCanonicalFunctions:
    """Verify the canonical projections.py functions directly."""

    def test_cover_prob_away_minus_home_convention(self) -> None:
        """cover_prob with away_minus_home computes P(margin > -line)."""
        line = -5.5  # Home favored by 5.5
        margin_mean = 6.0
        margin_sd = 12.0

        result = cover_prob(line, margin_mean, margin_sd, sign_convention="away_minus_home")

        # threshold = -(-5.5) = 5.5
        # P(margin > 5.5) with mean=6, sd=12
        expected = 1.0 - _normal_cdf(5.5, mean=6.0, sd=12.0)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_over_prob_canonical(self) -> None:
        """over_prob computes P(total > line)."""
        line = 220.0
        total_mean = 225.0
        total_sd = 12.0

        result = over_prob(line, total_mean, total_sd)

        expected = 1.0 - _normal_cdf(220.0, mean=225.0, sd=12.0)
        assert result == pytest.approx(expected, rel=1e-9)
