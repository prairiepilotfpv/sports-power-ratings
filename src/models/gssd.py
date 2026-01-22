"""GSSD (Generalized Scores Standard Deviation) power rating model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from config import (
    DEFAULT_MARGIN_SD_FALLBACK,
    DEFAULT_TOTAL_SD_FALLBACK,
    DEFAULT_WIN_PROB_K,
    MARGIN_SD_GUARDRAIL_MAX,
    MARGIN_SD_GUARDRAIL_MIN,
)
from models.base import BaseModel, GamePrediction, ModelMetadata, require_columns
from models.calibration import (
    ConditionalSDModel,
    align_spread_with_margin,
    fit_conditional_sd,
    fit_win_prob_bias,
    guardrail_margin_sd,
    recency_weight,
    resolve_fit_end_date,
    resolve_fit_end_date_from_games,
    weighted_least_squares,
    weighted_rmse,
)
from pipelines.projections import (
    fit_win_prob_scale,
    logistic_win_prob,
    win_prob_distribution,
)


@dataclass
class _StatAccumulator:
    sum_weight: float = 0.0
    sum_value: float = 0.0

    def add(self, value: float, weight: float) -> None:
        self.sum_weight += weight
        self.sum_value += value * weight

    def mean(self) -> float | None:
        if self.sum_weight <= 0:
            return None
        return self.sum_value / self.sum_weight


@dataclass
class _VarianceAccumulator:
    sum_weight: float = 0.0
    sum_value: float = 0.0
    sum_value_sq: float = 0.0

    def add(self, value: float, weight: float) -> None:
        self.sum_weight += weight
        self.sum_value += value * weight
        self.sum_value_sq += (value * value) * weight

    def mean(self) -> float | None:
        if self.sum_weight <= 0:
            return None
        return self.sum_value / self.sum_weight

    def sd(self) -> float | None:
        if self.sum_weight <= 0:
            return None
        mean = self.mean()
        if mean is None:
            return None
        variance = (self.sum_value_sq / self.sum_weight) - (mean * mean)
        if variance <= 0 or not math.isfinite(variance):
            return None
        return math.sqrt(variance)


class GSSDPowerRating:
    """Compute per-team scoring stats for GSSD."""

    def __init__(self) -> None:
        self.model_id = "gssd"
        self.model_version = "1.0"
        self.params: dict[str, Any] = {}
        self._team_stats: dict[str, dict[str, float]] = {}
        self._league_stats: dict[str, float] = {}

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=self.model_id,
            model_version=self.model_version,
            params=self.params,
            supports_margin=True,
            supports_total=False,
            supports_win_prob=True,
            role="primary",
            ensemble_weight=1.0,
        )

    def fit(
        self,
        games: Iterable[Mapping[str, Any]],
        *,
        recency_lambda: float | None = None,
        fit_end_date: pd.Timestamp | None = None,
    ) -> None:
        team_accums: dict[str, dict[str, _StatAccumulator]] = {}
        league_accums: dict[str, _StatAccumulator] = {}
        fit_end_date = (
            fit_end_date
            if fit_end_date is not None
            else resolve_fit_end_date_from_games(games)
        )

        def add_stat(team: str, key: str, value: float, weight: float) -> None:
            team_stats = team_accums.setdefault(team, {})
            team_acc = team_stats.setdefault(key, _StatAccumulator())
            team_acc.add(value, weight)

            league_acc = league_accums.setdefault(key, _StatAccumulator())
            league_acc.add(value, weight)

        for game in games:
            home = str(game.get("home_team", "")).strip()
            away = str(game.get("away_team", "")).strip()
            if not home or not away:
                continue
            try:
                home_score = float(game.get("home_score"))
                away_score = float(game.get("away_score"))
            except Exception:
                continue

            weight = recency_weight(
                game.get("date"), fit_end_date, recency_lambda
            )
            add_stat(home, "pfh", home_score, weight)
            add_stat(home, "pah", away_score, weight)
            add_stat(away, "pfa", away_score, weight)
            add_stat(away, "paa", home_score, weight)

        self._league_stats = {
            key: (acc.mean() if acc.mean() is not None else 0.0)
            for key, acc in league_accums.items()
        }
        self._team_stats = {}
        for team, stats in team_accums.items():
            self._team_stats[team] = {
                key: (acc.mean() if acc.mean() is not None else 0.0)
                for key, acc in stats.items()
            }

    def rankings(self) -> list[tuple[str, float]]:
        """Return ratings ordered from strongest to weakest."""
        ratings: dict[str, float] = {}
        for team in self._team_stats:
            stats = self.team_stats(team)
            net_rating = 0.5 * (
                (stats["pfh"] - stats["pah"]) + (stats["pfa"] - stats["paa"])
            )
            ratings[team] = net_rating
        return sorted(ratings.items(), key=lambda item: item[1], reverse=True)

    def team_stats(self, team: str) -> dict[str, float]:
        defaults = {
            "pfh": self._league_stats.get("pfh", 0.0),
            "pah": self._league_stats.get("pah", 0.0),
            "pfa": self._league_stats.get("pfa", 0.0),
            "paa": self._league_stats.get("paa", 0.0),
        }
        stats = self._team_stats.get(team)
        if stats is None:
            return dict(defaults)
        return {
            "pfh": stats.get("pfh", defaults["pfh"]),
            "pah": stats.get("pah", defaults["pah"]),
            "pfa": stats.get("pfa", defaults["pfa"]),
            "paa": stats.get("paa", defaults["paa"]),
        }


@dataclass
class GSSDCalibration:
    intercept: float
    beta_pfh: float
    beta_pah: float
    beta_pfa: float
    beta_paa: float
    home_advantage_points: float
    error_term: float


DEFAULT_CALIBRATION = GSSDCalibration(
    intercept=0.0,
    beta_pfh=0.0,
    beta_pah=0.0,
    beta_pfa=0.0,
    beta_paa=0.0,
    home_advantage_points=0.0,
    error_term=0.0,
)


class GSSDModel(BaseModel):
    """Backtest wrapper that calibrates GSSD ratings to points and win probs."""

    def __init__(
        self,
        *,
        recency_lambda: float | None = None,
        learn_home_advantage: bool = False,
        conditional_sd: bool = False,
        winprob_bias: float = 0.0,
        learn_winprob_bias: bool = False,
    ) -> None:
        """Initialize the backtest model.

        recency_lambda: None disables recency weighting (default = current behavior).
        conditional_sd: when False, uses the constant error_term for margin SDs.
        learn_winprob_bias: when False, uses the provided winprob_bias as-is.
        """
        self._gssd = GSSDPowerRating()
        self._coefficients = DEFAULT_CALIBRATION
        self._win_prob_k = DEFAULT_WIN_PROB_K
        self._recency_lambda = recency_lambda
        self._learn_home_advantage = learn_home_advantage
        self._conditional_sd = conditional_sd
        self._conditional_sd_model: ConditionalSDModel | None = None
        self._win_prob_bias = float(winprob_bias)
        self._learn_winprob_bias = learn_winprob_bias
        self._total_mean: float | None = None
        self._total_sd: float | None = None

    def metadata(self) -> ModelMetadata:
        meta = self._gssd.metadata()
        return ModelMetadata(
            model_id=meta.model_id,
            model_version=meta.model_version,
            params={
                **meta.params,
                "recency_lambda": self._recency_lambda,
                "learn_home_advantage": self._learn_home_advantage,
                "conditional_sd": self._conditional_sd,
                "winprob_bias": self._win_prob_bias,
                "learn_winprob_bias": self._learn_winprob_bias,
            },
            supports_margin=True,
            supports_total=True,
            supports_win_prob=True,
            role="primary",
            ensemble_weight=1.0,
        )

    def fit(self, games_df: Any, *, fit_end_date: pd.Timestamp | None = None) -> None:
        require_columns(
            games_df, ["home_team", "away_team", "home_score", "away_score"]
        )
        games = games_df.to_dict(orient="records")
        # Use explicit fit_end_date when provided; otherwise fall back to resolving from the DataFrame
        fit_end_date = fit_end_date if fit_end_date is not None else resolve_fit_end_date(games_df)
        self._gssd.fit(
            games,
            recency_lambda=self._recency_lambda,
            fit_end_date=fit_end_date,
        )

        design_matrix: list[list[float]] = []
        margins: list[float] = []
        # Recency weighting is applied in the core power-rating fit (GSSDPowerRating.fit).
        # Do not apply additional recency weighting in calibration to keep a single application.
        win_prob_samples: list[tuple[float, int]] = []
        win_prob_spreads: list[float] = []
        win_prob_outcomes: list[int] = []
        total_accumulator = _VarianceAccumulator()

        for game in games:
            home = str(game.get("home_team", "")).strip()
            away = str(game.get("away_team", "")).strip()
            if not home or not away:
                continue
            try:
                margin = float(game.get("home_score")) - float(game.get("away_score"))
            except Exception:
                continue

            neutral_raw = game.get("neutral", False)
            neutral = (
                False
                if isinstance(neutral_raw, float) and np.isnan(neutral_raw)
                else bool(neutral_raw)
            )
            home_advantage_flag = 0.0 if neutral else 1.0

            home_stats = self._gssd.team_stats(home)
            away_stats = self._gssd.team_stats(away)
            row = [
                1.0,
                home_stats["pfh"],
                home_stats["pah"],
                away_stats["pfa"],
                away_stats["paa"],
            ]
            if self._learn_home_advantage:
                row.append(home_advantage_flag)
            design_matrix.append(row)
            margins.append(margin)
            try:
                total_points = float(game.get("home_score")) + float(
                    game.get("away_score")
                )
                # Unweighted totals accumulation; recency applied once in core fit.
                total_accumulator.add(total_points, 1.0)
            except Exception:
                pass

        if design_matrix:
            matrix = np.asarray(design_matrix, dtype=float)
            target = np.asarray(margins, dtype=float)
            # Unweighted calibration to standardize single recency application.
            coeffs = weighted_least_squares(matrix, target, weights=None)
            predictions = matrix @ coeffs
            residuals = predictions - target
            error_term = weighted_rmse(residuals, None)
            home_advantage_points = (
                float(coeffs[5]) if self._learn_home_advantage else 0.0
            )
            self._coefficients = GSSDCalibration(
                intercept=float(coeffs[0]),
                beta_pfh=float(coeffs[1]),
                beta_pah=float(coeffs[2]),
                beta_pfa=float(coeffs[3]),
                beta_paa=float(coeffs[4]),
                home_advantage_points=home_advantage_points,
                error_term=error_term,
            )

            if self._conditional_sd:
                self._conditional_sd_model = fit_conditional_sd(
                    predictions, residuals, weights=None
                )
            else:
                self._conditional_sd_model = None

            for predicted_margin, actual_margin in zip(
                predictions, target, strict=False
            ):
                if actual_margin == 0:
                    continue
                projected_spread = -float(predicted_margin)
                win_prob_samples.append(
                    (projected_spread, 1 if actual_margin > 0 else 0)
                )
                win_prob_spreads.append(projected_spread)
                win_prob_outcomes.append(1 if actual_margin > 0 else 0)
        else:
            self._coefficients = DEFAULT_CALIBRATION
            self._conditional_sd_model = None

        self._win_prob_k = fit_win_prob_scale(
            win_prob_samples, default_k=DEFAULT_WIN_PROB_K
        )
        if self._learn_winprob_bias and win_prob_spreads:
            self._win_prob_bias = fit_win_prob_bias(
                win_prob_spreads,
                win_prob_outcomes,
                win_prob_k=self._win_prob_k if self._win_prob_k > 0 else DEFAULT_WIN_PROB_K,
            )

        self._total_mean = total_accumulator.mean()
        self._total_sd = total_accumulator.sd()

    def _canonical_matchup_prediction(
        self,
        home: str,
        away: str,
        *,
        neutral: bool,
        sport: str | None,
        date: str | None,
        game_id: str | None,
    ) -> dict[str, Any]:
        coefficients = self._coefficients
        win_prob_k = self._win_prob_k if self._win_prob_k > 0 else DEFAULT_WIN_PROB_K
        safe_game_id = (
            str(game_id)
            if game_id is not None
            else f"{date}_{home}_{away}"
            if date is not None
            else f"{home}_{away}"
        )
        safe_date = str(date) if date is not None else ""
        home_stats = self._gssd.team_stats(home)
        away_stats = self._gssd.team_stats(away)
        pred_margin = (
            coefficients.intercept
            + coefficients.beta_pfh * home_stats["pfh"]
            + coefficients.beta_pah * home_stats["pah"]
            + coefficients.beta_pfa * away_stats["pfa"]
            + coefficients.beta_paa * away_stats["paa"]
            + coefficients.home_advantage_points * (0.0 if neutral else 1.0)
        )
        total_mean = (home_stats["pfh"] + away_stats["paa"] + away_stats["pfa"] + home_stats["pah"]) / 2.0
        total_sd = (
            self._total_sd
            if self._total_sd is not None and self._total_sd > 0
            else DEFAULT_TOTAL_SD_FALLBACK
        )
        projected_home_score = 0.5 * (total_mean + pred_margin)
        projected_away_score = 0.5 * (total_mean - pred_margin)
        projected_total = total_mean
        projected_spread = -pred_margin
        adjusted_spread = align_spread_with_margin(
            pred_margin, projected_spread - self._win_prob_bias
        )
        p_home_win = logistic_win_prob(adjusted_spread, win_prob_k)
        margin_sd_raw = (
            self._conditional_sd_model.predict(pred_margin)
            if self._conditional_sd_model is not None
            else coefficients.error_term
        )
        margin_sd, _ = guardrail_margin_sd(
            margin_sd_raw,
            fallback_sd=DEFAULT_MARGIN_SD_FALLBACK,
            guardrail_min=MARGIN_SD_GUARDRAIL_MIN,
            guardrail_max=MARGIN_SD_GUARDRAIL_MAX,
        )
        win_prob_dist = win_prob_distribution(
            p_home_win,
            win_prob_k=win_prob_k,
            margin_std=margin_sd,
        )
        return {
            "game_id": safe_game_id,
            "date": safe_date,
            "home_team": home,
            "away_team": away,
            "p_home_win": p_home_win,
            "win_prob_dist": win_prob_dist,
            "pred_margin": pred_margin,
            "pred_total": total_mean,
            "margin_mean": pred_margin,
            "total_mean": total_mean,
            "margin_sd": margin_sd,
            "total_sd": total_sd,
            "projected_home_score": projected_home_score,
            "projected_away_score": projected_away_score,
            "projected_total": projected_total,
            "projected_win_prob": p_home_win,
            "model_p_home_win": p_home_win,
            "normal_p_home_win": None,
            "win_prob_source": "logistic",
            "margin_dist_assumption": "normal_approx",
            "logistic_home_win_prob": p_home_win,
            "win_prob_k": win_prob_k,
            "winprob_bias": self._win_prob_bias,
            "conditional_sd": self._conditional_sd,
            "conditional_sd_intercept": (
                self._conditional_sd_model.intercept
                if self._conditional_sd_model is not None
                else None
            ),
            "conditional_sd_slope": (
                self._conditional_sd_model.slope
                if self._conditional_sd_model is not None
                else None
            ),
        }

    def project_matchup(
        self,
        home_team: str,
        away_team: str,
        *,
        neutral: bool = False,
        sport: str | None = None,
        date: str | None = None,
        game_id: str | None = None,
    ) -> dict[str, Any]:
        return self._canonical_matchup_prediction(
            home_team,
            away_team,
            neutral=neutral,
            sport=sport,
            date=date,
            game_id=game_id,
        )

    def predict(self, upcoming_games_df: Any) -> list[GamePrediction]:
        require_columns(upcoming_games_df, ["date", "home_team", "away_team"])
        predictions: list[GamePrediction] = []
        coefficients = self._coefficients
        model_identity = self.metadata().identity_dict()

        for row in upcoming_games_df.to_dict(orient="records"):
            home = str(row.get("home_team", "")).strip()
            away = str(row.get("away_team", "")).strip()
            if not home or not away:
                continue

            neutral_raw = row.get("neutral", False)
            neutral = (
                False
                if isinstance(neutral_raw, float) and np.isnan(neutral_raw)
                else bool(neutral_raw)
            )
            sport = row.get("sport") if isinstance(row, dict) else None
            canonical = self._canonical_matchup_prediction(
                home,
                away,
                neutral=neutral,
                sport=sport,
                date=row.get("date"),
                game_id=row.get("game_id"),
            )
            predictions.append(
                GamePrediction(
                    game_id=str(canonical["game_id"]),
                    date=str(canonical["date"]),
                    home_team=home,
                    away_team=away,
                    p_home_win=canonical["p_home_win"],
                    win_prob_dist=canonical["win_prob_dist"],
                    pred_margin=canonical["pred_margin"],
                    pred_total=canonical["pred_total"],
                    margin_sd=canonical["margin_sd"],
                    total_sd=canonical["total_sd"],
                    total_mean=canonical["total_mean"],
                    win_prob_source=canonical["win_prob_source"],
                    margin_dist_assumption=canonical["margin_dist_assumption"],
                    metadata=dict(model_identity),
                    extra={
                        "intercept": coefficients.intercept,
                        "beta_pfh": coefficients.beta_pfh,
                        "beta_pah": coefficients.beta_pah,
                        "beta_pfa": coefficients.beta_pfa,
                        "beta_paa": coefficients.beta_paa,
                        "home_advantage_points": coefficients.home_advantage_points,
                        "error_term": coefficients.error_term,
                        "win_prob_k": canonical["win_prob_k"],
                        "winprob_bias": canonical["winprob_bias"],
                        "conditional_sd": canonical["conditional_sd"],
                        "total_mean": canonical["total_mean"],
                        "total_sd": canonical["total_sd"],
                        "projected_home_score": canonical["projected_home_score"],
                        "projected_away_score": canonical["projected_away_score"],
                        "projected_total": canonical["projected_total"],
                        "conditional_sd_intercept": canonical["conditional_sd_intercept"],
                        "conditional_sd_slope": canonical["conditional_sd_slope"],
                    },
                )
            )
        return predictions
