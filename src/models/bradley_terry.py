"""Bradley-Terry power rating model for win/loss outcomes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from math import isnan
from typing import Any, DefaultDict, Iterable, Mapping

import numpy as np

from models.base import BaseModel, GamePrediction, ModelMetadata, require_columns


@dataclass(frozen=True)
class BTCalibration:
    """Post-fit calibration used to translate rating differences into score/margin/total distributions.

    This is intentionally sport-agnostic: bounds and scales are learned from the fitted data
    instead of being hard-coded to NBA-like totals.
    """

    margin_a: float
    margin_b: float
    margin_sigma: float
    total_c: float
    total_u: float
    total_sigma: float
    # Learned bounds for plausible totals (used for safe clamping in projections).
    total_lower: float | None = None
    total_upper: float | None = None


class BradleyTerry:
    """Iterative Bradley-Terry solver with optional home advantage term."""

    def __init__(
        self,
        *,
        max_iter: int = 500,
        tol: float = 1e-8,
        temp: float = 3.0,
        l2_lambda: float = 1e-3,
        hfa_logit: float = 0.0,
        learn_hfa: bool = True,
    ) -> None:
        self.model_id = "bradley-terry"
        self.model_version = "1.0"
        self.params = {
            "max_iter": max_iter,
            "tol": tol,
            "temp": temp,
            "lambda": l2_lambda,
            "hfa_logit": hfa_logit,
            "learn_hfa": learn_hfa,
        }
        self.max_iter = max_iter
        self.tol = tol
        self.temp = temp
        self.l2_lambda = l2_lambda
        self.learn_hfa = learn_hfa
        self.ratings: DefaultDict[str, float] = defaultdict(float)
        self.games_played: DefaultDict[str, int] = defaultdict(int)
        self.hfa_logit = hfa_logit
        self.calibration = BTCalibration(
            margin_a=0.0,
            margin_b=0.0,
            margin_sigma=1.0,
            total_c=0.0,
            total_u=0.0,
            total_sigma=1.0,
        )

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=self.model_id,
            model_version=self.model_version,
            params=self.params,
            supports_margin=True,
            supports_total=True,
            supports_win_prob=True,
            role="primary",
            ensemble_weight=1.0,
        )

    @staticmethod
    def _sigmoid(score: float) -> float:
        """Numerically stable sigmoid for logistic probability."""
        if score >= 0:
            z = np.exp(-score)
            return float(1.0 / (1.0 + z))
        z = np.exp(score)
        return float(z / (1.0 + z))

    @staticmethod
    def _clip_prob(prob: float, *, eps: float = 1e-6) -> float:
        return float(min(max(prob, eps), 1.0 - eps))

    @staticmethod
    def _normal_cdf(x: float, *, mean: float, sd: float) -> float:
        if sd <= 0 or not np.isfinite(sd):
            raise ValueError("Standard deviation must be positive and finite.")
        z = (x - mean) / (sd * np.sqrt(2.0))
        return float(0.5 * (1.0 + math.erf(z)))

    def predict_probability(
        self, team_a: str, team_b: str, venue: str = "neutral"
    ) -> float:
        """Predict win probability for team_a vs team_b with a venue adjustment."""
        rating_a = self.ratings[team_a]
        rating_b = self.ratings[team_b]
        score = rating_a - rating_b
        if venue == "home":
            score += self.hfa_logit
        elif venue == "away":
            score -= self.hfa_logit
        score = score / self.temp if self.temp != 0 else score
        return self._clip_prob(self._sigmoid(score))

    def fit(self, games: Iterable[Mapping[str, Any]]) -> None:
        """Fit Bradley-Terry ratings from game results."""
        teams: set[str] = set()
        games_list: list[tuple[str, str, float, bool, float, float]] = []

        for game in games:
            # Drop rows without names or usable scores.
            home = str(game.get("home_team", "")).strip()
            away = str(game.get("away_team", "")).strip()
            if not home or not away:
                continue
            try:
                home_score = int(game.get("home_score"))
                away_score = int(game.get("away_score"))
            except Exception:
                continue

            teams.update([home, away])
            self.games_played[home] += 1
            self.games_played[away] += 1

            outcome = 0.5 if home_score == away_score else float(home_score > away_score)

            neutral_raw = game.get("neutral", False)
            if isinstance(neutral_raw, float) and isnan(neutral_raw):
                neutral = False
            else:
                neutral = False if neutral_raw is None else bool(neutral_raw)
            margin = float(home_score) - float(away_score)
            total = float(home_score) + float(away_score)
            games_list.append((home, away, outcome, neutral, margin, total))

        if not teams:
            return

        team_list = sorted(teams)
        team_index = {team: idx for idx, team in enumerate(team_list)}
        ratings = np.zeros(len(team_list), dtype=float)
        hfa_logit = float(self.hfa_logit)
        step = 0.1 / max(1.0, len(games_list) / 1000.0)
        temp = self.temp if self.temp != 0 else 1.0

        for _ in range(self.max_iter):
            grad = np.zeros_like(ratings)
            grad_hfa = 0.0
            for home, away, outcome, neutral, _, _ in games_list:
                home_idx = team_index[home]
                away_idx = team_index[away]
                score = ratings[home_idx] - ratings[away_idx]
                if not neutral:
                    score += hfa_logit
                score = score / temp
                win_prob = self._clip_prob(self._sigmoid(score))
                diff = win_prob - outcome
                grad[home_idx] += diff / temp
                grad[away_idx] -= diff / temp
                if not neutral and self.learn_hfa:
                    grad_hfa += diff / temp

            grad += 2.0 * self.l2_lambda * ratings
            updates = -step * grad
            ratings += updates
            max_delta = float(np.max(np.abs(updates))) if updates.size else 0.0
            if self.learn_hfa:
                hfa_update = -step * grad_hfa
                hfa_logit += hfa_update
                max_delta = max(max_delta, abs(hfa_update))
            if max_delta < self.tol:
                break

        for team in teams:
            self.ratings[team] = float(ratings[team_index[team]])
        self.hfa_logit = float(hfa_logit)
        self._fit_calibration(games_list)

    def _fit_calibration(
        self,
        games_list: list[tuple[str, str, float, bool, float, float]],
    ) -> None:
        if not games_list:
            return
        d_values = []
        margins = []
        totals = []
        for home, away, _, neutral, margin, total in games_list:
            d_value = self.ratings[home] - self.ratings[away]
            if not neutral:
                d_value += self.hfa_logit
            d_values.append(float(d_value))
            margins.append(float(margin))
            totals.append(float(total))

        if d_values:
            design = np.column_stack([np.ones(len(d_values)), np.asarray(d_values)])
            coeffs, *_ = np.linalg.lstsq(design, np.asarray(margins), rcond=None)
            margin_a = float(coeffs[0])
            margin_b = float(coeffs[1])
            margin_residuals = np.asarray(margins) - (design @ coeffs)
            margin_sigma = float(
                np.std(margin_residuals, ddof=1 if len(margin_residuals) > 1 else 0)
            )
        else:
            margin_a, margin_b, margin_sigma = 0.0, 0.0, 0.0

        total_values = np.asarray(totals)
        abs_d = np.abs(np.asarray(d_values))
        if len(total_values) > 0:
            total_design = np.column_stack([np.ones(len(abs_d)), abs_d])
            total_coeffs, *_ = np.linalg.lstsq(
                total_design, total_values, rcond=None
            )
            total_c = float(total_coeffs[0])
            total_u = float(total_coeffs[1])
            total_residuals = total_values - (total_design @ total_coeffs)
            total_sigma = float(
                np.std(total_residuals, ddof=1 if len(total_residuals) > 1 else 0)
            )
        else:
            total_c = float(np.mean(total_values)) if total_values.size else 0.0
            total_u = 0.0
            total_sigma = 0.0

        if not np.isfinite(margin_sigma) or margin_sigma <= 0:
            # Fallback to an observed margin spread if regression residuals are unusable.
            observed_margins = np.asarray(margins)
            margin_sigma = float(np.std(observed_margins, ddof=1 if len(observed_margins) > 1 else 0)) if observed_margins.size else 1.0
        if not np.isfinite(total_sigma) or total_sigma <= 0:
            observed_totals = np.asarray(totals)
            total_sigma = float(np.std(observed_totals, ddof=1 if len(observed_totals) > 1 else 0)) if observed_totals.size else 1.0

        # Learn reasonable total bounds from data (robust quantiles when possible).
        total_lower: float | None = None
        total_upper: float | None = None
        if total_values.size:
            if total_values.size >= 10:
                total_lower = float(np.quantile(total_values, 0.01))
                total_upper = float(np.quantile(total_values, 0.99))
            else:
                total_lower = float(np.min(total_values))
                total_upper = float(np.max(total_values))
            if not (np.isfinite(total_lower) and np.isfinite(total_upper) and total_lower < total_upper):
                total_lower = None
                total_upper = None


        self.calibration = BTCalibration(
            margin_a=margin_a,
            margin_b=margin_b,
            margin_sigma=margin_sigma,
            total_c=total_c,
            total_u=total_u,
            total_sigma=total_sigma,
            total_lower=total_lower,
            total_upper=total_upper,
        )

    def project_matchup(
        self, home_team: str, away_team: str, *, neutral: bool
    ) -> dict[str, float]:
        d_value = self.ratings[home_team] - self.ratings[away_team]
        if not neutral:
            d_value += self.hfa_logit

        margin_mean = self.calibration.margin_a + self.calibration.margin_b * d_value
        margin_sd = float(self.calibration.margin_sigma)
        if not np.isfinite(margin_sd) or margin_sd <= 0:
            margin_sd = 1.0

        total_mean = self.calibration.total_c + self.calibration.total_u * abs(d_value)
        # Clamp totals only if we learned bounds from data (keeps the model sport-agnostic).
        if self.calibration.total_lower is not None and self.calibration.total_upper is not None:
            total_mean = float(min(max(total_mean, self.calibration.total_lower), self.calibration.total_upper))

        total_sd = float(self.calibration.total_sigma)
        if not np.isfinite(total_sd) or total_sd <= 0:
            total_sd = 1.0

        projected_home_score = (total_mean + margin_mean) / 2.0
        projected_away_score = (total_mean - margin_mean) / 2.0

        p_home_win = 1.0 - self._normal_cdf(0.0, mean=margin_mean, sd=margin_sd)
        p_home_win = self._clip_prob(p_home_win)
        return {
            "margin_mean": float(margin_mean),
            "margin_sd": float(margin_sd),
            "total_mean": float(total_mean),
            "total_sd": float(total_sd),
            "p_home_win": float(p_home_win),
            "projected_home_score": float(projected_home_score),
            "projected_away_score": float(projected_away_score),
        }

    def rankings(self) -> list[tuple[str, float]]:
        """Return ratings ordered from strongest to weakest."""
        return sorted(self.ratings.items(), key=lambda item: item[1], reverse=True)


class BradleyTerryBacktest(BaseModel):
    """Backtest adapter that reuses the core BradleyTerry implementation."""

    def __init__(
        self,
        *,
        max_iter: int = 500,
        tol: float = 1e-8,
        temp: float = 3.0,
        l2_lambda: float = 1e-3,
        hfa_logit: float = 0.0,
        learn_hfa: bool = True,
        strict: bool = False,
    ) -> None:
        self._model = BradleyTerry(
            max_iter=max_iter,
            tol=tol,
            temp=temp,
            l2_lambda=l2_lambda,
            hfa_logit=hfa_logit,
            learn_hfa=learn_hfa,
        )
        self._strict = strict

    def metadata(self) -> ModelMetadata:
        return self._model.metadata()

    def fit(self, games_df: Any) -> None:
        require_columns(
            games_df, ["home_team", "away_team", "home_score", "away_score"]
        )
        self._model.fit(games_df.to_dict(orient="records"))

    def predict(self, upcoming_games_df: Any) -> list[GamePrediction]:
        require_columns(upcoming_games_df, ["date", "home_team", "away_team"])
        predictions: list[GamePrediction] = []
        metadata = self.metadata().identity_dict()

        for row in upcoming_games_df.to_dict(orient="records"):
            home = str(row.get("home_team", "")).strip()
            away = str(row.get("away_team", "")).strip()
            if not home or not away:
                continue

            neutral_raw = row.get("neutral", False)
            neutral = (
                False
                if isinstance(neutral_raw, float) and isnan(neutral_raw)
                else bool(neutral_raw)
            )
            projection = self._model.project_matchup(home, away, neutral=neutral)
            p_home_win = projection["p_home_win"]
            pred_margin = projection["margin_mean"]
            game_id = row.get("game_id") or f"{row['date']}_{home}_{away}"
            extra = {
                "projected_home_score": projection["projected_home_score"],
                "projected_away_score": projection["projected_away_score"],
                "projected_spread": -projection["margin_mean"],
                "model_p_home_win": p_home_win,
                "normal_p_home_win": p_home_win,
                "win_prob_source": "bt_margin_normal",
                "margin_dist_assumption": "normal_approx",
                "logistic_home_win_prob": None,
            }
            self._validate_prediction(
                p_home_win,
                projection["margin_sd"],
                projection["total_sd"],
                extra["win_prob_source"],
                game_id,
            )

            predictions.append(
                GamePrediction(
                    game_id=str(game_id),
                    date=str(row["date"]),
                    home_team=home,
                    away_team=away,
                    p_home_win=p_home_win,
                    win_prob_samples=None,
                    pred_margin=pred_margin,
                    pred_total=projection["total_mean"],
                    margin_sd=projection["margin_sd"],
                    total_sd=projection["total_sd"],
                    margin_mean=projection["margin_mean"],
                    total_mean=projection["total_mean"],
                    win_prob_source="direct",
                    margin_dist_assumption="none",
                    metadata=dict(metadata),
                    extra=extra,
                )
            )

        return predictions

    def _validate_prediction(
        self,
        p_home_win: float,
        margin_sd: float,
        total_sd: float,
        win_prob_source: str,
        game_id: str,
    ) -> None:
        errors = []
        if not (0.0 < p_home_win < 1.0):
            errors.append("p_home_win must be between 0 and 1.")
        if not (np.isfinite(margin_sd) and margin_sd > 0.0):
            errors.append("margin_sd must be positive and finite.")
        if not (np.isfinite(total_sd) and total_sd > 0.0):
            errors.append("total_sd must be positive and finite.")
        # win_prob_source may be 'direct' for models that return probabilities directly.
        if not errors:
            return
        message = f"Invalid BT prediction for {game_id}: " + "; ".join(errors)
        if self._strict:
            raise ValueError(message)
        import warnings

        warnings.warn(message, RuntimeWarning, stacklevel=2)
