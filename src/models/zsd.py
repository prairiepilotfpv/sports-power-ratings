"""ZSD (Z-Score Deviation) rating and backtest model."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import logging
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from config import (
    DEFAULT_MARGIN_SD_FALLBACK,
    DEFAULT_TOTAL_MEAN_FALLBACK,
    DEFAULT_TOTAL_SD_FALLBACK,
    LEAGUE_MARGIN_SD_DEFAULT,
    MARGIN_SD_GUARDRAIL_MAX,
    MARGIN_SD_GUARDRAIL_MIN,
)
from eval.validation import get_validation_config
from models.base import BaseModel, GamePrediction, ModelMetadata, _home_win_prob_from_margin, require_columns
from models.calibration import (
    guardrail_margin_sd,
    recency_weight,
    resolve_fit_end_date,
    resolve_fit_end_date_from_games,
    weighted_rmse,
)

_LOG = logging.getLogger(__name__)

_PROB_EPS = 1e-6


@dataclass
class _ZSDState:
    teams: list[str]
    team_index: dict[str, int]
    offense: np.ndarray
    defense: np.ndarray
    home_factor: float
    away_factor: float
    mean_home_score: float
    std_home_score: float
    mean_away_score: float
    std_away_score: float
    margin_sd: float
    total_mean: float
    total_sd: float
    optimization_success: bool
    optimization_message: str | None
    residual_min: float | None
    residual_max: float | None


def _logistic(x: np.ndarray) -> np.ndarray:
    x_clipped = np.clip(x, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def _transform_to_score(param: np.ndarray, mean: float, std: float) -> np.ndarray:
    prob = _logistic(param)
    prob = np.clip(prob, _PROB_EPS, 1.0 - _PROB_EPS)
    z = norm.ppf(prob)
    return mean + std * z


def _predict_scores_from_params(
    *,
    offense: np.ndarray,
    defense: np.ndarray,
    home_factor: float,
    away_factor: float,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_flag: np.ndarray,
    mean_home: float,
    std_home: float,
    mean_away: float,
    std_away: float,
) -> tuple[np.ndarray, np.ndarray]:
    home_param = home_factor * home_flag + offense[home_idx] - defense[away_idx]
    away_param = away_factor * home_flag + offense[away_idx] - defense[home_idx]
    home_scores = _transform_to_score(home_param, mean_home, std_home)
    away_scores = _transform_to_score(away_param, mean_away, std_away)
    return home_scores, away_scores


class ZSDPowerRating:
    """Z-score deviation rating model with offense/defense components."""

    def __init__(
        self,
        *,
        max_iter: int = 100000,
        tol: float = 1e-8,
        optimizer: str = "slsqp",
        recency_lambda: float | None = None,
        random_seed: int | None = None,
        score_sd_floor: float = 1.0,
    ) -> None:
        self.model_id = "zsd"
        self.model_version = "1.0"
        self.params = {
            "max_iter": max_iter,
            "tol": tol,
            "optimizer": optimizer,
            "recency_lambda": recency_lambda,
            "random_seed": random_seed,
            "score_sd_floor": score_sd_floor,
        }
        self._max_iter = int(max_iter)
        self._tol = float(tol)
        self._optimizer = str(optimizer).strip().lower()
        self._recency_lambda = recency_lambda
        self._random_seed = random_seed
        self._score_sd_floor = float(score_sd_floor)
        self._state: _ZSDState | None = None

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

    def fit(
        self,
        games: Iterable[Mapping[str, Any]],
        *,
        fit_end_date: pd.Timestamp | None = None,
        recency_lambda: float | None = None,
    ) -> None:
        samples: list[tuple[str, str, float, float, bool, object]] = []
        teams: set[str] = set()
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
            neutral_raw = game.get("neutral", False)
            neutral = (
                False
                if isinstance(neutral_raw, float) and np.isnan(neutral_raw)
                else bool(neutral_raw)
            )
            teams.update([home, away])
            samples.append((home, away, home_score, away_score, neutral, game.get("date")))

        if not teams or not samples:
            self._state = None
            return

        teams_sorted = sorted(teams)
        team_index = {team: idx for idx, team in enumerate(teams_sorted)}
        n_teams = len(teams_sorted)

        home_idx = np.array([team_index[h] for h, _, _, _, _, _ in samples], dtype=int)
        away_idx = np.array([team_index[a] for _, a, _, _, _, _ in samples], dtype=int)
        home_scores = np.array([hs for _, _, hs, _, _, _ in samples], dtype=float)
        away_scores = np.array([ascore for _, _, _, ascore, _, _ in samples], dtype=float)
        neutral_flags = np.array([neutral for _, _, _, _, neutral, _ in samples], dtype=bool)
        home_flag = np.where(neutral_flags, 0.0, 1.0)

        fit_end_date = (
            fit_end_date
            if fit_end_date is not None
            else resolve_fit_end_date_from_games(
                [
                    {"date": date}
                    for *_, date in samples
                ]
            )
        )
        recency = recency_lambda if recency_lambda is not None else self._recency_lambda
        weights = (
            np.array(
                [
                    recency_weight(date, fit_end_date, recency)
                    for *_, date in samples
                ],
                dtype=float,
            )
            if recency is not None
            else None
        )

        mean_home = float(np.mean(home_scores)) if home_scores.size else 0.0
        mean_away = float(np.mean(away_scores)) if away_scores.size else 0.0
        std_home = float(np.std(home_scores, ddof=0)) if home_scores.size else 0.0
        std_away = float(np.std(away_scores, ddof=0)) if away_scores.size else 0.0
        if std_home <= 0:
            std_home = max(self._score_sd_floor, 1.0)
        if std_away <= 0:
            std_away = max(self._score_sd_floor, 1.0)

        rng = np.random.default_rng(self._random_seed)
        initial = rng.normal(0.0, 0.1, size=2 * n_teams + 2)

        bounds = [(-50.0, 50.0)] * (2 * n_teams) + [(None, None), (None, None)]
        
        # Create constraint functions that avoid capturing large arrays
        def _offense_constraint(p: np.ndarray) -> float:
            return float(np.mean(p[:n_teams]))
        
        def _defense_constraint(p: np.ndarray) -> float:
            return float(np.mean(p[n_teams : 2 * n_teams]))
        
        constraints = [
            {"type": "eq", "fun": _offense_constraint},
            {"type": "eq", "fun": _defense_constraint},
        ]

        def _objective(params: np.ndarray) -> float:
            offense = params[:n_teams]
            defense = params[n_teams : 2 * n_teams]
            home_factor = float(params[-2])
            away_factor = float(params[-1])
            pred_home, pred_away = _predict_scores_from_params(
                offense=offense,
                defense=defense,
                home_factor=home_factor,
                away_factor=away_factor,
                home_idx=home_idx,
                away_idx=away_idx,
                home_flag=home_flag,
                mean_home=mean_home,
                std_home=std_home,
                mean_away=mean_away,
                std_away=std_away,
            )
            errors = (home_scores - pred_home) ** 2 + (away_scores - pred_away) ** 2
            if weights is not None:
                errors = errors * weights
            return float(np.sum(errors))

        methods = []
        if self._optimizer in {"auto", "slsqp"}:
            methods.append("SLSQP")
        if self._optimizer in {"auto", "trust-constr"}:
            methods.append("trust-constr")
        if not methods:
            methods = ["SLSQP"]

        result = None
        for method in methods:
            try:
                result = minimize(
                    _objective,
                    initial,
                    method=method,
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": self._max_iter, "ftol": self._tol},
                )
                if result.success:
                    break
            except Exception as exc:
                _LOG.warning("ZSD optimization failed with %s: %s", method, exc)
                result = None
            finally:
                # Explicitly clean up after each optimization attempt to free memory
                gc.collect()

        if result is None or not result.success:
            optimization_success = False
            optimization_message = (
                result.message if result is not None else "optimization_failed"
            )
            params = initial.copy()
            _LOG.warning("ZSD optimization did not converge; using initial params.")
        else:
            optimization_success = True
            optimization_message = None
            params = result.x.copy()

        offense = params[:n_teams]
        defense = params[n_teams : 2 * n_teams]
        home_factor = float(params[-2])
        away_factor = float(params[-1])

        pred_home, pred_away = _predict_scores_from_params(
            offense=offense,
            defense=defense,
            home_factor=home_factor,
            away_factor=away_factor,
            home_idx=home_idx,
            away_idx=away_idx,
            home_flag=home_flag,
            mean_home=mean_home,
            std_home=std_home,
            mean_away=mean_away,
            std_away=std_away,
        )
        margin_residuals = (home_scores - away_scores) - (pred_home - pred_away)
        raw_margin_sd = weighted_rmse(margin_residuals, weights)
        margin_sd, _ = guardrail_margin_sd(
            raw_margin_sd,
            fallback_sd=LEAGUE_MARGIN_SD_DEFAULT,
            guardrail_min=MARGIN_SD_GUARDRAIL_MIN,
            guardrail_max=MARGIN_SD_GUARDRAIL_MAX,
        )

        totals = home_scores + away_scores
        total_mean = float(np.mean(totals)) if totals.size else DEFAULT_TOTAL_MEAN_FALLBACK
        total_sd = float(np.std(totals, ddof=0)) if totals.size else DEFAULT_TOTAL_SD_FALLBACK
        if total_sd <= 0 or not np.isfinite(total_sd):
            total_sd = DEFAULT_TOTAL_SD_FALLBACK

        residual_min = float(np.min(margin_residuals)) if margin_residuals.size else None
        residual_max = float(np.max(margin_residuals)) if margin_residuals.size else None

        self._state = _ZSDState(
            teams=teams_sorted,
            team_index=team_index,
            offense=offense,
            defense=defense,
            home_factor=home_factor,
            away_factor=away_factor,
            mean_home_score=mean_home,
            std_home_score=std_home,
            mean_away_score=mean_away,
            std_away_score=std_away,
            margin_sd=float(margin_sd),
            total_mean=float(total_mean),
            total_sd=float(total_sd),
            optimization_success=optimization_success,
            optimization_message=optimization_message,
            residual_min=residual_min,
            residual_max=residual_max,
        )
        
        # Explicitly release temporary arrays to reduce memory pressure
        del (
            home_idx,
            away_idx,
            home_scores,
            away_scores,
            neutral_flags,
            home_flag,
            weights,
            initial,
            margin_residuals,
            totals,
            pred_home,
            pred_away,
        )
        gc.collect()

    def rankings(self) -> list[tuple[str, float]]:
        if self._state is None:
            return []
        ratings = {
            team: float(self._state.offense[idx] - self._state.defense[idx])
            for team, idx in self._state.team_index.items()
        }
        return sorted(ratings.items(), key=lambda item: item[1], reverse=True)

    def _lookup_rating(self, team: str) -> tuple[float, float]:
        if self._state is None:
            return 0.0, 0.0
        idx = self._state.team_index.get(team)
        if idx is None:
            return 0.0, 0.0
        return float(self._state.offense[idx]), float(self._state.defense[idx])

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
        if self._state is None:
            return {
                "game_id": game_id or f"{home}|{away}",
                "date": str(date) if date is not None else "",
                "home_team": home,
                "away_team": away,
                "p_home_win": None,
                "pred_margin": None,
                "pred_total": None,
                "margin_mean": None,
                "total_mean": None,
                "margin_sd": None,
                "total_sd": None,
                "projected_home_score": None,
                "projected_away_score": None,
                "projected_total": None,
                "projected_win_prob": None,
                "model_p_home_win": None,
                "normal_p_home_win": None,
                "win_prob_source": None,
                "margin_dist_assumption": None,
            }

        cfg = get_validation_config(sport)
        home_flag = 0.0 if neutral else 1.0
        home_off, home_def = self._lookup_rating(home)
        away_off, away_def = self._lookup_rating(away)

        home_param = self._state.home_factor * home_flag + home_off - away_def
        away_param = self._state.away_factor * home_flag + away_off - home_def
        home_score = float(
            _transform_to_score(
                np.array([home_param]),
                self._state.mean_home_score,
                self._state.std_home_score,
            )[0]
        )
        away_score = float(
            _transform_to_score(
                np.array([away_param]),
                self._state.mean_away_score,
                self._state.std_away_score,
            )[0]
        )
        pred_margin = home_score - away_score
        pred_total = home_score + away_score

        margin_sd = self._state.margin_sd
        margin_sd, _ = guardrail_margin_sd(
            margin_sd,
            fallback_sd=LEAGUE_MARGIN_SD_DEFAULT,
            guardrail_min=cfg.margin_sd_min,
            guardrail_max=cfg.margin_sd_max,
        )

        total_sd = self._state.total_sd
        if total_sd is None or not (cfg.total_sd_min <= total_sd <= cfg.total_sd_max):
            total_sd = DEFAULT_TOTAL_SD_FALLBACK

        p_home_win = _home_win_prob_from_margin(pred_margin, margin_sd)
        if p_home_win is None:
            p_home_win = 0.5

        safe_game_id = game_id
        if safe_game_id is None:
            from src.utils.game_id import make_game_id

            season = getattr(self, "_season", None)
            if sport and season and date is not None:
                try:
                    safe_game_id = make_game_id(sport, season, date, away, home)
                except Exception:
                    safe_game_id = f"{date}|{away}|{home}"
            elif date is not None:
                safe_game_id = f"{date}|{away}|{home}"
            else:
                safe_game_id = f"{home}|{away}"

        return {
            "game_id": safe_game_id,
            "date": str(date) if date is not None else "",
            "home_team": home,
            "away_team": away,
            "p_home_win": float(p_home_win),
            "pred_margin": pred_margin,
            "pred_total": pred_total,
            "margin_mean": pred_margin,
            "total_mean": pred_total,
            "margin_sd": margin_sd,
            "total_sd": total_sd,
            "projected_home_score": home_score,
            "projected_away_score": away_score,
            "projected_total": pred_total,
            "projected_win_prob": float(p_home_win),
            "model_p_home_win": float(p_home_win),
            "normal_p_home_win": float(p_home_win),
            "win_prob_source": "margin_normal",
            "margin_dist_assumption": "normal_approx",
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


class ZSDModel(BaseModel):
    """Backtest-ready ZSD model."""

    def __init__(
        self,
        *,
        max_iter: int = 100000,
        tol: float = 1e-8,
        optimizer: str = "slsqp",
        recency_lambda: float | None = None,
        random_seed: int | None = None,
        score_sd_floor: float = 1.0,
    ) -> None:
        self._rating_model = ZSDPowerRating(
            max_iter=max_iter,
            tol=tol,
            optimizer=optimizer,
            recency_lambda=recency_lambda,
            random_seed=random_seed,
            score_sd_floor=score_sd_floor,
        )

    def metadata(self) -> ModelMetadata:
        meta = self._rating_model.metadata()
        return ModelMetadata(
            model_id=meta.model_id,
            model_version=meta.model_version,
            params=meta.params,
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
        if fit_end_date is None:
            fit_end_date = resolve_fit_end_date(games_df)
        self._rating_model.fit(
            games_df.to_dict(orient="records"),
            fit_end_date=fit_end_date,
        )

    def predict(self, upcoming_games_df: Any) -> list[GamePrediction]:
        require_columns(upcoming_games_df, ["date", "home_team", "away_team"])
        predictions: list[GamePrediction] = []
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
            canonical = self._rating_model.project_matchup(
                home,
                away,
                neutral=neutral,
                sport=row.get("sport"),
                date=row.get("date"),
                game_id=row.get("game_id"),
            )
            if canonical.get("p_home_win") is None:
                continue

            predictions.append(
                GamePrediction(
                    game_id=str(canonical["game_id"]),
                    date=str(canonical["date"]),
                    home_team=home,
                    away_team=away,
                    p_home_win=float(canonical["p_home_win"]),
                    pred_margin=float(canonical["pred_margin"]),
                    pred_total=float(canonical["pred_total"]),
                    margin_mean=float(canonical["margin_mean"]),
                    total_mean=float(canonical["total_mean"]),
                    margin_sd=float(canonical["margin_sd"]),
                    total_sd=float(canonical["total_sd"]),
                    win_prob_source=canonical["win_prob_source"],
                    margin_dist_assumption=canonical["margin_dist_assumption"],
                    metadata=dict(model_identity),
                    extra={
                        "projected_home_score": canonical["projected_home_score"],
                        "projected_away_score": canonical["projected_away_score"],
                        "projected_total": canonical["projected_total"],
                        "projected_spread": -float(canonical["pred_margin"]),
                        "projected_win_prob": canonical["projected_win_prob"],
                        "model_p_home_win": canonical["model_p_home_win"],
                        "normal_p_home_win": canonical["normal_p_home_win"],
                        "home_factor": getattr(self._rating_model._state, "home_factor", None)
                        if getattr(self._rating_model, "_state", None) is not None
                        else None,
                        "away_factor": getattr(self._rating_model._state, "away_factor", None)
                        if getattr(self._rating_model, "_state", None) is not None
                        else None,
                        "optimization_success": getattr(
                            self._rating_model._state, "optimization_success", None
                        )
                        if getattr(self._rating_model, "_state", None) is not None
                        else None,
                    },
                )
            )

        return predictions

    def rankings(self) -> list[tuple[str, float]]:
        return self._rating_model.rankings()

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
        return self._rating_model.project_matchup(
            home_team,
            away_team,
            neutral=neutral,
            sport=sport,
            date=date,
            game_id=game_id,
        )
