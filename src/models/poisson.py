"""Poisson goals model with attack/defense components."""

from __future__ import annotations

from dataclasses import dataclass
from math import isnan, log
from typing import Any, Iterable, Mapping

import numpy as np

from models.base import BaseModel, GamePrediction, ModelMetadata, require_columns


@dataclass
class _PoissonState:
    teams: list[str]
    team_index: dict[str, int]
    attack: np.ndarray
    defense: np.ndarray
    mu: float
    home_advantage: float


class PoissonPowerRating:
    """Estimate attack/defense ratings with a log-link Poisson model."""

    def __init__(
        self,
        *,
        max_iter: int = 1500,
        learning_rate: float = 0.05,
        tol: float = 1e-6,
        reg_strength: float = 0.1,
        n_simulations: int = 20000,
        random_seed: int | None = None,
    ) -> None:
        self.model_id = "poisson"
        self.model_version = "1.0"
        self.params = {
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "tol": tol,
            "reg_strength": reg_strength,
            "n_simulations": n_simulations,
            "random_seed": random_seed,
        }
        self.max_iter = int(max_iter)
        self.learning_rate = float(learning_rate)
        self.tol = float(tol)
        self.reg_strength = float(reg_strength)
        self.n_simulations = int(n_simulations)
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)
        self._state: _PoissonState | None = None

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=self.model_id,
            model_version=self.model_version,
            params=self.params,
            supports_margin=True,
            supports_total=True,
            supports_win_prob=True,
        )

    def fit(self, games: Iterable[Mapping[str, Any]]) -> None:
        games_list: list[tuple[str, str, float, float, bool]] = []
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
                if isinstance(neutral_raw, float) and isnan(neutral_raw)
                else bool(neutral_raw)
            )
            teams.update([home, away])
            games_list.append((home, away, home_score, away_score, neutral))

        if not teams:
            self._state = None
            return

        teams_sorted = sorted(teams)
        team_index = {team: idx for idx, team in enumerate(teams_sorted)}
        home_idx = np.array([team_index[h] for h, _, _, _, _ in games_list], dtype=int)
        away_idx = np.array([team_index[a] for _, a, _, _, _ in games_list], dtype=int)
        home_goals = np.array([hg for _, _, hg, _, _ in games_list], dtype=float)
        away_goals = np.array([ag for _, _, _, ag, _ in games_list], dtype=float)
        neutral_flags = np.array([neutral for _, _, _, _, neutral in games_list], dtype=bool)
        home_flag = np.where(neutral_flags, 0.0, 1.0)

        n_teams = len(teams_sorted)
        attack = np.zeros(n_teams, dtype=float)
        defense = np.zeros(n_teams, dtype=float)

        avg_goals = (home_goals.sum() + away_goals.sum()) / max(1.0, 2.0 * len(games_list))
        mu = log(avg_goals) if avg_goals > 0 else 0.0

        home_avg = float(np.mean(home_goals)) if home_goals.size else 0.0
        away_avg = float(np.mean(away_goals)) if away_goals.size else 0.0
        home_advantage = log(home_avg / away_avg) if home_avg > 0 and away_avg > 0 else 0.0

        # Scale the learning step by scoring levels so basketball-style totals
        # don't blow up gradient updates.
        training_scale = max(1.0, avg_goals)
        step = self.learning_rate / max(
            1.0, float(len(games_list)) * training_scale
        )
        for _ in range(self.max_iter):
            log_lambda_home = np.clip(
                mu + home_advantage * home_flag + attack[home_idx] - defense[away_idx],
                -30.0,
                30.0,
            )
            log_lambda_away = np.clip(
                mu + attack[away_idx] - defense[home_idx],
                -30.0,
                30.0,
            )
            lambda_home = np.exp(log_lambda_home)
            lambda_away = np.exp(log_lambda_away)

            diff_home = home_goals - lambda_home
            diff_away = away_goals - lambda_away

            grad_attack = np.zeros(n_teams, dtype=float)
            np.add.at(grad_attack, home_idx, diff_home)
            np.add.at(grad_attack, away_idx, diff_away)

            grad_defense = np.zeros(n_teams, dtype=float)
            np.add.at(grad_defense, away_idx, -diff_home)
            np.add.at(grad_defense, home_idx, -diff_away)

            grad_mu = float(np.sum(diff_home + diff_away))
            grad_home_adv = float(np.sum(diff_home * home_flag))

            grad_attack -= self.reg_strength * attack
            grad_defense -= self.reg_strength * defense
            grad_home_adv -= self.reg_strength * home_advantage

            attack_update = step * grad_attack
            defense_update = step * grad_defense
            mu_update = step * grad_mu
            home_update = step * grad_home_adv

            attack += attack_update
            defense += defense_update
            mu += mu_update
            home_advantage += home_update

            mean_attack = float(np.mean(attack))
            mean_defense = float(np.mean(defense))
            attack -= mean_attack
            defense -= mean_defense
            mu += mean_attack - mean_defense

            max_update = max(
                float(np.max(np.abs(attack_update))),
                float(np.max(np.abs(defense_update))),
                abs(mu_update),
                abs(home_update),
            )
            if not (
                np.isfinite(mu)
                and np.isfinite(home_advantage)
                and np.isfinite(attack).all()
                and np.isfinite(defense).all()
            ):
                # Bail out if the optimizer diverges.
                self._state = None
                return
            if max_update < self.tol:
                break

        self._state = _PoissonState(
            teams=teams_sorted,
            team_index=team_index,
            attack=attack,
            defense=defense,
            mu=mu,
            home_advantage=home_advantage,
        )

    def expected_goals(
        self, home_team: str, away_team: str, *, neutral: bool = False
    ) -> tuple[float, float] | None:
        if self._state is None:
            return None
        home_idx = self._state.team_index.get(home_team)
        away_idx = self._state.team_index.get(away_team)
        home_attack = (
            float(self._state.attack[home_idx]) if home_idx is not None else 0.0
        )
        home_defense = (
            float(self._state.defense[home_idx]) if home_idx is not None else 0.0
        )
        away_attack = (
            float(self._state.attack[away_idx]) if away_idx is not None else 0.0
        )
        away_defense = (
            float(self._state.defense[away_idx]) if away_idx is not None else 0.0
        )
        home_flag = 0.0 if neutral else 1.0
        log_lambda_home = (
            self._state.mu
            + self._state.home_advantage * home_flag
            + home_attack
            - away_defense
        )
        log_lambda_away = (
            self._state.mu
            + away_attack
            - home_defense
        )
        return float(np.exp(log_lambda_home)), float(np.exp(log_lambda_away))

    def simulate_matchup(
        self,
        home_team: str,
        away_team: str,
        *,
        neutral: bool = False,
        n_simulations: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        expected = self.expected_goals(home_team, away_team, neutral=neutral)
        if expected is None:
            return None
        lambda_home, lambda_away = expected
        simulations = n_simulations or self.n_simulations
        if simulations <= 0:
            raise ValueError("n_simulations must be positive.")
        home_samples = self._rng.poisson(lambda_home, size=simulations)
        away_samples = self._rng.poisson(lambda_away, size=simulations)
        return home_samples, away_samples

    def rankings(self) -> list[tuple[str, float]]:
        """Return ratings ordered strongest to weakest."""
        if self._state is None:
            return []
        ratings = {
            team: float(self._state.attack[idx] - self._state.defense[idx])
            for team, idx in self._state.team_index.items()
        }
        return sorted(ratings.items(), key=lambda item: item[1], reverse=True)


class PoissonModel(BaseModel):
    """Backtest-ready Poisson model with Monte Carlo projections."""

    def __init__(
        self,
        *,
        max_iter: int = 1500,
        learning_rate: float = 0.05,
        tol: float = 1e-6,
        reg_strength: float = 0.1,
        n_simulations: int = 20000,
        random_seed: int | None = None,
    ) -> None:
        self._rating_model = PoissonPowerRating(
            max_iter=max_iter,
            learning_rate=learning_rate,
            tol=tol,
            reg_strength=reg_strength,
            n_simulations=n_simulations,
            random_seed=random_seed,
        )
        self._n_simulations = int(n_simulations)
        self._random_seed = random_seed

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id="poisson",
            model_version="1.0",
            params=self._rating_model.params,
            supports_margin=True,
            supports_total=True,
            supports_win_prob=True,
        )

    def fit(self, games_df: Any) -> None:
        require_columns(
            games_df, ["home_team", "away_team", "home_score", "away_score"]
        )
        self._rating_model.fit(games_df.to_dict(orient="records"))

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
                if isinstance(neutral_raw, float) and isnan(neutral_raw)
                else bool(neutral_raw)
            )
            samples = self._rating_model.simulate_matchup(
                home, away, neutral=neutral, n_simulations=self._n_simulations
            )
            if samples is None:
                continue
            home_samples, away_samples = samples
            total_samples = home_samples + away_samples
            margin_samples = home_samples - away_samples

            projected_home_score = float(np.mean(home_samples))
            projected_away_score = float(np.mean(away_samples))
            total_mean = float(np.mean(total_samples))
            margin_mean = float(np.mean(margin_samples))
            total_sd = float(np.std(total_samples))
            margin_sd = float(np.std(margin_samples))

            # Approximate moneyline probability: split ties 50/50 for OT/SO outcomes.
            p_home_win = float(
                np.mean(margin_samples > 0)
                + 0.5 * np.mean(margin_samples == 0)
            )

            game_id = row.get("game_id") or f"{row['date']}_{home}_{away}"
            predictions.append(
                GamePrediction(
                    game_id=str(game_id),
                    date=str(row["date"]),
                    home_team=home,
                    away_team=away,
                    p_home_win=p_home_win,
                    pred_margin=margin_mean,
                    pred_total=total_mean,
                    margin_sd=margin_sd,
                    total_sd=total_sd,
                    metadata=dict(model_identity),
                    extra={
                        "projected_home_score": projected_home_score,
                        "projected_away_score": projected_away_score,
                        "tie_rule": "split",
                    },
                )
            )

        return predictions

    def rankings(self) -> list[tuple[str, float]]:
        return self._rating_model.rankings()
