from __future__ import annotations

from typing import Any

import numpy as np

from models.base import BaseModel, GamePrediction, ModelMetadata, require_columns
from models.bradley_terry import BradleyTerry


class BradleyTerryHFA(BaseModel):
    """Bradley-Terry with home-field advantage.

    Notes:
        pred_margin is reported as the calibrated margin mean derived from
        the BT rating differential and a normal margin model.
    """

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
        suppress_small_sd_warning: bool = False,
    ) -> None:
        self._max_iter = max_iter
        self._tol = tol
        self._model = BradleyTerry(
            max_iter=max_iter,
            tol=tol,
            temp=temp,
            l2_lambda=l2_lambda,
            hfa_logit=hfa_logit,
            learn_hfa=learn_hfa,
        )
        self._strict = strict
        # If True, do not emit warnings about small margin/total sd which are
        # expected for some low-scoring sports (e.g., NHL).
        self._suppress_small_sd_warning = bool(suppress_small_sd_warning)

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id="bradley_terry_hfa",
            model_version="1.0",
            params={
                "max_iter": self._max_iter,
                "tol": self._tol,
                "temp": self._model.temp,
                "lambda": self._model.l2_lambda,
                "hfa_logit": self._model.hfa_logit,
                "learn_hfa": self._model.learn_hfa,
                "strict": self._strict,
                "suppress_small_sd_warning": self._suppress_small_sd_warning,
            },
            supports_margin=True,
            supports_total=True,
            supports_win_prob=True,
            role="primary",
            ensemble_weight=1.0,
        )

    def fit(self, games_df: Any) -> None:
        require_columns(
            games_df, ["home_team", "away_team", "home_score", "away_score"]
        )
        games = games_df.to_dict(orient="records")
        self._model.fit(games)

    def predict(self, upcoming_games_df: Any) -> list[GamePrediction]:
        require_columns(upcoming_games_df, ["date", "home_team", "away_team"])
        predictions: list[GamePrediction] = []
        model_identity = self.metadata().identity_dict()
        for row in upcoming_games_df.to_dict(orient="records"):
            neutral = bool(row.get("neutral", False))
            home_team = str(row["home_team"])
            away_team = str(row["away_team"])
            projection = self._model.project_matchup(
                home_team,
                away_team,
                neutral=neutral,
            )
            p_home_win = projection["p_home_win"]
            pred_margin = projection["margin_mean"]
            game_id = (
                row.get("game_id")
                or f"{row['date']}_{home_team}_{away_team}"
            )
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
                    home_team=home_team,
                    away_team=away_team,
                    p_home_win=p_home_win,
                    win_prob_dist=None,
                    pred_margin=pred_margin,
                    pred_total=projection["total_mean"],
                    margin_sd=projection["margin_sd"],
                    total_sd=projection["total_sd"],
                    margin_mean=projection["margin_mean"],
                    total_mean=projection["total_mean"],
                    win_prob_source="direct",
                    margin_dist_assumption="none",
                    metadata=dict(model_identity),
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

        # Allow callers to suppress small-sd warnings (useful for low-scoring sports
        # where learned margin/total sigmas are naturally small).
        if not self._suppress_small_sd_warning:
            calib = getattr(self._model, "calibration", None)
            try:
                calib_margin = float(calib.margin_sigma)
            except Exception:
                calib_margin = 1.0
            try:
                calib_total = float(calib.total_sigma)
            except Exception:
                calib_total = 1.0

            # Derive adaptive floor thresholds from the learned calibration but
            # do not raise the bar above the historical constants used previously
            # (5.0 for margin, 8.0 for total). This keeps the check conservative
            # for high-scoring sports while avoiding spurious warnings for low
            # scoring sports like NHL.
            min_margin = min(5.0, max(1.0, calib_margin))
            min_total = min(8.0, max(1.0, calib_total))

            if margin_sd < min_margin:
                errors.append(f"margin_sd must be at least {min_margin:.2f}.")
            if total_sd < min_total:
                errors.append(f"total_sd must be at least {min_total:.2f}.")

        if win_prob_source == "direct":
            errors.append("win_prob_source cannot be 'direct'.")
        if not errors:
            return
        message = f"Invalid BT prediction for {game_id}: " + "; ".join(errors)
        if self._strict:
            raise ValueError(message)
        import warnings

        warnings.warn(message, RuntimeWarning, stacklevel=2)
