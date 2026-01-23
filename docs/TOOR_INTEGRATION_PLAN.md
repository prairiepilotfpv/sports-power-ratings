# TOOR Model Integration Plan
## Integrating Example TOOR's Best Practices into Current Implementation

**Date:** January 23, 2026  
**Goal:** Adopt vectorized predictions, two-stage Bradley-Terry mapping, cleaner helper extraction, and format flexibility while maintaining all canon rules, current commands, and production features.

---

## Executive Summary

This plan integrates four key improvements from the example TOOR model:
1. **Vectorized prediction loop** (10-100x speedup)
2. **Two-stage Bradley-Terry → margin mapping** (cleaner architecture)
3. **Cleaner helper method extraction** (better testing/debugging)
4. **Format flexibility in return types** (raw vs. structured outputs)

**Core Principle:** Adapt the example TO our system, not our system to the example.

---

## Dependencies & Compatibility

### Current Dependencies (Keep)
- `scipy` - already in requirements (via scikit-learn)
- All current production libraries (pandas, numpy, etc.)

### New Dependencies (Add)
- None required! Example uses scipy which is already available via scikit-learn dependency

### Canon Compliance Checklist
- ✅ Keep `GamePrediction` as canonical output DTO
- ✅ Keep `BaseModel` interface (`fit()`, `predict()`, `project_matchup()`)
- ✅ Keep `ModelMetadata` reporting
- ✅ Keep all production features (recency weighting, conditional SD, guardrails)
- ✅ Maintain `require_columns` validation
- ✅ Keep current CLI commands unchanged
- ✅ Maintain backtest/schedule/matchup workflows

---

## Phase 1: Add Helper Methods (Low Risk)

**Goal:** Extract reusable computation logic into testable helper methods without changing external behavior.

### 1.1 Add Team Indexing Helpers

**New Methods:**
```python
def _build_team_index(self) -> dict[str, int]:
    """Build team name → index mapping from current ratings."""
    return {team: idx for idx, team in enumerate(sorted(self._rating_model._ratings.keys()))}

def _get_team_indices(
    self, 
    home_teams: np.ndarray, 
    away_teams: np.ndarray,
    team_index: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Convert team names to indices for vectorized lookup."""
    home_idx = np.array([team_index.get(h, -1) for h in home_teams])
    away_idx = np.array([team_index.get(a, -1) for a in away_teams])
    return home_idx, away_idx
```

**Location:** Add to `TOORModel` class in `src/models/toor.py`

**Testing:** Add unit tests in `tests/models/test_toor_helpers.py`

### 1.2 Add Margin Computation Helper

**New Method:**
```python
def _compute_margin_predictions(
    self,
    home_strengths: np.ndarray,
    away_strengths: np.ndarray,
    neutral_flags: np.ndarray,
) -> np.ndarray:
    """Vectorized margin prediction from team strengths.
    
    Args:
        home_strengths: Array of home team strength values
        away_strengths: Array of away team strength values
        neutral_flags: Boolean array indicating neutral venue
        
    Returns:
        Array of predicted margins (home - away)
    """
    coefficients = self._coefficients
    home_adv_contrib = coefficients.home_advantage * (~neutral_flags).astype(float)
    return (
        home_adv_contrib
        + coefficients.home_coeff * home_strengths
        + coefficients.away_coeff * away_strengths
    )
```

**Location:** Add to `TOORModel` class

**Testing:** Verify against current single-game predictions

### 1.3 Add Prediction Formatting Helper

**New Method:**
```python
def _format_game_predictions(
    self,
    games_df: pd.DataFrame,
    pred_margins: np.ndarray,
    pred_totals: np.ndarray,
    margin_sds: np.ndarray,
    total_sds: np.ndarray,
    p_home_wins: np.ndarray,
    win_prob_dists: list[list[dict[str, float]]],
) -> list[GamePrediction]:
    """Convert vectorized predictions to canonical GamePrediction objects."""
    predictions = []
    coefficients = self._coefficients
    model_identity = self.metadata().identity_dict()
    
    for i, row in enumerate(games_df.to_dict(orient="records")):
        predictions.append(
            GamePrediction(
                game_id=str(row.get("game_id", f"{row['date']}_{row['home_team']}_{row['away_team']}")),
                date=str(row.get("date", "")),
                home_team=str(row["home_team"]),
                away_team=str(row["away_team"]),
                p_home_win=float(p_home_wins[i]),
                win_prob_dist=win_prob_dists[i] if win_prob_dists else None,
                pred_margin=float(pred_margins[i]),
                pred_total=float(pred_totals[i]),
                margin_sd=float(margin_sds[i]),
                total_sd=float(total_sds[i]),
                total_mean=float(pred_totals[i]),
                win_prob_source="logistic",
                margin_dist_assumption="normal_approx",
                metadata=dict(model_identity),
                extra={
                    "home_advantage": coefficients.home_advantage,
                    "home_coeff": coefficients.home_coeff,
                    "away_coeff": coefficients.away_coeff,
                    "error_term": coefficients.error_term,
                    "win_prob_k": self._win_prob_k,
                    "winprob_bias": self._win_prob_bias,
                    "conditional_sd": self._conditional_sd,
                    # ... (keep all existing extra fields)
                },
            )
        )
    return predictions
```

**Location:** Add to `TOORModel` class

---

## Phase 2: Vectorize Prediction Loop (Medium Risk)

**Goal:** Replace iterative `predict()` with vectorized numpy operations.

### 2.1 Refactor `predict()` Method

**Current Pattern (lines 589-661):**
```python
for row in upcoming_games_df.to_dict(orient="records"):
    home = str(row.get("home_team", "")).strip()
    away = str(row.get("away_team", "")).strip()
    # ... single game computation
    predictions.append(GamePrediction(...))
```

**New Pattern:**
```python
def predict(self, upcoming_games_df: Any) -> list[GamePrediction]:
    require_columns(upcoming_games_df, ["date", "home_team", "away_team"])
    
    if len(upcoming_games_df) == 0:
        return []
    
    # Extract vectorized inputs
    home_teams = upcoming_games_df["home_team"].astype(str).str.strip().values
    away_teams = upcoming_games_df["away_team"].astype(str).str.strip().values
    neutral_raw = upcoming_games_df.get("neutral", pd.Series([False] * len(upcoming_games_df)))
    neutral_flags = neutral_raw.fillna(False).astype(bool).values
    
    # Build team index and lookup strengths
    team_index = self._build_team_index()
    strengths = self._rating_model.signed_strengths()
    home_idx, away_idx = self._get_team_indices(home_teams, away_teams, team_index)
    
    # Vectorized strength lookup with fallback to 0.0 for unknown teams
    strength_array = np.array([strengths.get(t, 0.0) for t in strengths.keys()])
    home_strengths = np.where(home_idx >= 0, strength_array[home_idx], 0.0)
    away_strengths = np.where(away_idx >= 0, strength_array[away_idx], 0.0)
    
    # Vectorized margin predictions
    pred_margins = self._compute_margin_predictions(home_strengths, away_strengths, neutral_flags)
    
    # Vectorized total predictions (league average)
    pred_totals = np.full(len(upcoming_games_df), self._total_mean or DEFAULT_TOTAL_MEAN_FALLBACK)
    
    # Vectorized margin SD computation
    margin_sds = self._compute_margin_sds_vectorized(pred_margins, upcoming_games_df)
    
    # Total SD (constant league-wide)
    total_sds = np.full(len(upcoming_games_df), self._total_sd or DEFAULT_TOTAL_SD_FALLBACK)
    
    # Vectorized win probabilities
    p_home_wins, win_prob_dists = self._compute_win_probs_vectorized(
        pred_margins, margin_sds, upcoming_games_df
    )
    
    # Format into canonical GamePrediction objects
    return self._format_game_predictions(
        upcoming_games_df, pred_margins, pred_totals, 
        margin_sds, total_sds, p_home_wins, win_prob_dists
    )
```

### 2.2 Add Vectorized SD Helper

**New Method:**
```python
def _compute_margin_sds_vectorized(
    self, 
    pred_margins: np.ndarray,
    games_df: pd.DataFrame,
) -> np.ndarray:
    """Compute margin standard deviations for all predictions.
    
    Uses conditional SD model if available, otherwise constant error_term.
    """
    if self._conditional_sd_model is not None:
        # Vectorized conditional SD prediction
        return self._conditional_sd_model.predict_vectorized(
            pred_margins,
            guardrail_min=MARGIN_SD_GUARDRAIL_MIN,
            guardrail_max=MARGIN_SD_GUARDRAIL_MAX,
            fallback_sd=LEAGUE_MARGIN_SD_DEFAULT,
        )
    else:
        # Constant error term for all predictions
        raw_sd = self._coefficients.error_term
        margin_sd, _ = guardrail_margin_sd(
            raw_sd,
            fallback_sd=LEAGUE_MARGIN_SD_DEFAULT,
            guardrail_min=MARGIN_SD_GUARDRAIL_MIN,
            guardrail_max=MARGIN_SD_GUARDRAIL_MAX,
        )
        return np.full(len(pred_margins), margin_sd)
```

### 2.3 Add Vectorized Win Probability Helper

**New Method:**
```python
def _compute_win_probs_vectorized(
    self,
    pred_margins: np.ndarray,
    margin_sds: np.ndarray,
    games_df: pd.DataFrame,
) -> tuple[np.ndarray, list[list[dict[str, float]]]]:
    """Compute win probabilities and distributions for all predictions."""
    win_prob_k = self._win_prob_k if self._win_prob_k > 0 else DEFAULT_WIN_PROB_K
    
    # Vectorized projected spread with bias adjustment
    projected_spreads = -pred_margins
    adjusted_spreads = projected_spreads - self._win_prob_bias
    
    # Vectorized logistic win probability
    p_home_wins = logistic_win_prob(adjusted_spreads, win_prob_k)
    
    # Vectorized win probability distributions
    win_prob_dists = [
        win_prob_distribution(p_home_win, win_prob_k=win_prob_k, margin_std=margin_sd)
        for p_home_win, margin_sd in zip(p_home_wins, margin_sds)
    ]
    
    return p_home_wins, win_prob_dists
```

**Testing:**
- Add `test_vectorized_predictions_match_iterative` to verify identical outputs
- Add `test_vectorized_predictions_performance` to verify speedup
- Add edge case tests (empty df, missing teams, all neutrals)

---

## Phase 3: Two-Stage Bradley-Terry Architecture (High Risk - Optional)

**Goal:** Adopt Bradley-Terry logistic ratings as the first stage, then map to margins in second stage.

### 3.1 Decision Point: Adopt or Skip?

**Pros of Two-Stage Approach:**
- Cleaner separation of concerns (probabilistic ratings → margins)
- Avoids exponentiation/stabilization complexity (lines 179-189)
- More mathematically principled (Bradley-Terry is proven for pairwise comparisons)
- Easier to understand and maintain

**Cons:**
- Requires adding `BradleyTerry` dependency to TOOR
- More complex architectural change
- Current approach works and is tested
- Risk of subtle regression

**Recommendation:** Start with **Skip** (keep current OLS approach) and revisit later if mathematical issues arise.

### 3.2 If Adopting: Implementation Plan

**Step 1:** Add Bradley-Terry as internal component
```python
class TOORPowerRating:
    def __init__(self, ...):
        self._bt_solver = BradleyTerry(
            max_iter=max_iter,
            tol=tol,
            learn_hfa=False,  # TOOR learns its own HFA
        )
```

**Step 2:** Fit Bradley-Terry first, then map to margins
```python
def fit(self, games: Iterable[Mapping[str, Any]], ...) -> None:
    # Stage 1: Fit Bradley-Terry logistic ratings
    self._bt_solver.fit(games)
    
    # Stage 2: Map BT ratings to margin coefficients via OLS
    bt_ratings = self._bt_solver.ratings
    
    # Build design matrix using BT logistic ratings
    for game in games:
        home_rating = bt_ratings[home]
        away_rating = bt_ratings[away]
        design_matrix.append([home_advantage, home_rating, away_rating])
        margins.append(margin)
    
    # Fit coefficients [home_adv, home_coef, away_coef]
    coeffs = weighted_least_squares(matrix, target, weights)
```

**Step 3:** Update `signed_strengths()` to return BT logistic ratings

**Testing:** Full regression suite to ensure no metric dropout

---

## Phase 4: Add Format Flexibility (Low Risk)

**Goal:** Support both canonical `GamePrediction` objects and raw numpy arrays.

### 4.1 Add `format` Parameter

**New Signature:**
```python
def predict(
    self, 
    upcoming_games_df: Any, 
    format: str = "canonical"  # "canonical" | "array" | "dataframe"
) -> list[GamePrediction] | np.ndarray | pd.DataFrame:
```

**Implementation:**
```python
def predict(self, upcoming_games_df: Any, format: str = "canonical") -> Any:
    # ... (vectorized computation as in Phase 2)
    
    if format == "array":
        # Return raw numpy arrays for optimization/tuning
        return np.column_stack([
            pred_margins,
            pred_totals,
            p_home_wins,
            margin_sds,
            total_sds,
        ])
    elif format == "dataframe":
        # Return pandas DataFrame for analysis
        return pd.DataFrame({
            "game_id": upcoming_games_df["game_id"],
            "date": upcoming_games_df["date"],
            "home_team": upcoming_games_df["home_team"],
            "away_team": upcoming_games_df["away_team"],
            "pred_margin": pred_margins,
            "pred_total": pred_totals,
            "p_home_win": p_home_wins,
            "margin_sd": margin_sds,
            "total_sd": total_sds,
        })
    else:  # format == "canonical" (default)
        return self._format_game_predictions(...)
```

### 4.2 Update `project_matchup()` for Raw Output

**New Signature:**
```python
def project_matchup(
    self,
    home_team: str,
    away_team: str,
    *,
    neutral: bool = False,
    sport: str | None = None,
    date: str | None = None,
    game_id: str | None = None,
    format: str = "dict",  # "dict" | "array"
) -> dict[str, Any] | np.ndarray:
```

**Use Case:** Hyperparameter tuning can use `format="array"` for 10x faster optimization loops.

---

## Phase 5: Enhanced Conditional SD Model (Medium Risk)

**Goal:** Add vectorized prediction method to `ConditionalSDModel`.

### 5.1 Extend `ConditionalSDModel` Class

**Location:** `src/models/calibration.py`

**New Method:**
```python
def predict_vectorized(
    self,
    predictions: np.ndarray,
    *,
    guardrail_min: float,
    guardrail_max: float,
    fallback_sd: float,
) -> np.ndarray:
    """Vectorized SD prediction for multiple margin predictions.
    
    Args:
        predictions: Array of predicted margins
        guardrail_min: Minimum allowed SD
        guardrail_max: Maximum allowed SD
        fallback_sd: Fallback when prediction is out of bounds
        
    Returns:
        Array of conditional standard deviations
    """
    # Linear conditional SD model: sd = intercept + slope * |margin|
    abs_margins = np.abs(predictions)
    raw_sds = self.intercept + self.slope * abs_margins
    
    # Vectorized guardrail clamping
    sds = np.clip(raw_sds, guardrail_min, guardrail_max)
    
    # Replace invalid values with fallback
    sds = np.where(np.isfinite(sds), sds, fallback_sd)
    
    return sds
```

**Testing:** Add `test_conditional_sd_vectorized` to verify match with iterative version

---

## Phase 6: Optimization Improvements (Low-Medium Risk)

**Goal:** Adopt scipy.optimize for more robust coefficient fitting.

### 6.1 Add Scipy Optimizer Option

**New Parameter:**
```python
def __init__(
    self,
    *,
    optimizer: str = "ols",  # "ols" | "lbfgsb" | "slsqp"
    ...
):
    self._optimizer = optimizer
```

### 6.2 Add Optimization Helper

**New Method:**
```python
def _fit_coefficients_scipy(
    self,
    design_matrix: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None,
) -> ToorCoefficients:
    """Fit TOOR coefficients using scipy.optimize.minimize."""
    from scipy.optimize import minimize
    
    def objective(params):
        """Sum of squared errors for optimization."""
        predictions = design_matrix @ params
        residuals = target - predictions
        if weights is not None:
            return np.sum((residuals ** 2) * weights)
        return np.sum(residuals ** 2)
    
    # Initial guess
    x0 = np.array([
        DEFAULT_COEFFICIENTS.home_advantage,
        DEFAULT_COEFFICIENTS.home_coeff,
        DEFAULT_COEFFICIENTS.away_coeff,
    ])
    
    # Try multiple methods with fallback
    for method in ["L-BFGS-B", "SLSQP"]:
        result = minimize(
            objective,
            x0,
            method=method,
            options={"ftol": self._tol, "maxiter": self._max_iter},
        )
        if result.success:
            break
    
    if not result.success:
        # Fall back to OLS
        return self._fit_coefficients_ols(design_matrix, target, weights)
    
    # Compute error term
    predictions = design_matrix @ result.x
    residuals = target - predictions
    error_term = weighted_rmse(residuals, weights)
    
    return ToorCoefficients(
        home_advantage=float(result.x[0]),
        home_coeff=float(result.x[1]),
        away_coeff=float(result.x[2]),
        error_term=error_term,
    )
```

**Usage in `fit()`:**
```python
if self._optimizer in ["lbfgsb", "slsqp"]:
    self._coefficients = self._fit_coefficients_scipy(matrix, target, weight_arr)
else:
    # Current OLS approach
    coeffs = weighted_least_squares(matrix, target, weight_arr)
```

---

## Implementation Sequence

### Week 1: Foundation (Phase 1)
- [ ] Add helper methods (_build_team_index, _get_team_indices, _compute_margin_predictions)
- [ ] Add tests for helpers
- [ ] Verify no external behavior change

### Week 2: Vectorization (Phase 2)
- [ ] Implement vectorized predict()
- [ ] Add _compute_margin_sds_vectorized()
- [ ] Add _compute_win_probs_vectorized()
- [ ] Add _format_game_predictions()
- [ ] Comprehensive testing (correctness + performance)
- [ ] Run full backtest suite to verify no metric dropout

### Week 3: Format Flexibility (Phase 4)
- [ ] Add format parameter to predict()
- [ ] Add format parameter to project_matchup()
- [ ] Update conditional SD model with predict_vectorized()
- [ ] Test all format variants
- [ ] Update relevant tests

### Week 4: Optional Enhancements (Phase 6)
- [ ] Add scipy optimizer option
- [ ] Test optimizer convergence
- [ ] Benchmark against current OLS

### Future: Bradley-Terry Integration (Phase 3 - Optional)
- [ ] Evaluate need based on Phase 2 results
- [ ] If adopting, implement two-stage architecture
- [ ] Full regression testing

---

## Testing Strategy

### Unit Tests
- `test_toor_helpers.py`: Test all new helper methods
- `test_toor_vectorized.py`: Vectorized operations
- `test_toor_formats.py`: Output format variants

### Integration Tests
- `test_toor_canon_contracts.py`: Verify canon compliance (already exists)
- `test_toor_predictions_match.py`: Vectorized == iterative outputs
- `test_toor_performance.py`: Measure speedup

### Regression Tests
- Run full backtest suite on NBA 2024-25
- Run full backtest suite on NHL 2025-26
- Verify all metrics (log_loss, brier, MAE margin, MAE total)
- Verify no metric dropout
- Compare to baseline results

### Performance Benchmarks
- Measure prediction time for 100, 1000, 10000 games
- Target: 10-50x speedup for large batches
- Ensure no memory explosion

---

## Risks & Mitigations

### Risk 1: Vectorization Introduces Subtle Bugs
**Mitigation:** 
- Comprehensive equivalence testing (vectorized == iterative)
- Keep iterative version as fallback during transition
- Test edge cases (empty df, unknown teams, all neutral)

### Risk 2: Performance Regression
**Mitigation:**
- Benchmark before/after
- Profile hotspots
- Ensure numpy operations don't cause memory spikes

### Risk 3: Breaking Existing Workflows
**Mitigation:**
- Default `format="canonical"` preserves current behavior
- All existing CLI commands unchanged
- Backward compatibility for all public methods

### Risk 4: Conditional SD Vectorization Edge Cases
**Mitigation:**
- Test with guardrails enabled/disabled
- Test with missing/invalid data
- Verify fallback behavior

---

## Success Criteria

1. ✅ All existing tests pass
2. ✅ Vectorized predictions match iterative predictions exactly
3. ✅ 10-50x speedup for large prediction batches
4. ✅ All canon contracts satisfied (no metric dropout)
5. ✅ All CLI commands work unchanged
6. ✅ Backtest results match baseline within numerical precision
7. ✅ No breaking changes to public API (except optional format parameter)

---

## Rollback Plan

If any phase introduces regressions:
1. Keep vectorized code in separate methods
2. Add `use_vectorized=False` parameter to TOORModel.__init__()
3. Conditional dispatch in predict(): if use_vectorized: ... else: (old code)
4. This allows A/B testing in production

---

## Dependencies Update

**requirements.txt additions:**
```
# scipy already included via scikit-learn dependency
# No new dependencies required!
```

**Optional performance dependencies (future):**
```
# numba==0.59.0  # JIT compilation for hot loops (future optimization)
```

---

## Documentation Updates

### docs/CLI.md
- Add note about vectorized prediction performance improvements
- Document format parameter for advanced users

### docs/model_canonization_playbook.md
- Add section on vectorized prediction patterns
- Document helper method extraction best practices

### src/models/toor.py docstrings
- Update predict() docstring with format parameter
- Add docstrings to all new helper methods
- Document vectorization guarantees

---

## Notes

- **Preserve all production features**: Recency weighting, conditional SD, guardrails, win-prob bias
- **Maintain canon compliance**: GamePrediction output, BaseModel interface, ModelMetadata
- **No CLI changes**: All commands work identically
- **Backward compatible**: Default format="canonical" preserves current behavior
- **Performance first**: Vectorization is the biggest win (10-100x speedup)
- **Incremental rollout**: Each phase is independently testable and deployable
- **Optional BT integration**: Two-stage architecture is high-risk, low-immediate-value; defer until proven need

---

## Open Questions

1. Should we add `use_vectorized` flag for gradual rollout?
2. Should we add `predict_proba()` method like example for probability-only use cases?
3. Should we expose raw Bradley-Terry ratings for inspection/debugging?
4. Should we add batch-size tuning for memory optimization on huge prediction sets?

---

**Next Steps:**
1. Review and approve plan
2. Create feature branch: `feature/toor-vectorization`
3. Start with Phase 1 (helpers) - lowest risk, easiest testing
4. Proceed incrementally with full test coverage at each phase
