# Heads System Specification

**Status**: Design Document (No Implementation)  
**Date**: 2025-01-28  
**Scope**: How to architect "heads" (market-specific output derivers) without layering new systems.

---

## Overview

A **"head"** is a deterministic function that transforms a model's base representation (e.g., team strength ratings) into market-specific outputs (e.g., win probability, margin distribution, total distribution).

**Current State**: Models produce outputs directly via `.predict()` and projection engines fill gaps. This works but is implicit and fragmented.

**Proposed Design**: Explicit heads that declare what they require and produce, plugged into the forecast pipeline at a single integration point.

---

## Problem Statement

### Current Gaps

1. **Bradley-Terry lacks margin/total outputs**
   - Produces team ratings but no inherent SD
   - Workaround: Calibration back-solves SD from training residuals
   - Result: Can't be used directly in SPREAD/TOTAL ensembles without calibrator present

2. **Projection engine is implicit**
   - Utilities scattered across [src/pipelines/projections.py](src/pipelines/projections.py) and [src/models/calibration.py](src/models/calibration.py)
   - Unclear which model→output conversions are "official" vs. experimental
   - Hard to test in isolation

3. **No standard interface for derived outputs**
   - Each model implements `.predict()` returning `GamePrediction` with mixed native/derived fields
   - Downstream doesn't know which field is native vs. derived
   - Changes to projection logic require modifying model classes

4. **Difficulty auditing output origin**
   - `win_prob_source` tracks calibration but not derivation (BT vs. margin-normal vs. etc.)
   - Impossible to distinguish "BT's native p_win" from "BT's margin-derived p_win"

---

## Proposed Heads Architecture

### 1. Head Interface (Pseudocode / TypeScript-style)

```python
class Head(Protocol):
  """A deterministic market-specific output deriver."""
  
  @property
  def market(self) -> Market:
    """Which market does this head produce output for?"""
    
  @property
  def required_fields(self) -> set[str]:
    """What model attributes are required? (e.g., {"ratings", "calibration"})"""
    
  @property
  def produces(self) -> dict[str, type]:
    """What columns does this head produce?
    Examples:
      {"p_home_win": float, "p_away_win": float}  # For ML head
      {"margin_mean": float, "margin_sd": float}  # For SPREAD head
    """
  
  def can_produce_for(self, team_a: str, team_b: str) -> bool:
    """Can this head produce output for this matchup?
    (Some heads may skip if data incomplete.)"""
  
  def derive(
    self,
    team_a: str,
    team_b: str,
    venue: str = "neutral",
    model_context: dict = None
  ) -> dict[str, Any]:
    """Derive market output for a matchup.
    
    Args:
      team_a, team_b: Team names
      venue: "home" (a is home), "away", "neutral"
      model_context: Dict with model's native attributes
        Example: {"ratings": {...}, "calibration": {...}}
    
    Returns:
      Dict with keys matching self.produces
      Example: {"p_home_win": 0.55, "p_away_win": 0.45}
    """
  
  def derive_batch(
    self,
    games_df: pd.DataFrame,
    model_context: dict = None
  ) -> pd.DataFrame:
    """Vectorized derivation for multiple games.
    
    Args:
      games_df: Columns [home_team, away_team, ...]
      model_context: Model-specific context (e.g., ratings dict)
    
    Returns:
      DataFrame with columns matching self.produces
    """
```

---

### 2. Example: Bradley-Terry Heads (Three Options)

#### Option A: Thurstone-Mosteller / Probit-Noise Head

**Concept**: Assume margin ~ Normal(μ, σ) where μ = rating difference, σ learned via MLE.

```python
class BradleyTerryThurstoneMostellerHead:
  """
  Derive margin distribution from BT team ratings + fitted noise term.
  
  BT produces team strength θ_i. This head assumes:
    Margin ~ Normal(θ_home - θ_away, σ)
  
  σ is learned during fit by maximizing log-likelihood of observed margins.
  """
  
  def __init__(self, bt_model: BradleyTerry):
    self.bt = bt_model
    self.sigma_learned = None  # Fitted during fit()
  
  @property
  def market(self) -> Market:
    return Market.SPREAD
  
  @property
  def required_fields(self) -> set[str]:
    return {"ratings"}  # BT must have fitted ratings
  
  @property
  def produces(self) -> dict[str, type]:
    return {"margin_mean": float, "margin_sd": float}
  
  def fit(self, games_df: pd.DataFrame) -> None:
    """Learn σ from training games' margins."""
    # Compute predicted margins for each game
    predicted = []
    actual = []
    
    for _, game in games_df.iterrows():
      home, away = game["home_team"], game["away_team"]
      margin_pred = self.bt.ratings[home] - self.bt.ratings[away]
      margin_actual = game["home_score"] - game["away_score"]
      predicted.append(margin_pred)
      actual.append(margin_actual)
    
    # Residuals ~ Normal(0, σ)
    residuals = np.array(actual) - np.array(predicted)
    self.sigma_learned = np.std(residuals)  # MLE for σ
  
  def derive_batch(self, games_df, model_context=None):
    """Return (margin_mean, margin_sd) for each game."""
    margin_means = []
    margin_sds = []
    
    for _, game in games_df.iterrows():
      home, away = game["home_team"], game["away_team"]
      margin_mean = self.bt.ratings[home] - self.bt.ratings[away]
      margin_sd = self.sigma_learned or 10.0  # Fallback if not fitted
      
      margin_means.append(margin_mean)
      margin_sds.append(margin_sd)
    
    return pd.DataFrame({
      "margin_mean": margin_means,
      "margin_sd": margin_sds
    })
```

**Pros**:
- Theoretically principled (assumes Normal margin dist.)
- Single σ parameter learned from data
- Natural decomposition: mean from ratings, SD from residuals

**Cons**:
- Requires re-fitting BT model to include margin learning
- Single global σ may not capture home team variance heterogeneity
- Assumes margin is Normal (may not hold in practice)

---

#### Option B: Probability-to-Margin Inversion Head

**Concept**: Invert BT's `p_win` + assumed SD to get margin estimate.

```python
class BradleyTerryMarginInversionHead:
  """
  Invert p_win to margin assuming Normal distribution.
  
  If Margin ~ Normal(μ, σ), then p_win = Φ((μ) / σ).
  We invert: Φ^{-1}(p_win) ≈ μ / σ.
  
  If we assume a fixed or learned σ, we can back-solve μ.
  """
  
  def __init__(self, bt_model: BradleyTerry, sigma: float = 10.0):
    self.bt = bt_model
    self.sigma = sigma  # Assumed margin SD
  
  @property
  def produces(self) -> dict[str, type]:
    return {"margin_mean": float, "margin_sd": float}
  
  def derive_batch(self, games_df, model_context=None):
    """Invert p_win to margin."""
    margin_means = []
    margin_sds = []
    
    for _, game in games_df.iterrows():
      home, away = game["home_team"], game["away_team"]
      
      # Get BT's probability
      p_home = self.bt.predict_probability(home, away, venue="home" if game.get("neutral") == False else "neutral")
      
      # Invert: Φ^{-1}(p) ≈ μ / σ
      z = scipy.stats.norm.ppf(p_home)  # Inverse normal CDF
      margin_mean = z * self.sigma
      
      margin_means.append(margin_mean)
      margin_sds.append(self.sigma)
    
    return pd.DataFrame({
      "margin_mean": margin_means,
      "margin_sd": margin_sds
    })
```

**Pros**:
- Simple; reuses BT's existing ratings
- Closed-form solution (no fitting)
- Allows tunable σ

**Cons**:
- Assumes Normal margin distribution (may not hold)
- σ is fixed, doesn't adapt per matchup
- Loses information (margin data not used in fitting)

---

#### Option C: Joint Likelihood Head

**Concept**: Fit BT to both outcomes (win/loss) AND margin values in one optimization.

```python
class BradleyTerryJointHead:
  """
  Extend BT to jointly fit win outcomes + margins.
  
  Likelihood: L = Π p_win^{outcome} * Φ(margin / σ)
  
  Produces ratings + σ that maximize both signals.
  """
  
  def __init__(self, **params):
    self.ratings = {}
    self.sigma = 10.0
    self.params = params
  
  def fit(self, games_df):
    """Fit via maximum likelihood using both outcomes and margins."""
    # (Pseudocode; full optimization omitted)
    # Maximize: log(p_win if outcome=1 else 1-p_win) + log(pdf(margin | predicted_margin))
    
    # Gradient descent over ratings + sigma
    # ...
    
  def derive_batch(self, games_df, model_context=None):
    margin_means = []
    margin_sds = []
    for _, game in games_df.iterrows():
      home, away = game["home_team"], game["away_team"]
      margin_mean = self.ratings[home] - self.ratings[away]
      margin_sds.append(self.sigma)
    return pd.DataFrame({
      "margin_mean": margin_means,
      "margin_sd": margin_sds
    })
```

**Pros**:
- Captures both win and margin information
- Single principled optimization
- Learned σ adapts to data

**Cons**:
- Complex; requires custom optimizer
- Potential overfitting
- No longer "pure Bradley-Terry" (mixed objective)

---

### 3. Head Integration Point

**Where Heads Plug In**: Right after model fit, before `.predict()` is called.

```python
# Current flow:
model = BradleyTerry(...)
model.fit(games_df)
predictions = model.predict(upcoming_df)  # Returns GamePrediction with mixed native/derived

# Proposed flow:
model = BradleyTerry(...)
model.fit(games_df)

# Instantiate heads for this model
ml_head = model.create_ml_head()      # Native: p_win
spread_head = BradleyTerryThurstoneMostellerHead(model)  # Derived: margin
total_head = None  # BT doesn't support total

# For each upcoming game, use heads to derive outputs
predictions = []
for _, game in upcoming_df.iterrows():
  game_pred = GamePrediction(
    game_id=game["game_id"],
    home_team=game["home_team"],
    away_team=game["away_team"],
    # ... basic fields ...
  )
  
  # Use ML head for p_win
  if ml_head:
    ml_out = ml_head.derive(
      game["home_team"], game["away_team"],
      venue="home" if not game["neutral"] else "neutral"
    )
    game_pred.p_home_win = ml_out["p_home_win"]
    game_pred.win_prob_source = "native"  # BT's direct p_win
  
  # Use SPREAD head for margin
  if spread_head:
    spread_out = spread_head.derive_batch(game[["home_team", "away_team"]])
    game_pred.margin_mean = spread_out["margin_mean"].iloc[0]
    game_pred.margin_sd = spread_out["margin_sd"].iloc[0]
    game_pred.win_prob_source = "margin_thurstone"  # Derived via TM head
  
  # TOTAL not supported
  
  predictions.append(game_pred)
```

**Benefits**:
- Clear separation: which outputs are native vs. derived
- Testable in isolation
- Easy to swap implementations (TM vs. inversion vs. joint)
- Audit trail: `win_prob_source` can indicate derivation method

---

### 4. Avoiding Duplicates

**Key principle**: Each model produces each market's outputs via **exactly one head** (native or derived).

**Registry** (pseudocode):

```python
MODEL_HEAD_REGISTRY = {
  "elo": {
    "ML": EloMLHead(native=True),
    "SPREAD": EloMarginHead(derived="calibration"),  # Via calibrator
    "TOTAL": None,
  },
  "bradley-terry": {
    "ML": BradleyTerryMLHead(native=True),
    "SPREAD": BradleyTerryThurstoneMostellerHead(derived="probit"),
    "TOTAL": None,
  },
  "gssd": {
    "ML": GSSDMLHead(derived="margin_inversion"),
    "SPREAD": GSSDSpreadHead(native=True),
    "TOTAL": GSSDTotalHead(native=True),
  },
  "poisson": {
    "ML": PoissonMLHead(derived="normal_cdf"),
    "SPREAD": None,
    "TOTAL": PoissonTotalHead(native=True),
  },
}
```

**Rule**: When `build_forecasts_df()` calls model.predict(), it should:
1. Check registry for model's heads
2. For each market, use the designated head (if present)
3. Mark derivation method in `win_prob_source` (e.g., "poisson_normal_cdf")

---

### 5. Removal of Layered Attempts

**Problem**: Currently, multiple systems try to produce the same output:
- Model `.predict()` method
- Projection engine utilities
- Calibration back-solving
- Ensemble component JSON

**Solution**: Single head per (model, market) pair. No layering.

**Cleanup**:
1. Remove ad-hoc margin derivation from Elo's `.predict()` method (currently line ~450)
2. Move into `EloMarginHead`
3. Projection engine becomes utility library (no public interface); only used by heads

---

## How Calibration Attaches

Calibrators operate **after** heads have produced base predictions.

```
Model → [Heads derive outputs] → [Calibrators refine] → Final DataFrame
        (margin_mean, margin_sd)  (adjust params)
```

**Calibration contract**:
- **Input**: `pred_mean`, `pred_sd` (from head)
- **Output**: `calibrated_mean`, `calibrated_sd` (distribution-aware adjustment)
- **Append tag**: `"+calibrated_<market>"` to `win_prob_source`

**Example**:
```python
# After head produces margin
game_pred.margin_mean = 3.5  # From BradleyTerryThurstoneMostellerHead
game_pred.margin_sd = 10.2
game_pred.win_prob_source = "bradley-terry+probit"

# Calibrator adjusts
calibrator = load_latest_calibrator(..., market="SPREAD")
calib_in = pd.DataFrame({"pred_mean": [3.5], "pred_sd": [10.2]})
calib_out = calibrator.transform(calib_in)

game_pred.margin_mean = calib_out["calibrated_mean"].iloc[0]
game_pred.margin_sd = calib_out["calibrated_sd"].iloc[0]
game_pred.win_prob_source = "bradley-terry+probit+calibrated_spread"
```

---

## Testing Strategy

### 1. Unit Tests Per Head

```python
def test_bradley_terry_thurstone_head_derives_margin():
  bt = BradleyTerry()
  bt.fit(training_games)
  
  head = BradleyTerryThurstoneMostellerHead(bt)
  head.fit(training_games)
  
  result = head.derive("Lakers", "Celtics", venue="home")
  
  assert result["margin_mean"] is not None
  assert result["margin_sd"] > 0
  assert result["margin_sd"] == pytest.approx(head.sigma_learned, rel=0.01)
```

### 2. Integration Tests

```python
def test_build_forecasts_uses_heads():
  """Verify build_forecasts_df uses designated heads for each (model, market)."""
  
  forecast_df = build_forecasts_df(
    db_path="...",
    model="bradley-terry",
    ...
  )
  
  # SPREAD market should use TM head (derived)
  assert forecast_df["win_prob_source"].unique() == {"bradley-terry+probit"}
  
  # Should have both margin_mean and margin_sd
  assert forecast_df["margin_mean"].notna().all()
  assert forecast_df["margin_sd"].notna().all()
```

### 3. Equivalence Tests (Old vs. New)

```python
def test_new_heads_match_old_projections():
  """Ensure new head logic produces same results as old projection engine."""
  
  # Old way: model.predict() with implicit projection
  old_result = elo_model.predict(games_df)
  old_margin = old_result[0].margin_mean
  
  # New way: explicit head
  head = EloMarginHead(elo_model)
  new_result = head.derive(games_df.iloc[0]["home_team"], games_df.iloc[0]["away_team"])
  new_margin = new_result["margin_mean"]
  
  assert old_margin == pytest.approx(new_margin, rel=0.001)
```

---

## Implementation Roadmap (No Code Yet)

### Phase 1: Design & Spec (Current)
- [x] Document problem
- [x] Define Head interface
- [x] Show 3 Bradley-Terry options
- [x] Explain integration point
- [ ] Stakeholder review

### Phase 2: Implement ML Heads (Lowest Risk)
- [ ] Create `MLHead` base class
- [ ] Implement for all models (mostly "native" already)
- [ ] Add tests
- [ ] No breaking changes to forecasts

### Phase 3: Implement SPREAD Heads (Medium Risk)
- [ ] Design Bradley-Terry SPREAD head (pick option: TM, inversion, or joint)
- [ ] Implement for all models
- [ ] Add equivalence tests
- [ ] Verify calibration still applies

### Phase 4: Implement TOTAL Heads (Medium Risk)
- [ ] Similar to SPREAD
- [ ] Handle models that don't support TOTAL (return None)

### Phase 5: Cleanup (Breaking)
- [ ] Remove ad-hoc projection logic from model `.predict()` methods
- [ ] Deprecate old projection utilities
- [ ] Update ensemble integration to use heads

---

## Summary

A **heads system** makes output derivation explicit, testable, and auditable. Rather than scattering margin/total computation across model classes and projection engines, each (model, market) pair has a designated head responsible for producing that output.

For **Bradley-Terry**, the recommended approach is **Thurstone-Mosteller** (Option A): fit a single noise term σ from margin residuals during model fit, then deterministically derive margins for new games. This is theoretically clean and compatible with calibration.

**Integration** happens at the forecast building step: after model fit, instantiate heads and use them instead of relying on implicit projection logic.

**No implementation in this task**—just design and options for future work.

