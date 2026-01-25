# Architectural Audit: Is Your System Optimized for Accuracy or Just Wired for ML?

## Executive Summary

**Your system IS heavily ML-centric, and this is causing spread AND total problems.**

You have separate ensembles (good!), but the underlying issue is more fundamental: your models output **margin estimates**, and everything else is derived from margins using mathematical transformations rather than being predicted directly.

This creates a **compounding error problem**:
- Margin estimate is inherently uncertain
- Spread = -Margin (simple enough)
- Win Prob = Normal CDF of Margin (introduces distribution assumption)
- Total = Season Average + (Home Rating + Away Rating) * Slope (derived indirectly)

When margins are wrong, everything downstream is wrong.

---

## Architecture Audit

### What Your System Actually Does

```
Models (5 types)
    ↓
Each predicts: [Rating/Skill, Margin Estimate, Total (optional)]
    ↓
Ensemble per market (ML, SPREAD, TOTAL)
    ↓
    ML Ensemble:        Average of p_home_win values
    SPREAD Ensemble:    Average of margin_mean + between-model variance
    TOTAL Ensemble:     Average of total_mean + between-model variance
    ↓
Probability Generation
    ↓
    Spread Prob:  Normal CDF of (margin_mean, margin_sd)
    ML Prob:      Normal CDF of margin + Logistic transform (double transformation!)
    Total Prob:   Normal CDF of (total_mean, total_sd=22.0 fixed)
```

### The Root Problem: Margin Dependency

```
╔════════════════════════════════════════════════╗
║       MARGIN is the SINGLE SOURCE OF TRUTH     ║
║  (All other markets derive from it)            ║
╚════════════════════════════════════════════════╝

ELO/Bradley-Terry/TOOR:
    Rating → Margin estimate (via rating difference)
    
Spread Market:
    Margin → Direct spread
    
ML Market:
    Margin → Normal CDF → Win Prob
    (But also tries to get p_home_win directly from some models)
    
TOTAL Market:
    Team ratings → (intercept + slope * (rating_sum))
    (Not based on margin at all, actually independent!)
```

Wait - TOTALS are actually independent in `total_from_ratings()`. Let me re-examine...

Actually, looking at the code:

```python
# In projections.py
total_from_ratings():
    Uses: home_rating + away_rating * slope
    This is a RATING-based calculation, not margin-based
    
But in models/toor.py:
    TOOR calculates: _total_mean and _total_sd from actual game totals
    This is margin-independent
```

So totals **should** be independent. Let me check what's actually happening in the forecast pipeline...

### Actually: The Real Issue (More Subtle)

Looking at ensemble implementations:

**ML Ensemble** (`ml_v1.py`):
```python
def combine(forecast_df):
    # Takes p_home_win from each model
    # Returns weighted average
    # This is CORRECT - direct market prediction
```

**Spread Ensemble** (`spread_v1.py`):
```python
def combine(game_rows):
    # Takes margin_mean, margin_sd from each model
    # Returns weighted average margin + between-model variance
    # Then elsewhere: Normal CDF(margin) → Spread Probability
    # This is CORRECT - direct market prediction
```

**Total Ensemble** (`total_v1.py`):
```python
def combine(game_rows):
    # Takes total_mean, total_sd from each model
    # Returns weighted average total + between-model variance
    # Then elsewhere: Normal CDF(total) → Total Probability
    # This is CORRECT - direct market prediction
```

So actually your **ensemble structure is fine**. The issue is different:

---

## The REAL Problem: Model Output Quality

### What's Actually Happening

1. **Models trained on MARGIN** (because it's the most direct signal)
   - ELO: optimized for predicting who wins (margin is indirect)
   - Bradley-Terry: optimized for pairwise comparisons (margin is indirect)
   - TOOR: explicitly trains on margin
   - Poisson: trains on team scoring, which relates to margin
   - GSSD: margin-based (goal differential)

2. **Totals are treated as secondary**
   - Most models don't directly predict totals
   - TOOR computes total_mean/sd as post-fit statistics
   - Poisson can predict scoring, but is it used?
   - Others fall back to `total_from_ratings()` which is a derivative

3. **Each model may have different quality on different markets**
   ```
   Model          ML Accuracy    Spread Accuracy    Total Accuracy
   TOOR           High           High               Medium (derived)
   Bradley-Terry  High           High               Low (not designed)
   ELO            High           High               Medium (derived)
   Poisson        Medium         Medium             High (direct scoring)
   GSSD           Medium         Medium             Medium (derived)
   ```

   But your ensemble uses **same weights across all markets** (or does it?).

---

## Questions to Answer

### 1. Do You Have Per-Market Ensemble Weights?

Looking at your code: **Yes**, you have separate weight files:
- `outputs/ensembles/<sport>/<season>/ML/ensemble_ml_v1.json`
- `outputs/ensembles/<sport>/<season>/SPREAD/ensemble_spread_v1.json`
- `outputs/ensembles/<sport>/<season>/TOTAL/ensemble_total_v1.json`

**But are these weights actually tuned per-market?**
- If they were tuned together (e.g., "optimize for profit on all three markets"), they're not optimal
- If they were tuned separately, they're better

### 2. How Are Model Totals Actually Generated?

```python
# In models/toor.py:
supports_total=True  # Some models claim to support total
_total_mean = np.mean(totals_from_training_data)
_total_sd = np.std(totals_from_training_data)

# In pipelines/projections.py:
projection.projected_total  # Where does this come from?
```

**For most models**: `projected_total` is probably **None** or derived from `total_from_ratings()`.
**Only TOOR**: Actually computes its own total forecast.

### 3. Are Models Cross-Correlated?

If all 5 models use similar underlying logic:
- They'll all be wrong in the same direction
- Ensemble weights won't help
- Example: If all predict margin too high → all predict totals too high (indirectly)

---

## Your Specific Problems Explained

### Spread Over-Confidence

**Current hypothesis**:
- Models predict margin too high (e.g., +8 when it should be +5)
- Normal CDF magnifies this: spread -8 is further from 0 than margin estimate warrants
- Result: Overconfident in margin-based predictions

**Is it ML-driven?**
- Not exactly, but spread predictions ARE based on margin
- If margins are consistently biased, spreads will be too

### Total Over-Confidence  

**Current hypothesis**:
- Total predictions come from: `intercept + slope * (rating_sum)`
- This is derived from **season-long average** (training data)
- January has lower-scoring games, but your model uses season average
- Your recency adjustment partially fixes this reactively

**Is it ML-driven?**
- Less directly, but there's a subtle connection:
- If ensemble weights are tuned to optimize all three markets together, the total weights might be suboptimal
- Some models (TOOR, GSSD, BT) were never designed to predict totals directly

---

## The Real Issue: Models Aren't Independent

```
Ideal System:
├─ Model A: Optimized for ML
├─ Model B: Optimized for Spreads
├─ Model C: Optimized for Totals
└─ Combine independently per market

Your System:
├─ ELO: Optimized for "who wins" → used for ML, SPREAD, TOTAL
├─ Bradley-Terry: Optimized for "head-to-head" → used for ML, SPREAD, TOTAL
├─ TOOR: Optimized for "margin" → used for SPREAD, with TOTAL as afterthought
├─ Poisson: Optimized for "scoring" → could be good for TOTAL if properly used
└─ GSSD: Optimized for "goal differential" → probably mediocre on all

Compounded by:
└─ Single ensemble weight set, or poorly tuned per-market weights
```

---

## Is This Actually the Problem?

Let me check what you're observing:

**If you see**:
- ✅ Spreads ALSO over-confident (not just totals) → **Yes, this is the root cause**
- ✅ Totals worse than spreads → **Yes, models aren't designed for it**
- ✅ Some models consistently wrong on totals → **Yes, weights aren't optimized per-market**
- ❌ Only totals wrong, spreads fine → **Maybe different issue, see next section**

---

## How to Know If This Is Really the Problem

Run this diagnostic:

```python
# For last 50 games:
actual_margins = [home_score - away_score for each game]
actual_totals = [home_score + away_score for each game]

# Per model, compute:
for model in ['TOOR', 'BT', 'ELO', 'Poisson', 'GSSD']:
    spread_preds = get_model_predictions(model, 'SPREAD')
    total_preds = get_model_predictions(model, 'TOTAL')
    
    spread_mae = MAE(spread_preds.margin_mean, actual_margins)
    total_mae = MAE(total_preds.total_mean, actual_totals)
    
    spread_calib = calibration_error(spread_preds.margin_prob, actual_spreads_covered)
    total_calib = calibration_error(total_preds.over_prob, actual_totals_over)
    
    print(f"{model:12} | Spread MAE: {spread_mae:5.2f} | Total MAE: {total_mae:5.2f}")
    print(f"{' ':12} | Spread Cal: {spread_calib:5.3f} | Total Cal: {total_calib:5.3f}")
```

If you see:
```
TOOR         | Spread MAE:  3.2  | Total MAE:  8.5  ← Terrible at totals
Bradley-Terry| Spread MAE:  4.1  | Total MAE:  12.4 ← Not designed for it
ELO          | Spread MAE:  3.5  | Total MAE:  9.2  ← Mediocre at totals
Poisson      | Spread MAE:  4.8  | Total MAE:  2.1  ← Great at totals!
GSSD         | Spread MAE:  3.8  | Total MAE:  10.1 ← Not designed for it
```

Then **YES**, you need model-specific market assignments.

---

## What to Do About It

### Option 1: Separate Model Assignments (Best)
```python
TOTAL_MODELS = ['Poisson']  # Only use Poisson for totals
SPREAD_MODELS = ['TOOR', 'ELO', 'Bradley-Terry', 'GSSD']  # Better at spreads
ML_MODELS = ['TOOR', 'ELO', 'Bradley-Terry', 'GSSD']  # All work for ML
```

Pros: Eliminates models bad at their market  
Cons: Reduces ensemble diversity

### Option 2: Per-Market Weight Tuning (Medium)
```python
# Tune ensemble weights separately for each market
# Using market-specific backtests
# TOTAL weights might be very different from SPREAD weights

# Example outcome:
ML weights:     {TOOR: 0.3, BT: 0.25, ELO: 0.25, Poisson: 0.1, GSSD: 0.1}
SPREAD weights: {TOOR: 0.35, BT: 0.2, ELO: 0.3, Poisson: 0.05, GSSD: 0.1}
TOTAL weights:  {TOOR: 0.1, BT: 0.05, ELO: 0.1, Poisson: 0.6, GSSD: 0.15}
```

Pros: Uses all models but weights them properly per-market  
Cons: Requires backtesting per market

### Option 3: Build Total-Specific Models (Best Long-term)
```python
# Instead of deriving totals from margins:
# 1. Train separate scoring models for home/away teams
# 2. Combine home_score_pred + away_score_pred → total

class POISSONTotalModel:
    def predict_total(self, home, away):
        home_ppg = fit_poisson_ppg(home_team_games)
        away_ppg = fit_poisson_ppg(away_team_games)
        return home_ppg + away_ppg
```

Pros: Correct approach to totals  
Cons: Needs new model implementation

---

## Summary of Your Real Problem

| Aspect | Status | Impact |
|--------|--------|--------|
| Do you have separate ensembles? | ✅ Yes | Good |
| Are they tuned per-market? | ❓ Unknown | Critical |
| Do models output good totals? | ❌ Probably not | **The core issue** |
| Are models correlated? | ✅ Likely | Reduces ensemble benefit |
| Is margin dependency causing issues? | 🟡 Partially | Indirect effect on totals |
| Is it ML-centric? | 🟡 Partially | Not the main problem |

---

## What I Need to Know

To give you specific fixes, answer these:

1. **Are your TOTAL ensemble weights tuned separately from SPREAD/ML weights?**
   - If yes, what are they?
   - If no, that's a quick win

2. **Which models actually output usable total predictions?**
   - Run the diagnostic above

3. **Do you want to keep using all 5 models for totals, or are you open to using only the best ones?**

4. **How much backtesting effort are you willing to invest?**

This will determine whether you need a quick fix (better per-market weighting) or a deeper redesign (separate total-specific models).
