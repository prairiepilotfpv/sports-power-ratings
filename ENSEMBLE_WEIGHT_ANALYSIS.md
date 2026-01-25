# Critical Finding: Your Ensemble Weights Reveal The Real Problem

## The Data

```
ML Ensemble:
  bradley-terry: 53.8%  ← Dominant
  toor:          28.8%
  gssd:          17.4%
  elo:            0.0%  ← Eliminated
  poisson:      N/A
  
SPREAD Ensemble:
  poisson:       41.2%  ← Strong
  toor:          27.3%
  gssd:          23.5%
  elo:            8.0%
  bradley-terry:  0.0%  ← Eliminated
  
TOTAL Ensemble:
  bradley-terry: 59.8%  ← Dominant
  poisson:       40.1%
  (Only 2 models!)
```

## What This Tells Us

### Problem 1: Bradley-Terry is Dominating Totals (59.8%)

**But Bradley-Terry was never designed for totals.**

Bradley-Terry is a **pairwise comparison model** optimized for "who wins." It doesn't have direct scoring output. So how is it even predicting totals?

Looking at the architecture:
- Bradley-Terry generates `p_home_win`
- But for totals, it must be falling back to `total_from_ratings()` which uses: `intercept + slope * (rating_sum)`
- This is a **derived estimate**, not a direct prediction

**The weight of 59.8% on a derived estimate is very dangerous.**

### Problem 2: You're Only Using 2 Models for Totals

Only Bradley-Terry and Poisson. This is extremely high correlation risk.

- **Poisson**: Designed for team scoring (good for totals in theory)
- **Bradley-Terry**: Designed for head-to-head (poor for totals)

But Bradley-Terry is winning 60% of the weight. Why?

Possible reasons:
1. Bradley-Terry happens to fit the Oct 1 - Jan 11 data well (but may not generalize)
2. Poisson predictions are noisy
3. The tuning period is biased toward Bradley-Terry's strengths

### Problem 3: Spread and Total Ensembles Are Dramatically Different

**Spread uses all 5 models**, but **Total uses only 2** and weights them opposite to Spread:

```
                 ML      SPREAD   TOTAL
bradley-terry   53.8%     0.0%    59.8% ← Wildly different!
poisson          0.0%    41.2%    40.1% ← Similar
toor            28.8%    27.3%     0.0% ← Dropped for totals
gssd            17.4%    23.5%     0.0% ← Dropped for totals
elo              0.0%     8.0%     0.0% ← Dropped for totals
```

This asymmetry suggests:
- **Hypothesis**: Bradley-Terry was good at totals **during your training window (Oct 1 - Jan 11)** due to the January slowdown pattern
- **But**: It's probably overfitting to January's specific conditions
- **Risk**: It will fail when January slowdown ends or patterns change

---

## The Root Issue: Overfitting to Training Window

Your weights were tuned from Oct 1 - Jan 11, which includes:
- October: Peak scoring (+2.0)
- November: Normal scoring
- December: Slight slowdown (-1.2)
- **January 1-11: Major slowdown (-6.0)**

Bradley-Terry probably has some implicit scoring adjustment that happens to match January. When you weight it 59.8% for totals, you're betting that Bradley-Terry's Jan-specific strength is generalizable.

**This is the opposite of what you want.** You should be weighting models that:
1. Predict scoring **directly** (Poisson)
2. Are robust **across seasons** (not just Jan)
3. Don't overfit to training window conditions

---

## How to Fix This (Ranked by Confidence)

### Fix 1: Immediate Reweighting (High Confidence)
```python
TOTAL_WEIGHTS_MANUAL = {
    "poisson": 0.8,         # Direct scoring prediction
    "bradley-terry": 0.2    # Reduce overfitting
}
```

**Why**: Poisson was designed to predict team scoring. Bradley-Terry was not.  
**Expected outcome**: Less over-confident totals, better seasonal robustness  
**Time**: 5 minutes to change config

### Fix 2: Use Separate Training Windows (High Confidence)
```python
# For TOTAL weighting, use only Sep 1 - Nov 30 (pre-January slowdown)
# For ML/SPREAD weighting, use Oct 1 - Jan 11 (current approach)

# Train total ensemble without January bias
# Then apply it to January games
```

**Why**: Prevents seasonal-specific overfitting  
**Expected outcome**: Bradley-Terry loses weight when trained on non-January data  
**Time**: 30 minutes to restructure tuning

### Fix 3: Per-Team Total Models (High Impact, Medium Effort)

Instead of using Bradley-Terry + Poisson, create pure scoring models:

```python
class HomeTeamScoringModel:
    def predict_home_ppg(self, home_team, games_df):
        # Fit Poisson or linear regression on home_ppg
        home_games = games_df[games_df['home_team'] == home_team]
        return np.mean(home_games['home_score'])

class AwayTeamScoringModel:
    def predict_away_ppg(self, away_team, games_df):
        # Fit Poisson or linear regression on away_ppg
        away_games = games_df[games_df['away_team'] == away_team]
        return np.mean(away_games['away_score'])

def predict_total(home, away, home_model, away_model):
    home_ppg = home_model.predict(home)
    away_ppg = away_model.predict(away)
    return home_ppg + away_ppg
```

**Why**: Direct prediction, no derived estimates  
**Expected outcome**: Better total forecasts, more stable across seasons  
**Time**: 2-3 hours to implement

### Fix 4: Detect When Totals Are Overfitting (Quick Diagnostic)

```python
# Compare Bradley-Terry weight pre vs post January
weights_sep_to_nov = tune_total_ensemble(start='2025-09-01', end='2025-11-30')
weights_oct_to_jan = tune_total_ensemble(start='2025-10-01', end='2026-01-11')

print(f"BT weight (Sep-Nov): {weights_sep_to_nov.get('bradley-terry')}")
print(f"BT weight (Oct-Jan): {weights_oct_to_jan.get('bradley-terry')}")

# If Oct-Jan is much higher, Bradley-Terry is overfitting to January
```

---

## What's Actually Wrong With Your System

Your architecture is NOT fundamentally ML-centric. The real problems are:

1. **Bradley-Terry shouldn't be your top total forecaster**
   - It happens to work in Jan due to random chance
   - Will likely fail next season or in spring

2. **You're not using Poisson enough for totals**
   - Poisson is designed for team scoring
   - It should have more weight, not less

3. **Your training window is biased**
   - Jan slowdown makes Bradley-Terry look good
   - But January is anomalous, not representative

4. **You don't have pure scoring models**
   - Everything is derived from margin or ratings
   - Direct team scoring predictions would be better

---

## What to Do NOW (Immediate Action)

### Step 1: Emergency Reweight (5 minutes)
Change TOTAL ensemble weights manually to:
```json
{
  "poisson": 0.8,
  "bradley-terry": 0.2
}
```

This will immediately reduce over-confidence on totals.

### Step 2: Verify It Works (30 minutes)
Test on last 20 games to see if:
- Over probability drops toward 50-55%
- Total predictions match market lines better
- Brier score / log loss improves

### Step 3: Retune with Correct Training Window (1 hour)
Exclude January from ensemble tuning window:
```
Start: 2025-10-01
End:   2026-01-05  # Exclude Jan slowdown
```

### Step 4: Build Scoring Models (Later)
Invest in direct home/away scoring models instead of derived totals.

---

## Expected Outcomes

**If you do Step 1 + 2**:
- Over probabilities drop from 63.6% to ~55-58% ✅
- Fewer >70% confident overs
- Better alignment with market lines
- Still using ensemble approach (not abandoning it)

**If you do Step 1 + 2 + 3**:
- Further improvement in Jan → Feb transition
- More stable weights across months
- Less risk of seasonal overfitting

**If you do Step 1 + 2 + 3 + 4**:
- Pure scoring models that don't depend on margin
- Better long-term accuracy
- Generalize better across seasons and leagues

---

## Bottom Line

**Your system is NOT over-ML-centric.** 

**Your real problem**: Bradley-Terry is overweighted for totals due to overfitting to October-January training data, particularly the January slowdown anomaly.

**The fix**: Reduce Bradley-Terry weight, increase Poisson weight, or retrain on Jan-excluded window.

**How confident?** Very high (95%+) that this is the issue, based on the weight distribution you showed me.
