# Over-Confidence Fix for BETS Total Probabilities

## Problem
"Over" bets in the BETS sheet were showing overly confident probabilities. Analysis revealed:
- **3 out of 7 overs** were >70% confident
- **Average over probability: 63.6%** vs unders: 36.4%
- Model predicted totals averaging **231.1** vs market lines **224.0** (difference: +7.1 points)

## Root Cause Analysis
Investigation identified two interconnected issues:

1. **Mid-Season Trend Not Captured**: 
   - Season average total scoring: 232.0 points
   - October-December average: 233.6 points
   - **January average: 227.5 points** (-6.1 point drop)
   - Models were using long-term averages (231.1) but January games were trending significantly lower
   - Markets had also conservatively lowered lines (224.0 average)

2. **Distribution Too Narrow**:
   - Default `TOTAL_SD_FALLBACK` of 20.0 was the standard deviation used when models didn't calculate their own
   - A smaller SD makes the normal CDF sharper, creating more extreme probabilities
   - Need wider distribution to account for uncertainty and recent trend shifts

## Solution Implemented

### Change 1: Increased Default Total SD (src/config.py)
```python
DEFAULT_TOTAL_SD_FALLBACK = 22.0  # Increased from 20.0
```
- Increased from 20.0 to 22.0 points to flatten the normal distribution
- Creates less extreme probability swings from small differences in predicted totals vs market lines
- Guardrails still enforce range [8.0, 35.0] so this is safe

### Change 2: Added Recency Adjustment Function (src/pipelines/schedule.py)
New function `_compute_total_recency_adjustment()`:
- Compares average total from recent 100 completed games to season-long average
- Calculates adjustment factor (e.g., -6.0 means "subtract 6 points from forecasts")
- Applied in both multi-model ensemble and single-model paths for totals
- Only applied if adjustment magnitude > 0.5 points (ignores noise)
- Gracefully falls back if database query fails

### Integration Points
Applied total_mean adjustment in two places in `_build_bets_dataframe()`:
1. **Multi-model ensemble path** (line ~2055): After ensemble combines forecasts
2. **Single-model passthrough path** (line ~2098): Before assigning to BETS

## Results

### Before Fix:
```
Total: 14 total bets
Very confident (>70%): 3
  Dallas Mavericks @ Milwaukee Bucks - Over 220.5: 72.1%
  Denver Nuggets @ Memphis Grizzlies - Over 221.5: 72.1%
  Brooklyn Nets @ Los Angeles Clippers - Over 211.5: 82.6%

Overs only:
  Count: 7
  Avg prob: 63.6%
  >70% confident: 3

Unders only:
  Count: 7
  Avg prob: 36.4%
  >70% confident: 0

Model mean diff vs line: +7.1 points
```

### After Fix:
```
Total: 14 total bets
Very confident (>70%): 1
  Brooklyn Nets @ Los Angeles Clippers - Over 211.5: 78.5%

Overs only:
  Count: 7
  Avg prob: 58.0%  ← Reduced from 63.6%
  >70% confident: 1  ← Reduced from 3

Unders only:
  Count: 7
  Avg prob: 42.0%  ← Increased from 36.4%
  >70% confident: 0

Model mean diff vs line: +4.2 points  ← Reduced from +7.1
```

## Improvements
✅ Reduced very confident overs from 3 to 1 (-67%)
✅ Reduced average over confidence from 63.6% to 58.0%
✅ Better balanced between overs (58%) and unders (42%)
✅ More realistic probability distributions
✅ Model predictions now closer to market consensus (+4.2 vs +7.1)
✅ **Zero test failures** - all 76 schedule/total tests pass

## Behavioral Notes
- **Recency adjustment is automatic**: Computed once per schedule run, applied to all total forecasts
- **Adaptive**: Will naturally adjust when recent games trend higher or lower
- **Safe defaults**: Only applies if adjustment >0.5 points; silent fallback if DB query fails
- **Logged**: Logs adjustment factor (e.g., "adjustment=-6.1") so it's visible in run output

## Follow-Up Considerations
1. **Calibration**: Consider running calibration on total probabilities to further improve in next iteration
2. **Per-Team Recency**: Could refine to apply per-team (home/away) recency rather than season-wide
3. **Seasonal Patterns**: January slowdown appears consistent; could bake in as seasonal factor if desired
4. **Market Line Analysis**: If market lines also use recency, our alignment improves naturally

## Testing
- All existing tests pass (76 schedule/total related tests)
- Manual verification on 14 upcoming games shows realistic probability ranges
- No regressions in any pipeline stage
