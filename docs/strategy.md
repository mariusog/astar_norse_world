# Competition Strategy

**Updated**: 2026-03-21 | **LOO avg**: 85.1 | **LOO best**: R3=92.1, R9=92.2 | **Leaderboard target**: 90+ on high-weight round

## Architecture

```
Initial Grid → 5 Probe Queries (regime detection)
             → 4-Regime Classification (survive/partial_collapse/deep_collapse/aggressive)
             → Regime-Specific XGBoost Training (include only matching rounds)
             → Ensemble with Regime-Matched Flat Priors
             → Power Transform (regime-specific exponent)
             → Equilibrium Shift (per-terrain aggregate from observations)
             → Regime Transforms (collapse_shift for deep_collapse)
             → 45 Observation Queries (settlement clusters + adaptive)
             → Observation Blending (adaptive K, count-scaled weights)
             → Floor + Normalize
             → Validation (6 sanity checks + prior consistency + backtest)
             → Submit
```

## What Works

| Component | Impact | Status |
|-----------|--------|--------|
| 4-regime XGBoost (regime-specific training) | +12 pts over single model | Implemented |
| Regime-specific power transforms | +2 pts avg (up to +12 on R7) | Implemented |
| Regime-matched flat priors | +3 pts on collapse rounds | Implemented |
| Equilibrium shift from observations | +2-4 pts | Implemented |
| Observation blending (adaptive K) | +1-4 pts with 10q/seed | Implemented |
| Pre-submission validation | Prevents 25-50 pt disasters | Implemented |
| collapse_shift transform | +15 pts on R3 | Implemented |

## What Doesn't Help

| Approach | Finding |
|----------|---------|
| Viewport size variation | All score within 0.3 pts |
| spatial_smooth sigma=0.3 | Zero measurable impact |
| Adding survive data to aggressive training | Hurts R7 by ~4 pts |
| Binary regime detection | 67% misclassification rate |

## Per-Regime Parameters

| Regime | XGBoost Weight | Power | Transform | Training Rounds |
|--------|---------------|-------|-----------|-----------------|
| survive | 0.9 | 0.9 | spatial_smooth(0.3) | R1,R2,R4,R5,R9 |
| partial_collapse | 0.9 | 1.05 | none | R1,R2,R4,R5,R9 |
| deep_collapse | 0.7 | 1.0 | collapse_shift(0.3) | R3,R4,R8,R9,R10 |
| aggressive | 0.4 | 0.8 | none | R6,R7 |

## Competitive Position

- **Leaderboard**: best single round score x round weight
- **Our LOO ceiling**: R9: 92.2 x 1.536 = 141.6 weighted
- **Our actual best**: R10: 75.4 x 1.629 = 122.8 weighted
- **Gap**: ~19 pts between actual and ceiling — closing this requires correct regime detection + observation strategy

## Priorities

1. **Regime detection accuracy** — wrong regime costs 15-30 pts
2. **Observation strategy** — equilibrium shift and blending add 3-8 pts
3. **Data collection** — each new round improves LOO by ~0.5 pts avg
4. **Aggressive round research** — R6/R7 still weakest (74.7/64.8)

## Submission Checklist

```bash
# 1. Capture any new completed rounds
python -m scripts.post_round --token $TOKEN

# 2. Submit with v3 pipeline
python -m scripts.submit_v3 --token $TOKEN

# 3. If validation fails, DO NOT use --force. Fix the bug first.
```

## Files

| File | Purpose |
|------|---------|
| `scripts/submit_v3.py` | Main submission pipeline (4-regime) |
| `scripts/loo_backtest_v3.py` | LOO backtest for offline validation |
| `src/ml_predictor.py` | XGBoost per-cell classifier |
| `src/prediction_validator.py` | Pre-submission sanity checks |
| `src/observation.py` | ObservationStore (alpha=0.01 smoothing) |
| `src/scoring.py` | KL divergence scoring (matches server) |
| `web/transforms.py` | Parametric post-processing transforms |
| `web/backtest.py` | Web-based backtest engine |
