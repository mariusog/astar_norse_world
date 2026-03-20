# Competition Strategy

**Updated**: 2026-03-20 | **Current best**: 89.8 (R4 backtest) | **Leaderboard target**: 85+ on high-weight round

## Architecture

```
Initial Grid → Terrain Priors (from historical data)
             → Distance Priors (terrain × distance to settlement)
             → 50 Observation Queries (overlapping 15×15 on settlement clusters)
             → Observation Blending (count-scaled weights)
             → Soft Regime Blending (survive/collapse confidence from observed settlements)
             → Validation (6 sanity checks)
             → Submit
```

## What Works

| Component | Impact | Status |
|-----------|--------|--------|
| Survive-weighted terrain priors | +70 pts over uniform | ✅ Implemented |
| Distance-conditioned priors | +1-8 pts (round-dependent) | ✅ Implemented |
| Observation blending (α=0.01, K=5) | +1-4 pts with 10q/seed | ✅ Implemented |
| Pre-submission validation | Prevents 25-50 pt disasters | ✅ Implemented |
| Soft regime blending | +8 pts on collapse rounds | ✅ Implemented |

## What Doesn't Help

| Approach | Finding |
|----------|---------|
| Viewport size (5×5 vs 15×15) | All score within 0.3 pts — coverage vs overlap tradeoff is balanced |
| Binary regime detection | 67% misclassification rate on survive rounds (settlements only 33% likely) |
| Local MC simulation | Our priors from 5 rounds outperform MC sim on the actual server |
| Fine-grained distance bins | Overfits to cross-round average, hurts survive rounds |

## Competitive Position

- **Leaderboard**: best single round × round weight
- **Top teams**: ~113 weighted → ~85 on best round
- **Our ceiling**: R4 backtest 89.8 × 1.22 = 109.5 (top 10 if submitted correctly)
- **Our actual submissions**: 28.9 (R2), 46.5 (R5) — all due to bugs, not algorithm

## Priorities

1. **Reliability** — never ship a bug again. The validator catches all past failure modes.
2. **Data collection** — each new round improves priors by ~1 pt. Capture immediately.
3. **Collapse round handling** — R3 scores 49.5. If we can predict collapse, use collapse priors.

## Submission Checklist

```bash
# 1. Capture any new completed rounds
python -m scripts.post_round --token $TOKEN

# 2. Submit
python -m scripts.submit_v2 --token $TOKEN

# 3. If validation fails, DO NOT use --force. Fix the bug first.
```

## Files

| File | Purpose |
|------|---------|
| `scripts/submit_v2.py` | Main submission pipeline |
| `src/unified_priors.py` | Terrain + distance prior builder |
| `src/soft_regime.py` | Post-observation regime blending |
| `src/prediction_validator.py` | Pre-submission sanity checks |
| `src/observation.py` | ObservationStore (α=0.01 smoothing) |
| `scripts/post_round.py` | Capture round data + rebuild priors |
| `scripts/backtest.py` | Backtest pipeline against historical rounds |
