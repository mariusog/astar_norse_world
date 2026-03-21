# Competition Strategy

**Updated**: 2026-03-21 | **Rounds**: R1-R13 | **LOO best**: R3=92.1, R9=92.2, R13=88.5

## Architecture

```
Initial Grid → 5 Probe Queries (regime detection)
             → Safe-Default Regime (survive unless strong evidence)
             → Regime-Specific XGBoost Training
             → Conditional Ensemble + Power Transform (skipped for aggressive)
             → Equilibrium Shift (per-terrain aggregate from observations)
             → Regime Transforms (collapse_shift for deep_collapse)
             → 45 Observation Queries (settlement clusters + tiling)
             → Observation Persistence (auto-save/load .npz cache)
             → Floor + Normalize → Validation → Submit
```

## Regime Detection (Safe-Default)

| Condition | Regime | Rationale |
|-----------|--------|-----------|
| < 3 settlements checked | survive | Insufficient data, safest default |
| > 35% survival rate | aggressive | Strong expansion signal |
| < 5% survival AND >= 5 checked | deep_collapse | High confidence collapse |
| Everything else | survive | Covers survive + partial_collapse |

**Key lesson (R13)**: Misclassifying as deep_collapse cost 42.6 pts. Survive priors work well across survive/partial_collapse rounds (LOO 88.5 on R13).

## Per-Regime Parameters

| Regime | XGBoost Wt | Power | Transform | Training Rounds |
|--------|-----------|-------|-----------|-----------------|
| survive | 0.9 | 0.9 | spatial_smooth(0.3) | R1,R2,R4,R5,R9,R13 |
| deep_collapse | 0.7 | 1.0 | collapse_shift(0.3) | R3,R4,R8,R9,R10,R13 |
| aggressive | 1.0 | 1.0 | none | R6,R7,R11,R12 |

**Aggressive uses XGBoost-only** (no ensemble, no power). R12 LOO: 69.4 vs 63.1 with ensemble.

## What Works

| Component | Impact | Status |
|-----------|--------|--------|
| Safe-default regime detection | Prevents 15-43 pt disasters | Implemented |
| XGBoost-only for aggressive | +6.3 pts on R12 | Implemented |
| Observation persistence (.npz) | Survives re-runs | Implemented |
| Equilibrium shift from observations | +2-4 pts | Implemented |
| Pre-submission validation | Prevents 25-50 pt disasters | Implemented |

## Competitive Position

- **Our actual best**: R10: 75.4 x 1.629 = 122.8 weighted
- **LOO ceiling R13**: 88.5 x 1.890 = 167.3 weighted
- **Gap**: Regime detection was the #1 problem — now fixed with safe default

## Submission Checklist

```bash
# 1. Capture any new completed rounds
python -m scripts.post_round --token $TOKEN

# 2. Submit with v3 pipeline (auto-detects regime safely)
python -m scripts.submit_v3 --token $TOKEN

# 3. Override regime if confident:
python -m scripts.submit_v3 --token $TOKEN --regime aggressive

# 4. If validation fails, DO NOT use --force. Fix the bug first.
```

## Files

| File | Purpose |
|------|---------|
| `scripts/submit_v3.py` | Main submission pipeline (safe-default regime) |
| `scripts/loo_backtest_v3.py` | LOO backtest for offline validation |
| `src/ml_predictor.py` | XGBoost per-cell classifier |
| `src/prediction_validator.py` | Pre-submission sanity checks |
| `src/observation.py` | ObservationStore with disk persistence |
| `src/scoring.py` | KL divergence scoring (matches server) |
| `web/transforms.py` | Parametric post-processing transforms |
