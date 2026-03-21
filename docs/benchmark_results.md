# Prediction Quality Benchmarks

**Updated**: 2026-03-21 | **Rounds**: R1-R10 | **Method**: Leave-one-out backtesting

## IMPORTANT: All backtests use Leave-One-Out

Each round is scored using priors built only from OTHER rounds. No data leakage.

## Leave-One-Out Backtest (4-regime XGBoost pipeline, no observations)

| Round | Weight | LOO Score | Regime | Notes |
|-------|--------|-----------|--------|-------|
| R1 | 1.050 | 88.3 | survive | |
| R2 | 1.103 | 89.7 | survive | |
| R3 | 1.158 | 92.1 | deep_collapse | All settlements collapse |
| R4 | 1.216 | 91.5 | partial_collapse | Stochastic collapse |
| R5 | 1.276 | 83.7 | survive | |
| R6 | 1.340 | 74.7 | aggressive | Massive expansion (diffuse GT) |
| R7 | 1.407 | 64.8 | aggressive | Massive expansion (concentrated GT) |
| R8 | 1.478 | 89.4 | deep_collapse | All settlements collapse |
| R9 | 1.536 | 92.2 | partial_collapse | |
| R10 | 1.629 | 84.5 | deep_collapse | |
| **Avg** | | **85.1** | | |

## Four Simulation Regimes

| Regime | Rounds | Final Settlements | Key Feature |
|--------|--------|-------------------|-------------|
| survive | R1, R2, R5 | 17-49 | Normal expansion |
| partial_collapse | R4, R9 | Stochastic | Some seeds collapse, some survive |
| deep_collapse | R3, R8, R10 | 0-2 | All settlements die |
| aggressive | R6, R7 | 89-119 | Massive settlement expansion |

## Per-Regime Pipeline Parameters

| Regime | XGBoost Weight | Power | Training Rounds | Transform |
|--------|---------------|-------|-----------------|-----------|
| survive | 0.9 | 0.9 | R1,R2,R4,R5,R9 | spatial_smooth(0.3) |
| partial_collapse | 0.9 | 1.05 | R1,R2,R4,R5,R9 | none |
| deep_collapse | 0.7 | 1.0 | R3,R4,R8,R9,R10 | collapse_shift(0.3) |
| aggressive | 0.4 | 0.8 | R6,R7 | none |

## Actual Submissions

| Round | Submitted | LOO Prior | Gap | Root Cause |
|-------|-----------|-----------|-----|------------|
| R2 | 28.9 | 89.7 | -60.8 | Laplace add-1 bug |
| R5 | 46.5 | 83.7 | -37.2 | Wrong regime detection |
| R6 | 56.5 | 74.7 | -18.2 | First submit burned queries |
| R7 | 54.7 | 64.8 | -10.1 | |
| R8 | 60.4 | 89.4 | -29.0 | |
| R9 | 77.0 | 92.2 | -15.2 | |
| R10 | 75.4 | 84.5 | -9.1 | |

## Score Progression

Best weighted score per round:
- R9: 77.0 × 1.536 = 118.3 (current best)
- R10: 75.4 × 1.629 = 122.8
- LOO ceiling R9: 92.2 × 1.536 = **141.6**
- LOO ceiling R10: 84.5 × 1.629 = **137.7**
