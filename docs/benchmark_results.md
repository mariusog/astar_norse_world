# Prediction Quality Benchmarks

**Updated**: 2026-03-21 | **Rounds**: R1-R13 | **Method**: Leave-one-out backtesting

## IMPORTANT: All backtests use Leave-One-Out

Each round is scored using priors built only from OTHER rounds. No data leakage.

## Leave-One-Out Backtest (XGBoost-only, no ensemble/power)

| Round | Weight | LOO Score | Regime | Notes |
|-------|--------|-----------|--------|-------|
| R1 | 1.050 | 88.3 | survive | |
| R2 | 1.103 | 89.7 | survive | |
| R3 | 1.158 | 92.1 | deep_collapse | All settlements collapse |
| R4 | 1.216 | 91.5 | partial_collapse | Stochastic collapse |
| R5 | 1.276 | 83.7 | survive | |
| R6 | 1.340 | 62.1 | aggressive | XGBoost-only (no ensemble) |
| R7 | 1.407 | 77.5 | aggressive | XGBoost-only (no ensemble) |
| R8 | 1.478 | 89.4 | deep_collapse | All settlements collapse |
| R9 | 1.536 | 92.2 | partial_collapse | |
| R10 | 1.629 | 84.5 | deep_collapse | |
| R11 | 1.710 | 66.3 | aggressive | XGBoost-only |
| R12 | 1.800 | 69.4 | aggressive | XGBoost-only |
| R13 | 1.890 | 88.5 | partial_collapse | Survive priors work best |

## Four Simulation Regimes

| Regime | Rounds | Final Settlements | Key Feature |
|--------|--------|-------------------|-------------|
| survive | R1, R2, R5, R9 | 17-49 | Normal expansion |
| partial_collapse | R4, R9, R13 | 0-4 (stochastic) | Mostly collapse, some survive |
| deep_collapse | R3, R8, R10 | 0 | All settlements die |
| aggressive | R6, R7, R11, R12 | 89-192 | Massive settlement expansion |

## Key Finding: XGBoost-only beats ensemble for aggressive rounds

| Approach | R6 | R7 | R11 | R12 | Avg |
|----------|-----|-----|------|------|------|
| XGBoost + 0.3 flat priors + power 0.8 | 79.7 | 72.8 | 78.9 | 63.1 | 73.6 |
| XGBoost only (no ensemble/power) | 62.1 | 77.5 | 66.3 | 69.4 | 68.8 |
| XGBoost R7,R11 only → R12 LOO | - | - | - | 71.8 | - |

For the most settlement-dense rounds (R12), XGBoost-only wins. Ensemble+power helps R6 but hurts R12.

## Actual Submissions

| Round | Submitted | LOO Ceiling | Gap | Root Cause |
|-------|-----------|-------------|-----|------------|
| R2 | 28.9 | 89.7 | -60.8 | Laplace add-1 bug |
| R5 | 46.5 | 83.7 | -37.2 | Wrong regime detection |
| R6 | 56.5 | 74.7 | -18.2 | First submit burned queries |
| R7 | 54.7 | 64.8 | -10.1 | |
| R8 | 60.4 | 89.4 | -29.0 | |
| R9 | 77.0 | 92.2 | -15.2 | |
| R10 | 75.4 | 84.5 | -9.1 | |
| R13 | 45.9 | 88.5 | -42.6 | Misclassified as deep_collapse |

## Score Progression

Best weighted score per round:
- R10: 75.4 x 1.629 = 122.8
- R13: 45.9 x 1.890 = 86.8
- LOO ceiling R13: 88.5 x 1.890 = **167.3**
- LOO ceiling R10: 84.5 x 1.629 = **137.7**

## Critical Lesson: Regime Misdetection

R13 lost 42.6 points from misclassifying as deep_collapse. The probe-based regime
detection saw few surviving settlements (2/9 = 22% rate) and classified as collapse.
But the GT probability distribution still has ~9% settlement on plains -- partial_collapse
needs survive-like priors, not collapse priors.
