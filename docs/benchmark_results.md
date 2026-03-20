# Prediction Quality Benchmarks

**Updated**: 2026-03-20 | **Rounds**: R1-R6 | **Method**: Leave-one-out backtesting

## IMPORTANT: All backtests use Leave-One-Out

Previous benchmarks were **inflated by ~3-7 pts** due to training-set leakage
(scoring a round using priors that included that round's data). All numbers
below use strict LOO: each round is scored using priors built only from OTHER rounds.

## Leave-One-Out Backtest (priors only, no observations)

| Round | Weight | LOO Score | Regime | Notes |
|-------|--------|-----------|--------|-------|
| R1 | 1.050 | 76.2 | survive | |
| R2 | 1.103 | 78.7 | survive | |
| R3 | 1.158 | 50.5 | collapse | All settlements collapse |
| R4 | 1.216 | 89.1 | collapse | High score despite collapse |
| R5 | 1.276 | 74.6 | survive | |
| R6 | 1.340 | 58.3 | survive (aggressive) | 89-119 final settlements |
| **Avg** | | **71.2** | | |

## Three Simulation Regimes Discovered

| Regime | Rounds | Final Settlements | Plains→Settlement | Key Feature |
|--------|--------|-------------------|-------------------|-------------|
| Survive | R1, R2, R5 | 17-49 | ~12-19% | Normal expansion |
| Collapse | R3, R4 | 0-2 | ~0-9% | All settlements die |
| Aggressive | R6 | 89-119 | **24.6%** | Massive expansion |

## Actual Submissions vs LOO Ceiling

| Round | Submitted | LOO Ceiling | Gap | Root Cause |
|-------|-----------|-------------|-----|------------|
| R2 | 28.9 | 78.7 | -49.8 | Laplace add-1 bug |
| R5 | 46.5 | 74.6 | -28.1 | Wrong regime detection |
| R6 | 56.5 | 58.3 | **-1.8** | First submit burned queries, resubmit was close |

R6 was actually our best submission relative to the LOO ceiling.

## Observation Impact (simulated, on top of LOO priors)

| Obs/cell | Avg LOO Score | Improvement |
|----------|--------------|-------------|
| 0 | 71.2 | baseline |
| 3 | ~73 | +2 |
| 5 | ~75 | +4 |
| 10 | ~79 | +8 |
