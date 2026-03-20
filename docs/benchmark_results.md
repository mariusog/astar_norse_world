# Prediction Quality Benchmarks

**Updated**: 2026-03-20 | **Rounds**: R1-R5 | **Method**: Historical prior backtesting

## Strategy Comparison (all 5 rounds, priors only)

| Strategy | R1 | R2 | R3 | R4 | R5 | Avg |
|----------|-----|-----|-----|-----|-----|-----|
| Uniform (1/6) | 5.9 | 6.7 | 2.9 | 4.3 | 6.7 | 5.3 |
| Flat terrain priors | 76.1 | 80.3 | 54.8 | 90.0 | 72.9 | 74.8 |
| + Distance priors | 79.8 | 80.2 | 49.5 | 89.8 | 80.4 | 75.9 |
| + 10 obs/seed (simulated) | 81.0 | 79.7 | 54.3 | 89.0 | 81.0 | 77.0 |

## Observation Density Impact (on top of distance priors)

| Obs/cell | Avg Score | Min Score | Notes |
|----------|-----------|-----------|-------|
| 0 | 75.9 | 48.4 | Priors only |
| 1 | 76.6 | 53.1 | +0.7 |
| 3 | 78.0 | 60.0 | +2.1 |
| 5 | 80.0 | 65.8 | +4.1 |
| 10 | 83.6 | 73.5 | +7.7 |

## Actual Submission History

| Round | Score | Rank | Bug |
|-------|-------|------|-----|
| R2 | 28.9 | 116/153 | Laplace add-1 smoothing destroyed observations |
| R5 | 46.5 | 99/144 | Regime detection misclassified survive as collapse |
| R6 | TBD | TBD | First submit had uniform priors (wrong data path), resubmitted with fix |

## Key Findings

- **Priors are the main driver** — terrain type + distance gives 76 avg, observations add ~1-4 pts
- **Query strategy barely matters** — viewport size, overlap, targeting all score within 0.3 pts
- **Reliability is the #1 issue** — every submission had a pipeline bug that cost 25-50 pts
- **Best potential score**: R4 = 89.8 × 1.22 weight = 109.5 (would be top 10)
