# Round-over-Round Analysis

**Rounds analyzed**: 5 (R1-R5) | **Updated**: 2026-03-20

## Two Simulation Regimes

| Regime | Rounds | Settlements | Avg Entropy | Characteristics |
|--------|--------|-------------|-------------|-----------------|
| **Survive** | R1, R2, R5 | 30-50 final | 0.49-0.69 | Settlements persist, high cell uncertainty |
| **Collapse** | R3, R4 | 0 final | 0.07-0.47 | All settlements collapse to empty/forest |

**Cannot predict regime from initial grid** — survive and collapse rounds have identical initial conditions (~40-50 settlements each).

## Per-Terrain Priors (survive-weighted, all 5 rounds)

| Terrain | Empty | Settlement | Port | Ruin | Forest | Mountain |
|---------|-------|------------|------|------|--------|----------|
| Ocean | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Plains | 0.804 | 0.136 | 0.011 | 0.013 | 0.036 | 0.000 |
| Settlement | 0.430 | 0.336 | 0.005 | 0.027 | 0.201 | 0.000 |
| Forest | 0.078 | 0.141 | 0.010 | 0.013 | 0.757 | 0.000 |
| Mountain | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |

## Error Breakdown (% of total weighted KL loss)

| Source | % of Loss | Notes |
|--------|-----------|-------|
| Plains cells | 61.6% | Largest source — high count, moderate entropy |
| Forest cells | 29.9% | Second largest — uncertain near settlements |
| Settlement cells | 8.2% | Small count but high entropy |
| Ocean/Mountain | 0.0% | Static — perfectly predicted |

## Backtest Scores (priors + distance, no observations)

| Round | Weight | Score | Regime |
|-------|--------|-------|--------|
| R1 | 1.050 | 79.8 | survive |
| R2 | 1.103 | 80.2 | survive |
| R3 | 1.158 | 49.5 | collapse |
| R4 | 1.216 | 89.8 | collapse |
| R5 | 1.276 | 80.4 | survive |
| **Avg** | | **75.9** | |
