# Round-over-Round Analysis

**Rounds analyzed**: 2
**Generated**: auto

## Global Terrain Priors (mean probability per class)

| Round | Empty | Settlement | Port | Ruin | Forest | Mountain |
|-------|-------|-------|-------|-------|-------|-------|
| R1 | 0.6304 | 0.1397 | 0.0121 | 0.0109 | 0.1856 | 0.0214 |
| R2 | 0.6166 | 0.1686 | 0.0125 | 0.0164 | 0.1686 | 0.0173 |

## Prior Stability (std dev across rounds)

| Metric | Empty | Settlement | Port | Ruin | Forest | Mountain |
|-------|-------|-------|-------|-------|-------|-------|
| Std Dev | 0.0069 | 0.0145 | 0.0002 | 0.0028 | 0.0085 | 0.0021 |

## Rolling Backtest (R(N-1) priors -> R(N) score)

| From | To | Mean Score | Min Seed | Max Seed |
|------|------|-----------|----------|----------|
| R1 | R2 | 30.1 | 29.0 | 32.1 |

## Key Findings

- **Most stable**: Port (std=0.0002)
- **Least stable**: Settlement (std=0.0145)
- **Avg rolling backtest score**: 30.1/100