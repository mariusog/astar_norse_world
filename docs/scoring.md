# Astar Island - Scoring

## Overview

Predictions are scored using **entropy-weighted KL divergence** against ground truth.

## Ground Truth

For each seed, organizers pre-compute ground truth by running the simulation **hundreds of times** with hidden parameters, producing probability distributions per cell.

Example ground truth for a cell: `[0.0, 0.60, 0.25, 0.15, 0.0, 0.0]` (60% settlement, 25% port, 15% ruin)

## Score Formula

### Step 1: KL Divergence per cell

```
KL(p || q) = Sum( p_i * log(p_i / q_i) )
```

Where `p` = ground truth, `q` = your prediction. Lower = better.

### Step 2: Entropy weighting

Static cells (unchanging ocean, mountains) are **excluded**. Only dynamic cells contribute, weighted by entropy:

```
entropy(cell) = -Sum( p_i * log(p_i) )
```

Higher entropy cells (more uncertain outcomes) receive **greater weight**.

### Step 3: Final score

```
weighted_kl = Sum( entropy(cell) * KL(ground_truth[cell], prediction[cell]) ) / Sum( entropy(cell) )

score = max(0, min(100, 100 * exp(-3 * weighted_kl) ))
```

- **100** = perfect prediction
- **0** = terrible prediction
- Exponential decay means diminishing returns on incremental improvements

## Per-Round and Leaderboard

- **Per-round score**: Average of 5 seed scores. **Unsubmitted seeds score 0.**
- **Leaderboard**: Best round score across all attempts. Later rounds may carry higher weights.

## Critical Pitfall: Zero Probabilities

**NEVER assign probability 0.0 to any terrain class.**

If ground truth has non-zero probability for a class you marked as 0, KL divergence becomes **infinite**, destroying that cell's entire score.

### Mitigation

```python
min_prob = 0.01
prediction = np.maximum(prediction, min_prob)
prediction = prediction / prediction.sum(axis=2, keepdims=True)
```

## Performance Baseline

- **Uniform predictions** (1/6 each): ~1-5 points
- Strategic query usage + informed model: significantly higher
