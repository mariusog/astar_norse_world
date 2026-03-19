# Astar Island - Game Overview

## What Is It

Astar Island is an AI challenge from NM i AI 2026. Competitors observe a **black-box Norse civilisation simulator** through a limited viewport and **predict the final world state**.

The simulator runs a procedurally generated Norse world for **50 years** where settlements grow, factions clash, trade routes form, alliances shift, forests reclaim ruins, and harsh winters reshape entire civilisations.

## Core Loop

1. **Observe**: Call `POST /astar-island/simulate` with viewport coordinates to see one stochastic run through a max **15x15 cell window**
2. **Learn**: Discover the world's hidden rules from partial observations
3. **Predict**: Submit a **W x H x 6 probability tensor** for each of 5 seeds — probabilities for 6 terrain classes per cell

## Key Constraints

- **50 queries total per round**, shared across all 5 seeds
- The simulation is **stochastic** — same map and parameters produce different outcomes every run
- Viewport is limited to **5-15 cells** per dimension
- Map size is **40x40** grid
- **5 seeds** per round, each producing a different simulation outcome

## Terrain Classes (Prediction Target)

| Index | Class      | Description |
|-------|------------|-------------|
| 0     | Empty      | Ocean, plains, generic empty cells |
| 1     | Settlement | Active Norse settlements |
| 2     | Port       | Coastal settlements with harbors |
| 3     | Ruin       | Collapsed settlements |
| 4     | Forest     | Provides food to adjacent settlements; mostly static but reclaims ruined land |
| 5     | Mountain   | Impassable, never changes |

## What Makes This Hard

- **Stochastic outcomes**: Must predict probability distributions, not deterministic states
- **Limited budget**: 50 queries across 5 seeds means ~10 queries per seed on a 40x40 map
- **Partial observability**: Each query reveals at most 15x15 = 225 cells out of 1600 total
- **Hidden mechanics**: Must infer simulation rules from observations
