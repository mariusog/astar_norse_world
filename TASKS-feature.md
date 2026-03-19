# Agent Plan: feature-agent

**Owner**: feature-agent (exclusively). Lead-agent creates tasks here; you fill out checklists and results.

## Active Tasks

### T20: Query budget optimizer
**Status**: open
**Branch**: `feature/T20-query-optimizer`
**Target**: Maximize map coverage and information gain from 50 queries across 5 seeds
**Depends on**: T10, T11

Create `src/query_strategy.py` (under 250 lines).

- [ ] Define `QueryPlanner` class that plans viewport placements for a given seed
- [ ] Constructor takes map dimensions (W, H), total budget (50), num_seeds (5)
- [ ] `plan_initial_queries(seed_index: int, initial_grid: np.ndarray) -> list[dict]` -- generate first batch of query viewports
- [ ] Strategy Phase 1 -- Coverage: use max 15x15 viewports to tile the map with minimal overlap. 10 queries per seed at 15x15 covers ~1400/1600 cells
- [ ] Strategy Phase 2 -- Focus: after initial coverage, target high-uncertainty areas (near settlements, borders between factions)
- [ ] `plan_adaptive_query(seed_index, observation_store, remaining_budget) -> dict | None` -- given current observations, pick next most informative viewport
- [ ] Identify "interesting" cells: initial settlements, coastal areas, expansion zones (plains near settlements)
- [ ] Budget allocation: at least 8 queries per seed for coverage, reserve 10 for adaptive follow-up across all seeds
- [ ] Return viewport as `{"seed_index": int, "viewport_x": int, "viewport_y": int, "viewport_w": int, "viewport_h": int}`
- [ ] Validate all viewports are within map bounds and viewport dimensions are 5-15
- [ ] Self-review: lint + format check
- [ ] Tests pass

**Acceptance criteria**: Full tiling plan covers >85% of map per seed with 8 queries. Adaptive queries target cells adjacent to settlements/ruins.

**Result**:

---

### T21: Prediction tensor generator
**Status**: open
**Branch**: `feature/T21-prediction-generator`
**Target**: Produce accurate W×H×6 probability tensor combining local sim and server observations
**Depends on**: T11, T12

Create `src/predictor.py` (under 250 lines).

- [ ] Define `Predictor` class
- [ ] Constructor takes initial grid, settlements, observation store, and config (blend weights)
- [ ] `predict(seed_index: int, num_mc_runs: int = 100) -> np.ndarray` -- produce H x W x 6 probability tensor
- [ ] Step 1: Run Monte Carlo simulation locally (reuse `runner.run_monte_carlo`) to get prior probabilities
- [ ] Step 2: Get observed probabilities from ObservationStore
- [ ] Step 3: Blend -- for observed cells, weight observations heavily (e.g. 0.8 obs / 0.2 sim); for unobserved cells, use sim output only
- [ ] Step 4: Apply static terrain certainty -- mountains stay mountains (prob ~0.99), ocean stays ocean (prob ~0.99)
- [ ] Step 5: Apply probability floor (0.01) and renormalize each cell to sum to 1.0
- [ ] Blending weights configurable via constants (add to `src/constants.py`)
- [ ] `OBSERVATION_WEIGHT` and `SIMULATION_WEIGHT` in constants
- [ ] Handle the case where initial state comes from server (not local map gen)
- [ ] Self-review: lint + format check
- [ ] Tests pass

**Acceptance criteria**: Predictions for static cells (mountain, ocean) are near-certain. Observed cells heavily reflect observations. Unobserved cells use local sim. All cells sum to 1.0 with probability floor applied.

**Result**:

---

### T22: Submission pipeline
**Status**: open
**Branch**: `feature/T22-submission-pipeline`
**Target**: End-to-end pipeline from round start to submission for all 5 seeds
**Depends on**: T10, T20, T21

Create `src/pipeline.py` (under 200 lines) and update entry point.

- [ ] Define `CompetitionPipeline` class
- [ ] Constructor takes `AstarClient` and config
- [ ] `run(round_id: str | None = None) -> dict` -- full pipeline returning scores
- [ ] Step 1: Get active round (or use provided round_id)
- [ ] Step 2: Load initial states for all 5 seeds via state_loader
- [ ] Step 3: For each seed, plan queries via QueryPlanner
- [ ] Step 4: Execute queries via API client, store in ObservationStore
- [ ] Step 5: For each seed, generate prediction via Predictor
- [ ] Step 6: Submit predictions for all 5 seeds
- [ ] Step 7: Log query usage, submission responses, and any self-scores
- [ ] Add CLI entry point `python -m src.pipeline --token <JWT>` with argparse
- [ ] Handle errors gracefully: if one seed fails, continue with others
- [ ] Log seed, query count, and timing for reproducibility
- [ ] Self-review: lint + format check
- [ ] Tests pass

**Acceptance criteria**: Can run `python -m src.pipeline --token <JWT>` and get predictions submitted for all 5 seeds. Respects 50-query budget. Logs all actions.

**Result**:

---

### T23: Simulation calibration
**Status**: open
**Branch**: `feature/T23-sim-calibration`
**Target**: Reduce prediction error by detecting and correcting local sim biases
**Depends on**: T11, T12

Create `src/calibration.py` (under 200 lines).

- [ ] `compute_divergence(observed_probs: np.ndarray, simulated_probs: np.ndarray, mask: np.ndarray) -> dict` -- compute per-cell and aggregate KL divergence between observed and simulated distributions for observed cells only
- [ ] `detect_biases(divergence_map: np.ndarray) -> list[dict]` -- identify systematic patterns: e.g. "sim over-predicts settlements in interior", "sim under-predicts ruins near fjords"
- [ ] `calibrate_weights(observed, simulated) -> float` -- compute optimal observation/simulation blend weight using observed cells as validation
- [ ] `suggest_constant_adjustments(biases: list[dict]) -> list[str]` -- return human-readable suggestions for which constants to tune
- [ ] Run calibration after initial queries, before final predictions
- [ ] Report calibration results to logs/calibration_<timestamp>.csv
- [ ] Self-review: lint + format check
- [ ] Tests pass

**Acceptance criteria**: Can detect when local sim systematically diverges from server. Optimal blend weight computed from data. Calibration report generated.

**Result**:

---

## Escalations

Tasks that need lead-agent attention. Tag each as `BLOCKED` or `CRITICAL`.

| Tag | Task | Description |
|-----|------|-------------|
| - | - | - |

## Completed Tasks
