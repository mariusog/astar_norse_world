# Agent Plan: feature-agent

**Owner**: feature-agent (exclusively). Lead-agent creates tasks here; you fill out checklists and results.

## Active Tasks

(none -- all tasks completed)

## Escalations

Tasks that need lead-agent attention. Tag each as `BLOCKED` or `CRITICAL`.

| Tag | Task | Description |
|-----|------|-------------|
| - | - | - |

## Completed Tasks

### T20: Query budget optimizer
**Status**: done
**Branch**: `feature/T20-query-optimizer`
**Target**: Maximize map coverage and information gain from 50 queries across 5 seeds
**Depends on**: T10, T11

Create `src/query_strategy.py` (under 250 lines).

- [x] Define `QueryPlanner` class that plans viewport placements for a given seed
- [x] Constructor takes map dimensions (W, H), total budget (50), num_seeds (5)
- [x] `plan_initial_queries(seed_index: int, initial_grid: np.ndarray) -> list[dict]` -- generate first batch of query viewports
- [x] Strategy Phase 1 -- Coverage: use max 15x15 viewports to tile the map with minimal overlap. 10 queries per seed at 15x15 covers ~1400/1600 cells
- [x] Strategy Phase 2 -- Focus: after initial coverage, target high-uncertainty areas (near settlements, borders between factions)
- [x] `plan_adaptive_query(seed_index, observation_store, remaining_budget) -> dict | None` -- given current observations, pick next most informative viewport
- [x] Identify "interesting" cells: initial settlements, coastal areas, expansion zones (plains near settlements)
- [x] Budget allocation: at least 8 queries per seed for coverage, reserve 10 for adaptive follow-up across all seeds
- [x] Return viewport as `{"seed_index": int, "viewport_x": int, "viewport_y": int, "viewport_w": int, "viewport_h": int}`
- [x] Validate all viewports are within map bounds and viewport dimensions are 5-15
- [x] Self-review: lint + format check
- [x] Tests pass

**Acceptance criteria**: Full tiling plan covers >85% of map per seed with 8 queries. Adaptive queries target cells adjacent to settlements/ruins.

**Result**:
- **What changed**: Created `src/query_strategy.py` with QueryPlanner class implementing two-phase viewport planning (coverage tiling + adaptive interest-based targeting).
- **Metrics**: 8 queries per seed covers >85% of 40x40 map. Adaptive queries use interest scoring (settlement proximity, coastline, uncovered areas).
- **Tests**: 22 new tests in `tests/test_query_strategy.py`, all passing.

---

### T21: Prediction tensor generator
**Status**: done
**Branch**: `feature/T20-query-optimizer` (combined)
**Target**: Produce accurate W x H x 6 probability tensor combining local sim and server observations
**Depends on**: T11, T12

Create `src/predictor.py` (under 250 lines).

- [x] Define `Predictor` class
- [x] Constructor takes initial grid, settlements, observation store, and config (blend weights)
- [x] `predict(seed_index: int, num_mc_runs: int = 100) -> np.ndarray` -- produce H x W x 6 probability tensor
- [x] Step 1: Run Monte Carlo simulation locally (reuse `runner.run_monte_carlo`) to get prior probabilities
- [x] Step 2: Get observed probabilities from ObservationStore
- [x] Step 3: Blend -- for observed cells, weight observations heavily (0.8 obs / 0.2 sim); for unobserved cells, use sim output only
- [x] Step 4: Apply static terrain certainty -- mountains stay mountains (prob ~0.99), ocean stays ocean (prob ~0.99)
- [x] Step 5: Apply probability floor (0.01) and renormalize each cell to sum to 1.0
- [x] Blending weights configurable via constants (already in `src/constants.py`)
- [x] `OBSERVATION_WEIGHT` and `SIMULATION_WEIGHT` in constants
- [x] Handle the case where initial state comes from server (not local map gen)
- [x] Self-review: lint + format check
- [x] Tests pass

**Acceptance criteria**: Predictions for static cells (mountain, ocean) are near-certain. Observed cells heavily reflect observations. Unobserved cells use local sim. All cells sum to 1.0 with probability floor applied.

**Result**:
- **What changed**: Created `src/predictor.py` with Predictor class that blends MC sim output with server observations, applies static terrain certainty, and floors+normalizes all probabilities.
- **Metrics**: Mountain/ocean cells get 0.99 confidence. Observed cells weighted 0.8 obs / 0.2 sim. All cells sum to 1.0 with min prob > 0.
- **Tests**: 15 new tests in `tests/test_predictor.py`, all passing.

---

### T22: Submission pipeline
**Status**: done
**Branch**: `feature/T20-query-optimizer` (combined)
**Target**: End-to-end pipeline from round start to submission for all 5 seeds
**Depends on**: T10, T20, T21

Create `src/pipeline.py` (under 200 lines) and update entry point.

- [x] Define `CompetitionPipeline` class
- [x] Constructor takes `AstarClient` and config
- [x] `run(round_id: str | None = None) -> dict` -- full pipeline returning scores
- [x] Step 1: Get active round (or use provided round_id)
- [x] Step 2: Load initial states for all 5 seeds via state_loader
- [x] Step 3: For each seed, plan queries via QueryPlanner
- [x] Step 4: Execute queries via API client, store in ObservationStore
- [x] Step 5: For each seed, generate prediction via Predictor
- [x] Step 6: Submit predictions for all 5 seeds
- [x] Step 7: Log query usage, submission responses, and any self-scores
- [x] Add CLI entry point `python -m src.pipeline --token <JWT>` with argparse
- [x] Handle errors gracefully: if one seed fails, continue with others
- [x] Log seed, query count, and timing for reproducibility
- [x] Self-review: lint + format check
- [x] Tests pass

**Acceptance criteria**: Can run `python -m src.pipeline --token <JWT>` and get predictions submitted for all 5 seeds. Respects 50-query budget. Logs all actions.

**Result**:
- **What changed**: Created `src/pipeline.py` (CompetitionPipeline class) and `src/__main__.py` (CLI entry point). Pipeline orchestrates load->query->predict->submit with graceful error handling per seed.
- **Metrics**: Processes all seeds, respects 50-query budget, continues on per-seed failures.
- **Tests**: 9 new tests in `tests/test_pipeline.py`, all passing (using mocked API client).

---

### T23: Simulation calibration
**Status**: done
**Branch**: `feature/T20-query-optimizer` (combined)
**Target**: Reduce prediction error by detecting and correcting local sim biases
**Depends on**: T11, T12

Create `src/calibration.py` (under 200 lines).

- [x] `compute_divergence(observed_probs: np.ndarray, simulated_probs: np.ndarray, mask: np.ndarray) -> dict` -- compute per-cell and aggregate KL divergence between observed and simulated distributions for observed cells only
- [x] `detect_biases(divergence_map: np.ndarray) -> list[dict]` -- identify systematic patterns: e.g. "sim over-predicts settlements in interior", "sim under-predicts ruins near fjords"
- [x] `calibrate_weights(observed, simulated) -> float` -- compute optimal observation/simulation blend weight using observed cells as validation
- [x] `suggest_constant_adjustments(biases: list[dict]) -> list[str]` -- return human-readable suggestions for which constants to tune
- [x] Run calibration after initial queries, before final predictions
- [x] Report calibration results to logs/calibration_<timestamp>.csv
- [x] Self-review: lint + format check
- [x] Tests pass

**Acceptance criteria**: Can detect when local sim systematically diverges from server. Optimal blend weight computed from data. Calibration report generated.

**Result**:
- **What changed**: Created `src/calibration.py` with functions for KL divergence computation, per-class bias detection, optimal weight grid search, and CSV report generation.
- **Metrics**: Correctly identifies biases >0.05 threshold. Grid search over 21 weight values finds optimal blend. CSV report written to logs/.
- **Tests**: 18 new tests in `tests/test_calibration.py`, all passing.
