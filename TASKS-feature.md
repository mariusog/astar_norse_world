# Agent Plan: feature-agent

**Owner**: feature-agent (exclusively). Lead-agent creates tasks here; you fill out checklists and results.

## Active Tasks

### T50: Overlap-focused query strategy
**Status**: open
**Branch**: `feature/T50-overlap-queries`
**Target**: 3-5 observations per high-entropy cell instead of 1 observation per cell
**Depends on**: T42

Replace the coverage-tiling strategy with one optimized for probability estimation.

- [ ] Create `src/query_strategy_v2.py` (under 250 lines) — keep old strategy as fallback
- [ ] `OverlapQueryPlanner` class, constructor takes: grid, settlement positions, budget, num_seeds
- [ ] Core idea: classify cells by expected entropy. Static cells (ocean, mountain, deep forest) need 0 queries. Dynamic cells near settlements need many.
- [ ] `plan_queries(seed_index, grid) -> list[Viewport]` — plan all queries for one seed upfront
- [ ] Step 1: Identify "dynamic zone" — cells within `INTEREST_SETTLEMENT_RADIUS` of any settlement + all settlement/port cells
- [ ] Step 2: Place viewports to maximize overlap within the dynamic zone. Use smallest viewport (5×5 or 7×7) centered on settlement clusters
- [ ] Step 3: Budget allocation — give more queries to seeds with more settlements / higher expected entropy
- [ ] Each dynamic cell should be observed 3-5 times; static cells 0 times
- [ ] Validate all viewports within bounds and dimensions 5-15
- [ ] Self-review: lint + format check
- [ ] Tests pass

**Acceptance criteria**: Backtested against R1+R2 ground truth, overlap strategy scores higher than tiling strategy. Dynamic cells have ≥3 observations on average.

**Result**:

---

### T51: Prior-based predictor
**Status**: open
**Branch**: `feature/T51-prior-predictor`
**Target**: Baseline score ≥85 using only historical priors, no MC simulation
**Depends on**: T41, T42

Update `src/predictor.py` (or create `src/predictor_v2.py` if cleaner).

- [ ] `PriorPredictor` class — uses pre-built terrain priors + settlement proximity features instead of MC simulation
- [ ] Constructor takes: grid, settlements, priors (from T41), features (from T42), observation_store
- [ ] `predict(seed_index) -> np.ndarray` — build H×W×6 tensor:
  - Base: terrain priors from historical data
  - Refinement 1: adjust based on settlement distance (cells closer to settlements → higher settlement probability)
  - Refinement 2: blend in observations with count-scaled weights (existing fix)
  - Refinement 3: static terrain override (ocean=1.0 empty, mountain=1.0 mountain)
  - Apply probability floor + renormalize
- [ ] No MC simulation needed — much faster (ms instead of seconds)
- [ ] Configurable via constants: distance decay factor, observation confidence K
- [ ] Type annotations, lint clean
- [ ] Tests: verify output shape, normalization, static terrain handling

**Acceptance criteria**: Scores ≥85 on R2 backtest with priors-only. Scores ≥88 with observations from overlap strategy.

**Result**:

---

### T52: Improved submission pipeline
**Status**: open
**Branch**: `feature/T52-pipeline-v2`
**Target**: Complete round-submission workflow with new strategy
**Depends on**: T50, T51

Update `src/pipeline.py` to use new components.

- [ ] Load historical priors from `data/priors.npy` at startup
- [ ] Use `OverlapQueryPlanner` from T50 instead of coverage tiling
- [ ] Use `PriorPredictor` from T51 instead of MC-based predictor
- [ ] Self-score each seed before submitting (using `score_prediction` with priors as pseudo-GT)
- [ ] Log per-seed: score estimate, query count, coverage %, submission status
- [ ] Respect 50-query budget with smart per-seed allocation
- [ ] CLI still works: `python -m src.pipeline --token <JWT>`
- [ ] Tests pass

**Acceptance criteria**: Full pipeline runs in <30 seconds (no MC needed). Backtested score ≥85 on R2.

**Result**:

---

### T53: Per-cell position-aware priors
**Status**: open
**Branch**: `feature/T53-position-priors`
**Target**: Learn position-relative patterns around settlements for +2-5 score improvement
**Depends on**: T40, T41

Create `src/position_priors.py` (under 250 lines).

- [ ] For each historical round: for each settlement in the initial state, collect a "neighborhood profile" — GT distributions at distances 1, 2, 3, 4, 5 from that settlement
- [ ] Aggregate across all rounds to build a "settlement influence model": probability of each class as a function of distance from nearest settlement
- [ ] `predict_from_position(grid, settlements, priors) -> np.ndarray` — start with terrain priors, then for cells near settlements, blend in the distance-based model
- [ ] Compare with flat terrain priors: does position-awareness improve score?
- [ ] Type annotations, lint clean
- [ ] Tests: verify distance model produces valid distributions

**Acceptance criteria**: Position-aware priors score at least 2 points higher than flat terrain priors on R2 backtest.

**Result**:

---

---

### T80: Clean submission script v2
**Status**: open
**Branch**: `feature/T80-submit-v2`
**Target**: Reliable 80+ score on backtest, single clean script
**Depends on**: T70, T72

Create `scripts/submit_v2.py` (under 250 lines). Replaces `submit_round.py` and `resubmit_round.py`.

**Key lessons from R2 (28.9) and R5 (46.5)**:
- R2: Laplace smoothing destroyed observations → fixed
- R5: Regime detection misclassified survive as collapse → drop regime detection entirely

- [ ] Load unified priors from T70 (survive-weighted, no regime detection)
- [ ] Plan all 50 queries using T72 query planner (zero probes, all observations)
- [ ] Execute queries, record observations in ObservationStore
- [ ] Blend observations with count-scaled weights (LAPLACE_ALPHA=0.01, OBS_CONFIDENCE_K=5)
- [ ] Apply static terrain overrides (ocean, mountain)
- [ ] Floor + renormalize
- [ ] Submit all 5 seeds
- [ ] CLI: `python -m scripts.submit_v2 --token <JWT> [--budget N] [--dry-run]`
- [ ] Log per-seed: queries used, coverage %, estimated quality
- [ ] Self-review: lint + format + tests pass

**Acceptance criteria**: Backtested avg ≥80 across all 5 historical rounds. R3 (hardest) ≥55.

**Result**:

---

### T81: Soft regime blending from observations
**Status**: open
**Branch**: `feature/T81-soft-regime`
**Target**: +3-5 pts over fixed survive priors by adapting to observed data
**Depends on**: T80

Add to the submission pipeline (under 100 lines of new code).

**Insight**: After observing cells, we can estimate the regime from the data — but softly, not binary.

- [ ] After all observations collected: count how many observed settlement cells still show as settlement
- [ ] Compute `survive_confidence = settlement_survival_rate`
- [ ] Soft-blend: `pred = confidence * survive_pred + (1-confidence) * collapse_pred`
- [ ] This is done AFTER observations, not before — so observations are used for both blending AND regime estimation
- [ ] Only apply soft blend to unobserved cells (observed cells already have direct data)
- [ ] Backtest: should help on R3/R4 without hurting R1/R2/R5

**Acceptance criteria**: Avg score across 5 rounds improves by ≥2 pts over fixed survive priors + observations.

**Result**:

---

### T82: Per-terrain observation weighting
**Status**: open
**Branch**: `feature/T82-terrain-obs-weight`
**Target**: Smarter observation blending per terrain type
**Depends on**: T71, T72

- [ ] Different terrain types have different prior confidence. Settlement cells (high entropy, ~40% settlement) need more observation weight than forest cells (low entropy, ~70% forest).
- [ ] Compute per-terrain-type observation weight: `w = base_weight * terrain_entropy / max_entropy`
- [ ] Cells with high prior entropy (settlements, ports) get more observation weight
- [ ] Cells with low prior entropy (forest, plains far from settlements) get less observation weight
- [ ] Backtest improvement

**Acceptance criteria**: +1 pt over uniform observation weighting.

**Result**:

---

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
