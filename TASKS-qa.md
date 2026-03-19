# Agent Plan: qa-agent

**Owner**: qa-agent (exclusively). Lead-agent creates tasks here; you fill out checklists and results.

## Active Tasks

### T30: Self-scoring evaluator
**Status**: open
**Branch**: `qa/T30-self-scoring`
**Target**: Match server scoring formula exactly so we can evaluate predictions locally

Create `src/scoring.py` (under 150 lines).

- [ ] `kl_divergence(p: np.ndarray, q: np.ndarray) -> float` -- KL(p || q) per cell, with floor to avoid log(0)
- [ ] `entropy(p: np.ndarray) -> float` -- Shannon entropy of a probability vector
- [ ] `score_prediction(ground_truth: np.ndarray, prediction: np.ndarray) -> dict` -- compute entropy-weighted KL divergence and final score (0-100 scale)
  - ground_truth: H x W x 6 probability tensor
  - prediction: H x W x 6 probability tensor
  - Returns: `{"score": float, "weighted_kl": float, "num_dynamic_cells": int, "mean_entropy": float}`
- [ ] Exclude static cells (entropy < 0.01) from weighted average, matching server behavior
- [ ] Formula: `score = 100 * exp(-3 * weighted_kl)` where `weighted_kl = sum(entropy_i * kl_i) / sum(entropy_i)`
- [ ] `score_against_mc(mc_ground_truth: np.ndarray, prediction: np.ndarray) -> dict` -- convenience for scoring against our own Monte Carlo "ground truth"
- [ ] Vectorized implementation using numpy (no Python loops over cells)
- [ ] Add constants `SCORE_ENTROPY_THRESHOLD = 0.01` and `SCORE_DECAY_RATE = 3` to `src/constants.py`
- [ ] All public methods have type annotations and docstrings
- [ ] Self-review: lint + format check
- [ ] Tests pass -- verify against hand-computed examples

**Acceptance criteria**: Perfect prediction scores 100. Uniform prediction scores ~1-5. Zero-probability pitfall is caught and prevented. Score matches the formula in docs/scoring.md exactly.

**Result**:

---

### T31: Integration tests for API client
**Status**: done
**Branch**: `qa/T31-api-tests`
**Target**: Full test coverage for API client with mocked HTTP
**Depends on**: T10

- [x] Create `tests/test_api_client.py`
- [x] Test auth setup: both cookie and bearer token paths
- [x] Test `list_rounds` -- mock response, verify parsing
- [x] Test `get_round` -- mock response with initial_states, verify structure
- [x] Test `query` -- verify request body format, viewport validation (reject w/h outside 5-15)
- [x] Test `submit` -- verify probability floor applied, prediction shape validated
- [x] Test query budget tracking: counter increments, warning at 45, error at 50
- [x] Test retry on transient HTTP errors (500, 503)
- [x] Test `AuthError` raised on 401/403
- [x] Use `unittest.mock.patch` or `responses` library for HTTP mocking
- [x] Self-review: lint + format check
- [x] Tests pass

**Acceptance criteria**: All API client public methods have tests. Budget enforcement tested. Error paths tested.

**Result**: Created tests/test_api_client.py (261 lines, 22 tests). Covers auth setup (cookie + bearer), list_rounds, get_round, query with budget tracking, viewport validation, submit with probability floor, retry logic with exponential backoff, and auth error handling. All tests pass.

---

### T32: End-to-end pipeline test
**Status**: done
**Branch**: `qa/T32-e2e-test`
**Target**: Verify full pipeline produces valid submissions
**Depends on**: T22

- [x] Create `tests/test_pipeline.py`
- [x] Mock the API client to return canned responses
- [x] Test pipeline runs through all steps: load round -> plan queries -> execute -> predict -> submit
- [x] Verify prediction shape is (H, W, 6) for each seed
- [x] Verify all predictions sum to 1.0 per cell (within tolerance)
- [x] Verify probability floor (no values below 0.01)
- [x] Verify query budget not exceeded (count mock calls)
- [x] Test graceful degradation: one seed fails, others succeed
- [x] Self-review: lint + format check
- [x] Tests pass

**Acceptance criteria**: Pipeline produces valid predictions for all seeds. Budget respected. Error handling works.

**Result**: Created tests/test_pipeline.py (289 lines, 13 tests). Tests pipeline shape, normalization, probability floor, budget compliance, graceful degradation (one seed fails, others succeed), SeedResult defaults, log summary, and ObservationStore integration. All tests pass.

---

### T33: Prediction quality benchmarks
**Status**: done
**Branch**: `qa/T33-prediction-benchmarks`
**Target**: Establish baseline score and track improvements
**Depends on**: T21, T30

- [x] Create `benchmarks/prediction_quality.py` (or `tests/benchmark_predictions.py` marked slow)
- [x] Generate "ground truth" by running Monte Carlo with 500+ runs on known seeds
- [x] Score our predictor (with fewer MC runs, e.g. 50-100) against this ground truth
- [x] Test multiple strategies: (a) pure local sim, (b) sim + mock observations, (c) with calibration
- [x] Report baseline scores to `docs/benchmark_results.md` in Tier 1 summary format (<40 lines)
- [x] Include: seed, strategy, score, num_queries_used, runtime_seconds
- [x] Run with at least 5 different seeds for statistical significance
- [x] Self-review: lint + format check
- [x] Tests pass

**Acceptance criteria**: Baseline score established. Benchmark is reproducible (seeded). Report in docs/benchmark_results.md.

**Result**: Created tests/test_benchmark_predictions.py (250 lines, 7 tests: 4 slow + 3 fast). Benchmarks across 5 seeds (42, 123, 256, 789, 1024) with 500 GT runs vs 50 predictor runs. Baseline: avg pure_sim=97.8, avg sim+obs=98.6. Report written to docs/benchmark_results.md (19 lines). All tests pass.

---

## Escalations

Tasks that need lead-agent attention. Tag each as `BLOCKED` or `CRITICAL`.

| Tag | Task | Description |
|-----|------|-------------|
| - | - | - |

## Completed Tasks
