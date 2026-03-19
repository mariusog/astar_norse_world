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
**Status**: open
**Branch**: `qa/T31-api-tests`
**Target**: Full test coverage for API client with mocked HTTP
**Depends on**: T10

- [ ] Create `tests/test_api_client.py`
- [ ] Test auth setup: both cookie and bearer token paths
- [ ] Test `list_rounds` -- mock response, verify parsing
- [ ] Test `get_round` -- mock response with initial_states, verify structure
- [ ] Test `query` -- verify request body format, viewport validation (reject w/h outside 5-15)
- [ ] Test `submit` -- verify probability floor applied, prediction shape validated
- [ ] Test query budget tracking: counter increments, warning at 45, error at 50
- [ ] Test retry on transient HTTP errors (500, 503)
- [ ] Test `AuthError` raised on 401/403
- [ ] Use `unittest.mock.patch` or `responses` library for HTTP mocking
- [ ] Self-review: lint + format check
- [ ] Tests pass

**Acceptance criteria**: All API client public methods have tests. Budget enforcement tested. Error paths tested.

**Result**:

---

### T32: End-to-end pipeline test
**Status**: open
**Branch**: `qa/T32-e2e-test`
**Target**: Verify full pipeline produces valid submissions
**Depends on**: T22

- [ ] Create `tests/test_pipeline.py`
- [ ] Mock the API client to return canned responses
- [ ] Test pipeline runs through all steps: load round -> plan queries -> execute -> predict -> submit
- [ ] Verify prediction shape is (H, W, 6) for each seed
- [ ] Verify all predictions sum to 1.0 per cell (within tolerance)
- [ ] Verify probability floor (no values below 0.01)
- [ ] Verify query budget not exceeded (count mock calls)
- [ ] Test graceful degradation: one seed fails, others succeed
- [ ] Self-review: lint + format check
- [ ] Tests pass

**Acceptance criteria**: Pipeline produces valid predictions for all seeds. Budget respected. Error handling works.

**Result**:

---

### T33: Prediction quality benchmarks
**Status**: open
**Branch**: `qa/T33-prediction-benchmarks`
**Target**: Establish baseline score and track improvements
**Depends on**: T21, T30

- [ ] Create `benchmarks/prediction_quality.py` (or `tests/benchmark_predictions.py` marked slow)
- [ ] Generate "ground truth" by running Monte Carlo with 500+ runs on known seeds
- [ ] Score our predictor (with fewer MC runs, e.g. 50-100) against this ground truth
- [ ] Test multiple strategies: (a) pure local sim, (b) sim + mock observations, (c) with calibration
- [ ] Report baseline scores to `docs/benchmark_results.md` in Tier 1 summary format (<40 lines)
- [ ] Include: seed, strategy, score, num_queries_used, runtime_seconds
- [ ] Run with at least 5 different seeds for statistical significance
- [ ] Self-review: lint + format check
- [ ] Tests pass

**Acceptance criteria**: Baseline score established. Benchmark is reproducible (seeded). Report in docs/benchmark_results.md.

**Result**:

---

## Escalations

Tasks that need lead-agent attention. Tag each as `BLOCKED` or `CRITICAL`.

| Tag | Task | Description |
|-----|------|-------------|
| - | - | - |

## Completed Tasks
