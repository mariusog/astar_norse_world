# Tasks

**Owner**: lead-agent (exclusively). Other agents read this file but do not write to it.

Each agent has a separate plan file with detailed checklists:
- [TASKS-core.md](TASKS-core.md) -- core-agent tasks
- [TASKS-feature.md](TASKS-feature.md) -- feature-agent tasks
- [TASKS-qa.md](TASKS-qa.md) -- qa-agent tasks

## Format

Each task has: ID, status, agent, title, details, and optional dependencies.

**Statuses**: `open`, `in-progress`, `done`, `blocked`, `deferred`

## Open Tasks

| ID | Agent | Title | Details | Depends on |
|----|-------|-------|---------|------------|
| - | - | - | - | - |

## In Progress

| ID | Agent | Title | Status | Notes |
|----|-------|-------|--------|-------|
| - | - | - | - | - |

## Done

| ID | Agent | Title | Result |
|----|-------|-------|--------|
| T1 | qa-agent | Set up test infrastructure | Completed -- conftest.py with shared fixtures, 40 tests passing |
| T2 | core-agent | Implement core simulation | Completed -- terrain, settlement, map_generator, simulation modules |
| T3 | feature-agent | Implement Monte Carlo runner | Completed -- runner.py with single/MC runs and ASCII renderer |
| T4 | qa-agent | Write unit tests | Completed -- tests for terrain, settlement, map_generator, simulation |
| T10 | core-agent | API client for competition server | Completed -- AstarClient with auth, 4 endpoints, budget tracking, retry, typed exceptions (238 lines, 19 tests) |
| T11 | core-agent | Initial state loader from server response | Completed -- load_initial_state/load_round parsing server JSON to InternalTerrain + Settlement (148 lines, 16 tests) |
| T12 | core-agent | Observation aggregator | Completed -- ObservationStore with frequency counting, Laplace smoothing, overlap handling (163 lines, 18 tests) |
| T20 | feature-agent | Query budget optimizer | Completed -- QueryPlanner with coverage tiling + adaptive queries, >85% coverage in 8 queries (254 lines, 22 tests) |
| T21 | feature-agent | Prediction tensor generator | Completed -- Predictor blending MC sim + observations, static terrain certainty, probability floor (229 lines, 15 tests) |
| T22 | feature-agent | Submission pipeline | Completed -- CompetitionPipeline with CLI entry point, graceful error handling (291 lines + 74 line __main__, 9 tests) |
| T23 | feature-agent | Simulation calibration | Completed -- KL divergence, bias detection, grid-search weight calibration, CSV reports (244 lines, 18 tests) |
| T30 | qa-agent | Self-scoring evaluator | Completed -- vectorized entropy-weighted KL scoring matching server formula (138 lines, 20 tests) |
| T31 | qa-agent | Integration tests for API client | Completed -- 22 tests covering auth, endpoints, viewport validation, budget tracking, retry, error handling |
| T32 | qa-agent | End-to-end pipeline test | Completed -- 13 tests covering full pipeline, normalization, probability floor, budget, graceful degradation |
| T33 | qa-agent | Prediction quality benchmarks | Completed -- 7 tests (4 slow), baseline scores: avg pure_sim=97.8, sim+obs=98.6 across 5 seeds |
