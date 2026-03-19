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
| T10 | core-agent | API client for competition server | HTTP client with auth, rate limiting, and all 4 endpoints (list rounds, get round, simulate/query, submit) | - |
| T11 | core-agent | Initial state loader from server response | Parse server initial_states into our InternalTerrain grid + Settlement objects, replacing local map_generator for competition runs | T10 |
| T12 | core-agent | Observation aggregator | Merge multiple stochastic viewport observations into per-cell probability estimates using frequency counting + Bayesian smoothing | T10 |
| T20 | feature-agent | Query budget optimizer | Decide which viewport rectangles to query for each seed to maximize information gain; 50 queries / 5 seeds, viewport 5-15 cells | T10, T11 |
| T21 | feature-agent | Prediction tensor generator | Combine local Monte Carlo sim output with server observations into final W×H×6 probability tensor; weight observed cells higher | T11, T12 |
| T22 | feature-agent | Submission pipeline | End-to-end: load round -> query server -> build predictions -> submit all 5 seeds -> report scores | T10, T20, T21 |
| T23 | feature-agent | Simulation calibration | Compare local sim predictions against server observations to detect systematic biases; adjust constants or blending weights | T11, T12 |
| T30 | qa-agent | Self-scoring evaluator | Compute entropy-weighted KL divergence locally to estimate our score before submitting; match server scoring formula exactly | - |
| T31 | qa-agent | Integration tests for API client | Mock-based tests for all API endpoints; test auth, error handling, rate limiting | T10 |
| T32 | qa-agent | End-to-end pipeline test | Test full pipeline from initial state through prediction to submission with mocked server | T22 |
| T33 | qa-agent | Prediction quality benchmarks | Run Monte Carlo predictions on known seeds, compute self-score, establish baseline and track improvements | T21, T30 |

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
