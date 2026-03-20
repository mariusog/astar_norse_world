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
| T40 | core-agent | Historical round collector | Auto-fetch ground truth, initial states, and analysis from ALL completed rounds; store in `data/rounds/` as `.npy` + JSON; idempotent (skip already-captured rounds) | - |
| T41 | core-agent | Multi-round terrain prior builder | Build per-terrain-type probability priors by aggregating ground truth across all historical rounds; weight recent rounds higher; output to `data/priors.npy` | T40 |
| T42 | core-agent | Settlement proximity features | For each cell, compute distance to nearest settlement, coastal flag, adjacent terrain types; store as feature arrays that improve per-cell prediction beyond flat terrain priors | T40 |
| T50 | feature-agent | Overlap-focused query strategy | Replace coverage-tiling with overlap strategy: skip static terrain (ocean/mountain), concentrate 50 queries on dynamic cells near settlements; aim for 3-5 observations per high-entropy cell | T42 |
| T51 | feature-agent | Prior-based predictor | Replace MC simulation prior with historical terrain priors from T41; use settlement proximity features from T42 for per-cell refinement; target: 85+ baseline before observations | T41, T42 |
| T52 | feature-agent | Improved submission pipeline | Update pipeline to: (1) load historical priors, (2) use overlap query strategy, (3) blend observations with count-scaled weights, (4) self-score before submit | T50, T51 |
| T53 | feature-agent | Per-cell position-aware priors | Learn that cells at specific (relative) positions around settlements have different distributions; e.g. cells 1 step from a settlement are more likely to become settlement vs 3 steps away | T40, T41 |
| T60 | qa-agent | Round-over-round analysis | Compare terrain priors across rounds; measure prior stability; identify which terrain types drift most; output report to `docs/round_analysis.md` | T40 |
| T61 | qa-agent | Query strategy backtesting | Backtest different query strategies against R1+R2 ground truth: (a) full tiling, (b) settlement-focused overlap, (c) hybrid; report scores for each | T40, T50 |
| T62 | qa-agent | Automated round capture hook | Create a script that runs after each round: captures data, rebuilds priors, backtests, reports expected scores for the next round | T40, T41 |

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
