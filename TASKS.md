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
| T70 | core-agent | Unified survive-weighted prior builder | Build single prior set from all 5 rounds, weighted toward survive regime (3/5 rounds); drop regime detection; output to `data/priors.npy` | - |
| T71 | core-agent | Dynamic cell classifier | Classify each cell as static (ocean/mountain, needs 0 queries) or dynamic (near settlements, needs observations); output boolean mask per seed | - |
| T72 | core-agent | Observation-focused query planner | All 50 queries go to observations, zero probes; place overlapping viewports on dynamic cells to maximize obs/cell; target 4-5 obs per dynamic cell | T71 |
| T80 | feature-agent | Clean submission script v2 | Single script: load priors → plan queries on dynamic cells → observe → blend → submit; no regime detection; use survive priors always; target 80+ on backtest | T70, T72 |
| T81 | feature-agent | Soft regime blending from observations | After observing cells, estimate regime confidence from observed settlement survival rate; soft-blend survive/collapse priors based on confidence; safer than binary detection | T70, T80 |
| T82 | feature-agent | Per-terrain-type observation weighting | Dynamic cells near settlements need more observations than cells far away; allocate viewport overlap proportional to expected entropy from priors | T71, T72 |
| T90 | qa-agent | Backtest framework | Simulate full submission pipeline against all 5 rounds using GT-sampled observations; report per-round and avg scores; run before every real submission | T80 |
| T91 | qa-agent | Capture and rebuild after each round | One-command: capture new round GT → rebuild priors → backtest → report | T70 |

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
| T40 | core-agent | Historical round collector | Completed -- src/round_collector.py with idempotent capture, CLI, 11 tests |
| T41 | core-agent | Multi-round terrain prior builder | Completed -- src/prior_builder.py with decay weighting, floor maintenance, 13 tests |
| T42 | core-agent | Settlement proximity features | Completed -- src/features.py with BFS distance, coastal mask, forest density, 15 tests |
| T51 | feature-agent | Prior-based predictor | Completed -- src/predictor_v2.py with terrain priors + distance adjustment, R2 backtest=86.6 |
| T53 | feature-agent | Position-aware priors | Completed -- src/position_priors.py with settlement distance model, +1.5 pts over flat |
| T60 | qa-agent | Round-over-round analysis | Completed -- scripts/analyze_rounds.py + docs/round_analysis.md |
