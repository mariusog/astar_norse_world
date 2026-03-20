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
| T300 | core-agent | Web: FastAPI backend + data API | FastAPI app with JSON endpoints for rounds, predictions, models; serves from data/rounds/ | - |
| T301 | core-agent | Web: Map explorer page | HTMX/Jinja page with 40x40 colored grid, GT/prediction/observation overlays, seed selector, terrain legend | T300 |
| T302 | feature-agent | Web: Model research dashboard | Strategy leaderboard with LOO scores, per-round bars, run/compare, automated model search | T300 |
| T303 | feature-agent | Web: Round dashboard + submission | All rounds with scores, regime, active status, one-click submit with live progress | T300 |
| T304 | qa-agent | Web: Backtest page | Interactive LOO backtest: pick strategy, run against history, per-round scores with charts | T300 |

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
| T70 | core-agent | Unified survive-weighted prior builder | Completed -- src/unified_priors.py, survive 3x weighting, distance priors, 16 tests |
| T71 | core-agent | Dynamic cell classifier | Completed -- src/cell_classifier.py, 35-54% dynamic fraction, 14 tests |
| T72 | core-agent | Observation-focused query planner | Completed -- src/query_planner_v2.py, cluster-based viewport placement, 11 tests |
| T80 | feature-agent | Clean submission script v2 | Completed -- scripts/submit_v2.py, no regime detection, survive priors + observations |
| T90 | qa-agent | Backtest framework | Completed -- scripts/backtest.py, LOO/full modes, configurable obs density |
| T91 | qa-agent | Post-round automation | Completed -- scripts/post_round.py, capture + backtest + commit |
| T100 | core-agent | Distance priors in submit_v2 | Completed -- submit_v2 uses build_distance_priors, +1-8 pts |
| T101 | core-agent | Observation blending pipeline test | Completed -- tests/test_submit_pipeline.py, 9 pipeline tests |
| T102 | core-agent | Pre-submission validation | Completed -- src/prediction_validator.py with prior consistency + backtest checks |
| T110 | feature-agent | Soft regime blending | Completed -- src/soft_regime.py, +7.8 on R3 |
| T120 | qa-agent | Pre-submission validator | Completed -- 6 sanity checks, catches all 3 past failure modes |
