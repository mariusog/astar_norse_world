# Agent Plan: core-agent

**Owner**: core-agent (exclusively). Lead-agent creates tasks here; you fill out checklists and results.

## Active Tasks

### T10: API client for competition server
**Status**: open
**Branch**: `core/T10-api-client`
**Target**: All 4 API endpoints working with auth and error handling

Create `src/api_client.py` (under 300 lines). Must handle:

- [ ] Define `AstarClient` class wrapping `requests.Session`
- [ ] Constructor accepts JWT token, sets up auth (both cookie and bearer header)
- [ ] `list_rounds() -> list[dict]` -- GET /astar-island/rounds
- [ ] `get_round(round_id) -> dict` -- GET /astar-island/rounds/{round_id}
- [ ] `query(round_id, seed_index, x, y, w, h) -> dict` -- POST /astar-island/simulate with viewport validation (w,h in 5-15)
- [ ] `submit(round_id, seed_index, prediction: np.ndarray) -> dict` -- POST /astar-island/submit, auto-applies probability floor and renormalization
- [ ] `get_active_round() -> dict | None` -- convenience: find active round from list
- [ ] Track query count per round (warn at 45, hard-stop at 50)
- [ ] Raise typed exceptions: `AuthError`, `BudgetExhaustedError`, `APIError`
- [ ] Add retry with exponential backoff for transient failures (max 3 retries)
- [ ] All public methods have type annotations
- [ ] Self-review: `ruff check src/api_client.py && ruff format --check src/api_client.py`
- [ ] Tests pass

**Acceptance criteria**: Can authenticate, list rounds, query viewport, submit prediction. Query counter prevents exceeding budget.

**Result**:

---

### T11: Initial state loader from server response
**Status**: open
**Branch**: `core/T11-initial-state-loader`
**Target**: Parse server JSON into our internal data structures

Create `src/state_loader.py` (under 150 lines).

- [ ] `load_initial_state(state_json: dict) -> tuple[np.ndarray, list[Settlement]]` -- parse server's `grid` (list of lists of ints) and `settlements` into InternalTerrain grid + Settlement objects
- [ ] Map server terrain codes to our `InternalTerrain` enum (document any code differences)
- [ ] Handle server settlement format: extract x, y, owner_id, is_port; set default values for population, food, etc. from constants
- [ ] `load_round(round_json: dict) -> list[tuple[np.ndarray, list[Settlement]]]` -- load all seeds' initial states
- [ ] Validate grid dimensions match round's map_width/map_height
- [ ] All public methods have type annotations and docstrings
- [ ] Self-review: lint + format check
- [ ] Tests pass

**Acceptance criteria**: Server JSON round-trips correctly into our internal types. Grid dimensions validated. Settlement properties populated.

**Result**:

---

### T12: Observation aggregator
**Status**: open
**Branch**: `core/T12-observation-aggregator`
**Target**: Aggregate multiple viewport observations into per-cell probability estimates

Create `src/observation.py` (under 200 lines).

- [ ] Define `ObservationStore` class that accumulates viewport observations per seed
- [ ] `add_observation(seed_index, viewport_x, viewport_y, grid_patch: np.ndarray)` -- store one query result
- [ ] `get_observed_probs(seed_index) -> np.ndarray` -- return H x W x 6 probability tensor from observations only, using frequency counts where observed, NaN where unobserved
- [ ] `get_coverage_mask(seed_index) -> np.ndarray` -- boolean H x W mask of which cells have been observed
- [ ] `observation_count(seed_index) -> np.ndarray` -- H x W count of how many times each cell was observed
- [ ] Apply Laplace smoothing (add-1) to avoid zero probabilities in observed cells
- [ ] Handle overlapping viewports correctly (accumulate counts)
- [ ] Type annotations on all public methods
- [ ] Self-review: lint + format check
- [ ] Tests pass

**Acceptance criteria**: Multiple overlapping observations merge correctly. Unobserved cells are distinguished from observed. Probability floors applied.

**Result**:

---

## Escalations

Tasks that need lead-agent attention. Tag each as `BLOCKED` or `CRITICAL`.

| Tag | Task | Description |
|-----|------|-------------|
| - | - | - |

## Completed Tasks

### T40: Historical round collector
**Status**: done
**Branch**: `core/T40-T42-data-collection-features`

- [x] `collect_all_rounds(client, data_dir)` — fetch all completed rounds, save ground truth + initial states
- [x] Idempotent: skip already-captured rounds (check if ground_truth.npy exists)
- [x] For each round: save round.json, per-seed ground_truth.npy, initial_grid.npy, initial_settlements.json, analysis_meta.json
- [x] CLI: `python -m src.round_collector --token <JWT>` (also accept ASTAR_TOKEN env var)
- [x] Uses RoundClient protocol (compatible with future api_client.AstarClient)
- [x] Type annotations, lint clean

**Result**:
- **What changed**: Created `src/round_collector.py` (236 lines) with server terrain mapping, idempotent collection, and CLI entry point
- **Metrics**: 11 tests passing for round collector
- **Tests**: tests/test_round_collector.py with MockClient, grid parsing, idempotency, and end-to-end collection tests

---

### T41: Multi-round terrain prior builder
**Status**: done
**Branch**: `core/T40-T42-data-collection-features`

- [x] `build_terrain_priors(data_dir)` — scan all rounds, aggregate GT per initial terrain type
- [x] Weight recent rounds higher (exponential decay with round_number)
- [x] `save_priors(priors, path)` and `load_priors(path)` for persistence
- [x] `build_prior_prediction(grid, priors)` — apply priors to grid → H×W×6 tensor with floor + renormalize
- [x] Proper probability floor maintenance (iterative clamp-and-redistribute)
- [x] Type annotations, lint clean

**Result**:
- **What changed**: Created `src/prior_builder.py` (263 lines) with terrain priors, decay weighting, and prediction generation
- **Metrics**: 13 tests passing for prior builder
- **Tests**: tests/test_prior_builder.py covering uniform priors, normalization, floor enforcement, save/load roundtrip, and prediction tensor correctness

---

### T42: Settlement proximity features
**Status**: done
**Branch**: `core/T40-T42-data-collection-features`

- [x] `compute_settlement_distance(grid)` — H×W Manhattan distance to nearest settlement/port via BFS
- [x] `compute_coastal_mask(grid)` — H×W bool for land cells adjacent to ocean
- [x] `compute_ocean_distance(grid)` — H×W distance to nearest ocean
- [x] `compute_forest_density(grid)` — forest count within radius using summed area table
- [x] `compute_feature_grid(grid)` — returns dict with all feature arrays
- [x] All functions O(H×W)
- [x] Type annotations, lint clean

**Result**:
- **What changed**: Created `src/features.py` (174 lines) with BFS distance fields, coastal detection, and forest density via SAT
- **Metrics**: 15 tests passing for features
- **Tests**: tests/test_features.py covering all 5 public functions with happy path, edge cases, and shape validation
