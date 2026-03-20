# Agent Plan: core-agent

**Owner**: core-agent (exclusively). Lead-agent creates tasks here; you fill out checklists and results.

## Active Tasks

### T10: API client for competition server
**Status**: done
**Branch**: `core/T10-api-client`
**Target**: All 4 API endpoints working with auth and error handling

Create `src/api_client.py` (under 300 lines). Must handle:

- [x] Define `AstarClient` class wrapping `requests.Session`
- [x] Constructor accepts JWT token, sets up auth (both cookie and bearer header)
- [x] `list_rounds() -> list[dict]` -- GET /astar-island/rounds
- [x] `get_round(round_id) -> dict` -- GET /astar-island/rounds/{round_id}
- [x] `query(round_id, seed_index, x, y, w, h) -> dict` -- POST /astar-island/simulate with viewport validation (w,h in 5-15)
- [x] `submit(round_id, seed_index, prediction: np.ndarray) -> dict` -- POST /astar-island/submit, auto-applies probability floor and renormalization
- [x] `get_active_round() -> dict | None` -- convenience: find active round from list
- [x] Track query count per round (warn at 45, hard-stop at 50)
- [x] Raise typed exceptions: `AuthError`, `BudgetExhaustedError`, `APIError`
- [x] Add retry with exponential backoff for transient failures (max 3 retries)
- [x] All public methods have type annotations
- [x] Self-review: `ruff check src/api_client.py && ruff format --check src/api_client.py`
- [x] Tests pass

**Acceptance criteria**: Can authenticate, list rounds, query viewport, submit prediction. Query counter prevents exceeding budget.

**Result**: Created `src/api_client.py` (238 lines) with AstarClient class. All 4 endpoints implemented. Budget tracking with warning at 45 and hard-stop at 50. Typed exceptions (AuthError, BudgetExhaustedError, APIError). Exponential backoff retry for transient failures. Auto probability floor + renormalization on submit. 19 tests added in `tests/test_api_client.py`, all passing. Clean lint.

---

### T11: Initial state loader from server response
**Status**: done
**Branch**: `core/T11-initial-state-loader`
**Target**: Parse server JSON into our internal data structures

Create `src/state_loader.py` (under 150 lines).

- [x] `load_initial_state(state_json: dict) -> tuple[np.ndarray, list[Settlement]]` -- parse server's `grid` (list of lists of ints) and `settlements` into InternalTerrain grid + Settlement objects
- [x] Map server terrain codes to our `InternalTerrain` enum (document any code differences)
- [x] Handle server settlement format: extract x, y, owner_id, is_port; set default values for population, food, etc. from constants
- [x] `load_round(round_json: dict) -> list[tuple[np.ndarray, list[Settlement]]]` -- load all seeds' initial states
- [x] Validate grid dimensions match round's map_width/map_height
- [x] All public methods have type annotations and docstrings
- [x] Self-review: lint + format check
- [x] Tests pass

**Acceptance criteria**: Server JSON round-trips correctly into our internal types. Grid dimensions validated. Settlement properties populated.

**Result**: Created `src/state_loader.py` (148 lines) with load_initial_state and load_round functions. Server terrain codes 0-6 map to InternalTerrain enum. Settlement fields use defaults from constants when not provided. Grid dimension validation against round metadata. 16 tests added in `tests/test_state_loader.py`, all passing. Clean lint.

---

### T12: Observation aggregator
**Status**: done
**Branch**: `core/T12-observation-aggregator`
**Target**: Aggregate multiple viewport observations into per-cell probability estimates

Create `src/observation.py` (under 200 lines).

- [x] Define `ObservationStore` class that accumulates viewport observations per seed
- [x] `add_observation(seed_index, viewport_x, viewport_y, grid_patch: np.ndarray)` -- store one query result
- [x] `get_observed_probs(seed_index) -> np.ndarray` -- return H x W x 6 probability tensor from observations only, using frequency counts where observed, NaN where unobserved
- [x] `get_coverage_mask(seed_index) -> np.ndarray` -- boolean H x W mask of which cells have been observed
- [x] `observation_count(seed_index) -> np.ndarray` -- H x W count of how many times each cell was observed
- [x] Apply Laplace smoothing (add-1) to avoid zero probabilities in observed cells
- [x] Handle overlapping viewports correctly (accumulate counts)
- [x] Type annotations on all public methods
- [x] Self-review: lint + format check
- [x] Tests pass

**Acceptance criteria**: Multiple overlapping observations merge correctly. Unobserved cells are distinguished from observed. Probability floors applied.

**Result**: Created `src/observation.py` (163 lines) with ObservationStore class. Frequency counting with Laplace (add-1) smoothing for observed cells, NaN for unobserved. Overlapping viewports accumulate correctly. Coverage mask and observation count accessors. Out-of-bounds viewport cells clipped. Seeds isolated. 18 tests added in `tests/test_observation.py`, all passing. Clean lint.

---

---

### T40: Historical round collector
**Status**: open
**Branch**: `core/T40-round-collector`
**Target**: All completed rounds captured to `data/rounds/` with ground truth

Create `src/round_collector.py` (under 200 lines).

- [ ] `collect_all_rounds(client: AstarClient, data_dir: str = "data/rounds") -> list[str]` — fetch all completed/scoring rounds, skip already-captured
- [ ] For each round: call `client.get_round()` → save `round.json`
- [ ] For each round: call `client.analysis(round_id, seed_index)` for all seeds → save `ground_truth.npy`, `analysis_meta.json`
- [ ] Parse initial states via `load_round()` → save `initial_grid.npy`, `initial_settlements.json`
- [ ] Idempotent: check if `ground_truth.npy` exists before fetching
- [ ] Return list of round_ids that were newly captured
- [ ] Add CLI: `python -m src.round_collector --token <JWT>`
- [ ] JWT token read from `--token` arg or `ASTAR_TOKEN` env var
- [ ] Type annotations, lint clean, tests pass
- [ ] Log each round captured with seed count

**Acceptance criteria**: Running twice produces the same output. All completed rounds have ground truth saved.

**Result**:

---

### T41: Multi-round terrain prior builder
**Status**: open
**Branch**: `core/T41-terrain-priors`
**Target**: Aggregate ground truth across rounds into reusable priors
**Depends on**: T40

Create `src/prior_builder.py` (under 200 lines).

- [ ] `build_terrain_priors(data_dir: str = "data/rounds") -> dict[int, np.ndarray]` — scan all rounds, aggregate GT per initial terrain type
- [ ] For each `InternalTerrain` type, collect all matching cells' GT probability vectors across all rounds and seeds
- [ ] Weight more recent rounds higher (e.g. exponential decay with `round_number`)
- [ ] Return dict mapping `InternalTerrain` int value → mean probability vector (shape 6,)
- [ ] `save_priors(priors: dict, path: str = "data/priors.npy")` and `load_priors(path)` — persist/load
- [ ] `build_prior_prediction(grid: np.ndarray, priors: dict) -> np.ndarray` — apply priors to a grid, returning H×W×6 tensor with floor + renormalization
- [ ] Handle missing terrain types gracefully (fall back to uniform)
- [ ] All public methods have type annotations
- [ ] Self-review: lint + format check
- [ ] Tests pass

**Acceptance criteria**: Priors built from R1+R2 score ≥85 when backtested against R2. Loading from disk matches in-memory.

**Result**:

---

### T42: Settlement proximity features
**Status**: open
**Branch**: `core/T42-proximity-features`
**Target**: Per-cell feature arrays that improve prediction beyond flat terrain priors
**Depends on**: T40

Create `src/features.py` (under 200 lines).

- [ ] `compute_settlement_distance(grid: np.ndarray) -> np.ndarray` — H×W Manhattan distance to nearest initial settlement/port cell
- [ ] `compute_coastal_mask(grid: np.ndarray) -> np.ndarray` — H×W bool, True for land cells adjacent to ocean
- [ ] `compute_feature_grid(grid: np.ndarray) -> dict[str, np.ndarray]` — returns dict with `settlement_dist`, `coastal`, `terrain_type` arrays
- [ ] Reuse `_distance_field` from `query_strategy.py` (extract to shared util if needed) or reimplement with BFS
- [ ] All functions O(H×W) or better
- [ ] Type annotations, lint clean
- [ ] Tests: verify distances correct on small grids, coastal detection at borders

**Acceptance criteria**: Features computable in <50ms for 40×40 grid. Correct distance field verified by test.

**Result**:

---

## Escalations

Tasks that need lead-agent attention. Tag each as `BLOCKED` or `CRITICAL`.

| Tag | Task | Description |
|-----|------|-------------|
| - | - | - |

## Completed Tasks
