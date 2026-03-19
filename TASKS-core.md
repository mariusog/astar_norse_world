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
