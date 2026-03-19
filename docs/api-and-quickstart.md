# Astar Island - API & Quickstart

## Base URL

```
https://api.ainm.no
```

## Authentication

Two options — both use JWT tokens:

**Cookie-based:**
```python
import requests
session = requests.Session()
session.cookies.set("access_token", "YOUR_JWT_TOKEN")
```

**Bearer token:**
```python
session = requests.Session()
session.headers["Authorization"] = "Bearer YOUR_JWT_TOKEN"
```

Get your token by logging into `app.ainm.no` and inspecting browser cookies (DevTools -> Application -> Cookies -> `access_token`).

## Endpoints

### 1. List Rounds

```
GET /astar-island/rounds
```

Returns list of rounds. Find the active one:

```python
rounds = session.get(f"{BASE}/astar-island/rounds").json()
active = next((r for r in rounds if r["status"] == "active"), None)
round_id = active["id"]
```

### 2. Get Round Details

```
GET /astar-island/rounds/{round_id}
```

Returns:
- `map_width` (e.g. 40)
- `map_height` (e.g. 40)
- `seeds_count` (e.g. 5)
- `initial_states` — array of per-seed initial states, each containing:
  - `grid` — height x width terrain codes
  - `settlements` — list of initial settlements

```python
detail = session.get(f"{BASE}/astar-island/rounds/{round_id}").json()
width = detail["map_width"]       # 40
height = detail["map_height"]     # 40
seeds = detail["seeds_count"]     # 5

for i, state in enumerate(detail["initial_states"]):
    grid = state["grid"]              # height x width terrain codes
    settlements = state["settlements"]
```

### 3. Simulate (Query)

```
POST /astar-island/simulate
```

**Request body:**
```json
{
    "round_id": "<round_id>",
    "seed_index": 0,
    "viewport_x": 10,
    "viewport_y": 5,
    "viewport_w": 15,
    "viewport_h": 15
}
```

**Constraints:**
- `viewport_w` and `viewport_h`: 5-15 cells
- **50 queries total per round** shared across all 5 seeds

**Response:**
```json
{
    "grid": [[...], ...],
    "settlements": [...],
    "viewport": {"x": 10, "y": 5, "w": 15, "h": 15}
}
```

- `grid` — viewport-sized terrain grid (final state after 50 years)
- `settlements` — settlements within the viewport (with full properties)
- `viewport` — confirmed viewport coordinates

### 4. Submit Prediction

```
POST /astar-island/submit
```

**Request body:**
```json
{
    "round_id": "<round_id>",
    "seed_index": 0,
    "prediction": [[[0.01, 0.60, 0.25, 0.13, 0.005, 0.005], ...], ...]
}
```

- `prediction` — height x width x 6 array of probabilities
- Each cell sums to 1.0
- **Never use 0.0** — use min floor of 0.01 and renormalize

```python
import numpy as np

prediction = np.full((height, width, 6), 1/6)  # uniform baseline
# ... improve predictions with observations ...

# Safety floor
prediction = np.maximum(prediction, 0.01)
prediction = prediction / prediction.sum(axis=2, keepdims=True)

resp = session.post(f"{BASE}/astar-island/submit", json={
    "round_id": round_id,
    "seed_index": seed_idx,
    "prediction": prediction.tolist(),
})
```

## MCP Server for Docs

```bash
claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp
```

## Query Budget Strategy

With 50 queries across 5 seeds on a 40x40 map:
- 10 queries per seed if split evenly
- Each 15x15 query covers 225/1600 = ~14% of the map
- 10 queries at max viewport could cover ~140% of the map per seed (with overlap)
- But each query is stochastic — same area queried twice gives different results
- Multiple queries of the same area help estimate probability distributions
