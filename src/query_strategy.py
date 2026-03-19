"""Query budget optimizer for viewport placement.

Plans which viewport rectangles to query for each seed to maximize
map coverage and information gain within the 50-query budget.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

from src.constants import (
    DEFAULT_MAP_HEIGHT,
    DEFAULT_MAP_WIDTH,
    NUM_SEEDS,
    QUERIES_PER_SEED_COVERAGE,
    TOTAL_QUERY_BUDGET,
    VIEWPORT_MAX_SIZE,
    VIEWPORT_MIN_SIZE,
)
from src.terrain import InternalTerrain

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Viewport:
    """A query viewport rectangle."""

    seed_index: int
    viewport_x: int
    viewport_y: int
    viewport_w: int
    viewport_h: int

    def to_dict(self) -> dict[str, int]:
        """Convert to API-compatible dict."""
        return {
            "seed_index": self.seed_index,
            "viewport_x": self.viewport_x,
            "viewport_y": self.viewport_y,
            "viewport_w": self.viewport_w,
            "viewport_h": self.viewport_h,
        }


class QueryPlanner:
    """Plan viewport placements to maximize coverage and info gain.

    Phase 1: Tile the map with max-size viewports for coverage.
    Phase 2: Target high-uncertainty areas near settlements.
    """

    def __init__(
        self,
        map_width: int = DEFAULT_MAP_WIDTH,
        map_height: int = DEFAULT_MAP_HEIGHT,
        total_budget: int = TOTAL_QUERY_BUDGET,
        num_seeds: int = NUM_SEEDS,
    ) -> None:
        self._width = map_width
        self._height = map_height
        self._total_budget = total_budget
        self._num_seeds = num_seeds
        self._queries_used = 0

    @property
    def queries_remaining(self) -> int:
        """Return remaining query budget."""
        return self._total_budget - self._queries_used

    def record_query(self) -> None:
        """Record that a query has been used."""
        self._queries_used += 1

    def plan_initial_queries(
        self,
        seed_index: int,
        initial_grid: np.ndarray,
    ) -> list[Viewport]:
        """Generate tiling viewports for initial map coverage."""
        tiles = _compute_tiling(self._width, self._height, QUERIES_PER_SEED_COVERAGE)
        viewports = [
            Viewport(seed_index=seed_index, viewport_x=x, viewport_y=y, viewport_w=w, viewport_h=h)
            for x, y, w, h in tiles
        ]
        logger.info(
            "Planned %d coverage queries for seed %d (%.0f%% coverage)",
            len(viewports),
            seed_index,
            _estimate_coverage(tiles, self._width, self._height) * 100,
        )
        return viewports

    def plan_adaptive_query(
        self,
        seed_index: int,
        coverage_mask: np.ndarray,
        initial_grid: np.ndarray,
    ) -> Viewport | None:
        """Pick the most informative viewport based on current state."""
        if self.queries_remaining <= 0:
            return None
        interest = _compute_interest_map(initial_grid, coverage_mask, self._width, self._height)
        return _select_best_viewport(interest, seed_index, self._width, self._height)


# ---------------------------------------------------------------------------
# Tiling helpers
# ---------------------------------------------------------------------------


def _compute_tiling(
    map_w: int,
    map_h: int,
    max_queries: int,
) -> list[tuple[int, int, int, int]]:
    """Compute viewport positions that tile the map using row-major order."""
    tile_size = VIEWPORT_MAX_SIZE
    tiles: list[tuple[int, int, int, int]] = []
    for y_start in _axis_positions(map_h, tile_size):
        for x_start in _axis_positions(map_w, tile_size):
            if len(tiles) >= max_queries:
                return tiles
            w = _clamp(min(tile_size, map_w - x_start), VIEWPORT_MIN_SIZE, VIEWPORT_MAX_SIZE)
            h = _clamp(min(tile_size, map_h - y_start), VIEWPORT_MIN_SIZE, VIEWPORT_MAX_SIZE)
            tiles.append((x_start, y_start, w, h))
    return tiles


def _clamp(value: int, lo: int, hi: int) -> int:
    """Clamp value to [lo, hi] range."""
    return max(lo, min(value, hi))


def _axis_positions(map_size: int, tile_size: int) -> list[int]:
    """Compute start positions along one axis for even tiling."""
    if map_size <= tile_size:
        return [0]
    num_tiles = _tiles_needed(map_size, tile_size)
    if num_tiles == 1:
        return [0]
    overlap = (num_tiles * tile_size - map_size) / (num_tiles - 1)
    step = tile_size - overlap
    return [_clamp(round(i * step), 0, map_size - VIEWPORT_MIN_SIZE) for i in range(num_tiles)]


def _tiles_needed(map_size: int, tile_size: int) -> int:
    """Compute minimum number of tiles to cover an axis."""
    return max(1, (map_size + tile_size - 1) // tile_size)


def _estimate_coverage(
    tiles: list[tuple[int, int, int, int]],
    map_w: int,
    map_h: int,
) -> float:
    """Estimate fraction of map covered by tiles."""
    covered = np.zeros((map_h, map_w), dtype=bool)
    for x, y, w, h in tiles:
        covered[y : min(y + h, map_h), x : min(x + w, map_w)] = True
    return float(covered.sum()) / (map_w * map_h)


# ---------------------------------------------------------------------------
# Adaptive query helpers
# ---------------------------------------------------------------------------


def _compute_interest_map(
    initial_grid: np.ndarray,
    coverage_mask: np.ndarray,
    map_w: int,
    map_h: int,
) -> np.ndarray:
    """Score each cell by how much we want to observe it."""
    interest = np.zeros((map_h, map_w), dtype=np.float64)
    uncovered = ~coverage_mask
    interest[uncovered] += 1.0

    settlement_mask = _find_settlement_cells(initial_grid)
    proximity = _distance_field(settlement_mask, map_w, map_h)
    near_settlement = proximity <= 5
    interest[near_settlement & uncovered] += 3.0

    plains_mask = initial_grid == InternalTerrain.PLAINS
    interest[plains_mask & near_settlement & uncovered] += 2.0

    ocean_mask = initial_grid == InternalTerrain.OCEAN
    coastal = _dilate_mask(ocean_mask) & ~ocean_mask
    interest[coastal & uncovered] += 1.5
    return interest


def _find_settlement_cells(grid: np.ndarray) -> np.ndarray:
    """Return boolean mask of settlement and port cells."""
    return (grid == InternalTerrain.SETTLEMENT) | (grid == InternalTerrain.PORT)


def _distance_field(source_mask: np.ndarray, map_w: int, map_h: int) -> np.ndarray:
    """Compute Manhattan distance from each cell to nearest source via bounded BFS."""
    dist = np.full((map_h, map_w), map_w + map_h, dtype=np.int32)
    queue: deque[tuple[int, int]] = deque()
    for y in range(map_h):
        for x in range(map_w):
            if source_mask[y, x]:
                dist[y, x] = 0
                queue.append((y, x))
    max_steps = map_w * map_h
    steps = 0
    while queue and steps < max_steps:
        cy, cx = queue.popleft()
        steps += 1
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < map_h and 0 <= nx < map_w and dist[ny, nx] > dist[cy, cx] + 1:
                dist[ny, nx] = dist[cy, cx] + 1
                queue.append((ny, nx))
    return dist


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    """Expand a boolean mask by one cell in 4-connected directions."""
    result = mask.copy()
    result[1:, :] |= mask[:-1, :]
    result[:-1, :] |= mask[1:, :]
    result[:, 1:] |= mask[:, :-1]
    result[:, :-1] |= mask[:, 1:]
    return result


def _select_best_viewport(
    interest: np.ndarray,
    seed_index: int,
    map_w: int,
    map_h: int,
) -> Viewport:
    """Select the viewport covering the highest total interest score."""
    best_score = -1.0
    best_vp = (0, 0, VIEWPORT_MIN_SIZE, VIEWPORT_MIN_SIZE)
    step = VIEWPORT_MIN_SIZE
    for vw in (VIEWPORT_MAX_SIZE, 10, VIEWPORT_MIN_SIZE):
        for vh in (VIEWPORT_MAX_SIZE, 10, VIEWPORT_MIN_SIZE):
            for y in range(0, map_h - vh + 1, step):
                for x in range(0, map_w - vw + 1, step):
                    score = float(interest[y : y + vh, x : x + vw].sum())
                    if score > best_score:
                        best_score = score
                        best_vp = (x, y, vw, vh)
    x, y, w, h = best_vp
    return Viewport(seed_index=seed_index, viewport_x=x, viewport_y=y, viewport_w=w, viewport_h=h)
