"""Query budget optimizer for viewport placement.

Plans which viewport rectangles to query for each seed to maximize
map coverage and information gain within the 50-query budget.
"""

from __future__ import annotations

import logging
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

    Budget: TOTAL_QUERY_BUDGET queries across NUM_SEEDS seeds.
    Phase 1: Tile the map with max-size viewports for coverage.
    Phase 2: Target high-uncertainty areas near settlements.

    Args:
        map_width: Width of the map grid.
        map_height: Height of the map grid.
        total_budget: Total queries allowed across all seeds.
        num_seeds: Number of seeds in the round.
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
        """Generate tiling viewports for initial map coverage.

        Uses max-size viewports to tile the map with minimal overlap.
        Allocates QUERIES_PER_SEED_COVERAGE queries per seed.

        Args:
            seed_index: Which seed to plan for (0-based).
            initial_grid: H x W initial terrain grid.

        Returns:
            List of Viewport objects for initial coverage.
        """
        tiles = _compute_tiling(
            self._width,
            self._height,
            QUERIES_PER_SEED_COVERAGE,
        )
        viewports = []
        for x, y, w, h in tiles:
            vp = Viewport(
                seed_index=seed_index,
                viewport_x=x,
                viewport_y=y,
                viewport_w=w,
                viewport_h=h,
            )
            viewports.append(vp)
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
        """Pick the most informative viewport based on current state.

        Targets: uncovered cells near settlements, coastal zones,
        and expansion corridors (plains near settlements).

        Args:
            seed_index: Which seed to query.
            coverage_mask: H x W boolean mask of already-observed cells.
            initial_grid: H x W initial terrain grid.

        Returns:
            A Viewport for the next query, or None if budget exhausted.
        """
        if self.queries_remaining <= 0:
            return None

        interest = _compute_interest_map(
            initial_grid,
            coverage_mask,
            self._width,
            self._height,
        )
        return _select_best_viewport(
            interest,
            seed_index,
            self._width,
            self._height,
        )


# ---------------------------------------------------------------------------
# Tiling helpers
# ---------------------------------------------------------------------------


def _compute_tiling(
    map_w: int,
    map_h: int,
    max_queries: int,
) -> list[tuple[int, int, int, int]]:
    """Compute viewport positions that tile the map.

    Uses greedy row-major tiling with max viewport size.
    Returns list of (x, y, w, h) tuples.
    """
    tile_size = VIEWPORT_MAX_SIZE
    tiles: list[tuple[int, int, int, int]] = []

    y_positions = _axis_positions(map_h, tile_size)
    x_positions = _axis_positions(map_w, tile_size)

    for y_start in y_positions:
        for x_start in x_positions:
            if len(tiles) >= max_queries:
                break
            w = min(tile_size, map_w - x_start)
            h = min(tile_size, map_h - y_start)
            w = max(w, VIEWPORT_MIN_SIZE)
            h = max(h, VIEWPORT_MIN_SIZE)
            w = min(w, VIEWPORT_MAX_SIZE)
            h = min(h, VIEWPORT_MAX_SIZE)
            tiles.append((x_start, y_start, w, h))
        if len(tiles) >= max_queries:
            break

    return tiles


def _axis_positions(map_size: int, tile_size: int) -> list[int]:
    """Compute start positions along one axis for tiling.

    Distributes tiles to cover the full axis with minimal overlap.
    """
    if map_size <= tile_size:
        return [0]

    num_tiles = _tiles_needed(map_size, tile_size)
    if num_tiles == 1:
        return [0]

    # Distribute tiles evenly with overlap
    positions = []
    total_coverage = num_tiles * tile_size
    overlap = (total_coverage - map_size) / (num_tiles - 1)
    step = tile_size - overlap

    for i in range(num_tiles):
        pos = round(i * step)
        pos = min(pos, map_size - VIEWPORT_MIN_SIZE)
        pos = max(pos, 0)
        positions.append(pos)

    return positions


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
        y_end = min(y + h, map_h)
        x_end = min(x + w, map_w)
        covered[y:y_end, x:x_end] = True
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
    """Score each cell by how much we want to observe it.

    High interest: uncovered cells near settlements/ports,
    expansion corridors (plains near settlements), coastlines.
    """
    interest = np.zeros((map_h, map_w), dtype=np.float64)

    # Uncovered cells get base interest
    uncovered = ~coverage_mask
    interest[uncovered] += 1.0

    # Cells near settlements/ports get bonus
    settlement_mask = _find_settlement_cells(initial_grid)
    proximity = _distance_field(settlement_mask, map_w, map_h)
    near_settlement = proximity <= 5
    interest[near_settlement & uncovered] += 3.0

    # Plains near settlements (expansion zones)
    plains_mask = initial_grid == InternalTerrain.PLAINS
    interest[plains_mask & near_settlement & uncovered] += 2.0

    # Coastal cells (adjacent to ocean)
    ocean_mask = initial_grid == InternalTerrain.OCEAN
    coastal = _dilate_mask(ocean_mask) & ~ocean_mask
    interest[coastal & uncovered] += 1.5

    return interest


def _find_settlement_cells(grid: np.ndarray) -> np.ndarray:
    """Return boolean mask of settlement and port cells."""
    return (grid == InternalTerrain.SETTLEMENT) | (grid == InternalTerrain.PORT)


def _distance_field(
    source_mask: np.ndarray,
    map_w: int,
    map_h: int,
) -> np.ndarray:
    """Compute Manhattan distance from each cell to nearest source.

    Uses bounded BFS for safety (max map_w * map_h iterations).
    """
    from collections import deque

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
    """Select the viewport that covers the highest total interest.

    Tries multiple viewport sizes and positions, picks the best.
    Uses a sliding window approach bounded to reasonable candidates.
    """
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
    return Viewport(
        seed_index=seed_index,
        viewport_x=x,
        viewport_y=y,
        viewport_w=w,
        viewport_h=h,
    )
