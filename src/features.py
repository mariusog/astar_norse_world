"""Spatial feature extraction for prediction enhancement.

Computes settlement proximity, coastal masks, and other grid-level
features that improve terrain prediction accuracy.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from src.terrain import InternalTerrain, neighbors_4


def compute_settlement_distance(grid: np.ndarray) -> np.ndarray:
    """Compute Manhattan distance to nearest settlement or port.

    Uses multi-source BFS from all settlement/port cells.
    Complexity: O(H * W).

    Args:
        grid: H x W array of InternalTerrain values.

    Returns:
        H x W int array of Manhattan distances. Cells that are
        settlements/ports have distance 0.
    """
    height, width = grid.shape
    dist = np.full((height, width), -1, dtype=np.int32)
    queue: deque[tuple[int, int]] = deque()

    # Seed BFS from all settlement and port cells
    for y in range(height):
        for x in range(width):
            val = int(grid[y, x])
            if val in (InternalTerrain.SETTLEMENT, InternalTerrain.PORT):
                dist[y, x] = 0
                queue.append((x, y))

    _bfs_fill(queue, dist, width, height)
    # Any unreachable cells get max distance
    dist[dist < 0] = width + height
    return dist


def compute_coastal_mask(grid: np.ndarray) -> np.ndarray:
    """Identify land cells adjacent to ocean.

    Args:
        grid: H x W array of InternalTerrain values.

    Returns:
        H x W bool array. True for land cells with at least one
        ocean neighbor.
    """
    height, width = grid.shape
    mask = np.zeros((height, width), dtype=bool)

    for y in range(height):
        for x in range(width):
            if _is_land_cell(int(grid[y, x])):
                for nx, ny in neighbors_4(x, y, width, height):
                    if int(grid[ny, nx]) == InternalTerrain.OCEAN:
                        mask[y, x] = True
                        break

    return mask


def compute_ocean_distance(grid: np.ndarray) -> np.ndarray:
    """Compute Manhattan distance to nearest ocean cell.

    Uses multi-source BFS from all ocean cells.
    Complexity: O(H * W).

    Args:
        grid: H x W array of InternalTerrain values.

    Returns:
        H x W int array of distances. Ocean cells have distance 0.
    """
    height, width = grid.shape
    dist = np.full((height, width), -1, dtype=np.int32)
    queue: deque[tuple[int, int]] = deque()

    for y in range(height):
        for x in range(width):
            if int(grid[y, x]) == InternalTerrain.OCEAN:
                dist[y, x] = 0
                queue.append((x, y))

    _bfs_fill(queue, dist, width, height)
    dist[dist < 0] = width + height
    return dist


def compute_forest_density(grid: np.ndarray, radius: int = 3) -> np.ndarray:
    """Count forest cells within Manhattan distance of each cell.

    Uses a summed area table for O(H * W) complexity regardless
    of radius.

    Args:
        grid: H x W array of InternalTerrain values.
        radius: Manhattan distance radius for counting.

    Returns:
        H x W int array of forest neighbor counts.
    """
    height, width = grid.shape
    forest = (grid == InternalTerrain.FOREST).astype(np.int32)

    # Summed area table
    sat = np.zeros((height + 1, width + 1), dtype=np.int32)
    sat[1:, 1:] = np.cumsum(np.cumsum(forest, axis=0), axis=1)

    result = np.zeros((height, width), dtype=np.int32)
    for y in range(height):
        for x in range(width):
            y0 = max(0, y - radius)
            y1 = min(height - 1, y + radius)
            x0 = max(0, x - radius)
            x1 = min(width - 1, x + radius)
            result[y, x] = _sat_query(sat, y0, x0, y1, x1)

    return result


def compute_feature_grid(grid: np.ndarray) -> dict[str, np.ndarray]:
    """Compute all spatial features for a terrain grid.

    Args:
        grid: H x W array of InternalTerrain values.

    Returns:
        Dictionary with feature name -> H x W array:
        - settlement_dist: distance to nearest settlement/port
        - coastal_mask: bool mask of coastal land cells
        - ocean_dist: distance to nearest ocean cell
        - forest_density: count of nearby forest cells
    """
    return {
        "settlement_dist": compute_settlement_distance(grid),
        "coastal_mask": compute_coastal_mask(grid),
        "ocean_dist": compute_ocean_distance(grid),
        "forest_density": compute_forest_density(grid),
    }


def _bfs_fill(
    queue: deque[tuple[int, int]],
    dist: np.ndarray,
    width: int,
    height: int,
) -> None:
    """Run BFS from pre-seeded queue to fill distance array."""
    while queue:
        x, y = queue.popleft()
        d = int(dist[y, x])
        for nx, ny in neighbors_4(x, y, width, height):
            if dist[ny, nx] < 0:
                dist[ny, nx] = d + 1
                queue.append((nx, ny))


def _is_land_cell(val: int) -> bool:
    """Check if a terrain value represents a land cell."""
    return val != InternalTerrain.OCEAN


def _sat_query(sat: np.ndarray, y0: int, x0: int, y1: int, x1: int) -> int:
    """Query summed area table for rectangle sum."""
    return int(sat[y1 + 1, x1 + 1] - sat[y0, x1 + 1] - sat[y1 + 1, x0] + sat[y0, x0])
