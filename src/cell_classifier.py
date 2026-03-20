"""Dynamic cell classifier for observation targeting.

Classifies map cells as static (predictable from priors alone) or
dynamic (require observation for accurate prediction). Dynamic cells
are near settlements, ports, and other areas of change.
"""

from __future__ import annotations

import numpy as np

from src.constants import DYNAMIC_FOREST_RADIUS, DYNAMIC_SETTLEMENT_RADIUS
from src.features import compute_settlement_distance
from src.terrain import InternalTerrain

# Target fraction of map that should be classified dynamic


def classify_cells(grid: np.ndarray) -> np.ndarray:
    """Classify each cell as dynamic (True) or static (False).

    Static cells: ocean, mountain -- predictable with high confidence.
    Dynamic cells: settlements, ports, cells near them, forest near
    settlements -- these change across simulations.

    Args:
        grid: H x W array of InternalTerrain values.

    Returns:
        H x W boolean mask where True = dynamic cell.
    """
    h, w = grid.shape
    mask = np.zeros((h, w), dtype=bool)

    _mark_inherently_dynamic(grid, mask)
    dist = compute_settlement_distance(grid)
    _mark_proximity_dynamic(grid, dist, mask)

    return mask


def classify_static_confident(grid: np.ndarray) -> np.ndarray:
    """Return mask of cells we can predict with near-certainty.

    These are ocean and mountain cells that almost never change.

    Args:
        grid: H x W array of InternalTerrain values.

    Returns:
        H x W boolean mask where True = statically confident cell.
    """
    ocean = grid == InternalTerrain.OCEAN
    mountain = grid == InternalTerrain.MOUNTAIN
    return ocean | mountain


def dynamic_fraction(grid: np.ndarray) -> float:
    """Return fraction of cells classified as dynamic.

    Args:
        grid: H x W array of InternalTerrain values.

    Returns:
        Float in [0, 1].
    """
    mask = classify_cells(grid)
    total = grid.shape[0] * grid.shape[1]
    if total == 0:
        return 0.0
    return float(mask.sum()) / total


# -- Internal helpers -------------------------------------------------------


def _mark_inherently_dynamic(grid: np.ndarray, mask: np.ndarray) -> None:
    """Mark settlements and ports as always dynamic."""
    mask |= grid == InternalTerrain.SETTLEMENT
    mask |= grid == InternalTerrain.PORT
    mask |= grid == InternalTerrain.RUIN


def _mark_proximity_dynamic(
    grid: np.ndarray,
    dist: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Mark cells near settlements/ports as dynamic."""
    # Any cell within radius of settlement/port
    near = dist <= DYNAMIC_SETTLEMENT_RADIUS
    # Exclude ocean and mountain from proximity marking
    land = _is_changeable_terrain(grid)
    mask |= near & land

    # Forest cells near settlements get extra radius
    forest = grid == InternalTerrain.FOREST
    forest_near = dist <= DYNAMIC_FOREST_RADIUS
    mask |= forest & forest_near


def _is_changeable_terrain(grid: np.ndarray) -> np.ndarray:
    """Return mask of terrain types that can change in simulation."""
    return (grid != InternalTerrain.OCEAN) & (grid != InternalTerrain.MOUNTAIN)
