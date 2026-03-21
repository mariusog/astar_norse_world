"""Terrain types and grid utilities."""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class Terrain(IntEnum):
    """Terrain types matching the 6 prediction classes."""

    EMPTY = 0  # Ocean or plains
    SETTLEMENT = 1
    PORT = 2
    RUIN = 3
    FOREST = 4
    MOUNTAIN = 5

    NUM_CLASSES = 6


# Internal terrain subtypes (ocean vs plains both map to EMPTY for prediction)
class InternalTerrain(IntEnum):
    """Fine-grained terrain for simulation internals."""

    OCEAN = 0
    PLAINS = 1
    SETTLEMENT = 2
    PORT = 3
    RUIN = 4
    FOREST = 5
    MOUNTAIN = 6

    def to_prediction_class(self) -> Terrain:
        """Map internal terrain to the 6-class prediction target."""
        mapping = {
            InternalTerrain.OCEAN: Terrain.EMPTY,
            InternalTerrain.PLAINS: Terrain.EMPTY,
            InternalTerrain.SETTLEMENT: Terrain.SETTLEMENT,
            InternalTerrain.PORT: Terrain.PORT,
            InternalTerrain.RUIN: Terrain.RUIN,
            InternalTerrain.FOREST: Terrain.FOREST,
            InternalTerrain.MOUNTAIN: Terrain.MOUNTAIN,
        }
        return mapping[self]


# ---------------------------------------------------------------------------
# Server terrain code mappings (canonical — import from here, not elsewhere)
# ---------------------------------------------------------------------------

# Server codes → InternalTerrain
# Server: 0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain,
#         10=Ocean, 11=Plains
SERVER_TO_INTERNAL: dict[int, InternalTerrain] = {
    0: InternalTerrain.PLAINS,
    1: InternalTerrain.SETTLEMENT,
    2: InternalTerrain.PORT,
    3: InternalTerrain.RUIN,
    4: InternalTerrain.FOREST,
    5: InternalTerrain.MOUNTAIN,
    10: InternalTerrain.OCEAN,
    11: InternalTerrain.PLAINS,
}

# Server codes → prediction class index (0-5)
SERVER_TO_PRED_CLASS: dict[int, int] = {
    0: Terrain.EMPTY,
    1: Terrain.SETTLEMENT,
    2: Terrain.PORT,
    3: Terrain.RUIN,
    4: Terrain.FOREST,
    5: Terrain.MOUNTAIN,
    10: Terrain.EMPTY,
    11: Terrain.EMPTY,
}

# Default InternalTerrain for unknown server codes
SERVER_CODE_DEFAULT = InternalTerrain.PLAINS


def map_server_codes(grid_patch: np.ndarray) -> np.ndarray:
    """Map raw server terrain codes to prediction class indices.

    Server codes 0-5 map to prediction classes 0-5.
    Server codes 10, 11 map to class 0 (Empty).

    Args:
        grid_patch: Array of server terrain codes.
    Returns:
        Array of prediction class indices (0-5).
    Raises:
        ValueError: If any code is not in SERVER_TO_PRED_CLASS.
    """
    max_code = max(SERVER_TO_PRED_CLASS) + 1
    lookup = np.full(max_code, -1, dtype=np.int8)
    for code, cls in SERVER_TO_PRED_CLASS.items():
        lookup[code] = cls
    flat = grid_patch.ravel()
    if np.any((flat < 0) | (flat >= max_code)):
        bad = flat[(flat < 0) | (flat >= max_code)]
        raise ValueError(f"Server codes out of range: {bad}")
    result = lookup[flat].reshape(grid_patch.shape)
    if np.any(result < 0):
        unmapped = flat[lookup[flat] < 0]
        raise ValueError(f"Unmapped server codes: {np.unique(unmapped)}")
    return result.astype(np.int8)


def server_grid_to_internal(grid_data: list[list[int]]) -> np.ndarray:
    """Convert a server grid (list of lists) to InternalTerrain ndarray."""
    arr = np.array(grid_data, dtype=np.int32)
    mapper = np.vectorize(lambda v: SERVER_TO_INTERNAL.get(v, SERVER_CODE_DEFAULT))
    return mapper(arr).astype(np.int8)


def neighbors_4(x: int, y: int, width: int, height: int) -> list[tuple[int, int]]:
    """Return valid 4-connected neighbors within grid bounds."""
    result = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            result.append((nx, ny))
    return result


def neighbors_8(x: int, y: int, width: int, height: int) -> list[tuple[int, int]]:
    """Return valid 8-connected neighbors within grid bounds."""
    result = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                result.append((nx, ny))
    return result


def grid_to_prediction(grid: np.ndarray) -> np.ndarray:
    """Convert an InternalTerrain grid to a prediction-class grid."""
    vectorized = np.vectorize(lambda t: InternalTerrain(t).to_prediction_class())
    return vectorized(grid).astype(np.int8)
