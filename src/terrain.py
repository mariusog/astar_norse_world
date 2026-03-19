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
