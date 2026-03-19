"""Parse server JSON round data into internal data structures.

Converts the competition server's initial_states into InternalTerrain
grids and Settlement objects for local simulation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.constants import (
    INITIAL_FOOD,
    INITIAL_POPULATION,
)
from src.settlement import Settlement
from src.terrain import InternalTerrain

logger = logging.getLogger(__name__)

# Server terrain codes -> InternalTerrain mapping.
# The server uses integer codes for terrain; this maps them to our enum.
# If the server ever adds new codes, add them here.
_SERVER_TERRAIN_MAP: dict[int, InternalTerrain] = {
    0: InternalTerrain.OCEAN,
    1: InternalTerrain.PLAINS,
    2: InternalTerrain.SETTLEMENT,
    3: InternalTerrain.PORT,
    4: InternalTerrain.RUIN,
    5: InternalTerrain.FOREST,
    6: InternalTerrain.MOUNTAIN,
}


def load_initial_state(
    state_json: dict[str, Any],
) -> tuple[np.ndarray, list[Settlement]]:
    """Parse one seed's initial state into grid and settlements.

    Args:
        state_json: Server JSON with 'grid' and 'settlements' keys.

    Returns:
        Tuple of (InternalTerrain grid as np.ndarray, list of Settlement).

    Raises:
        ValueError: If grid data is missing or empty.
        KeyError: If required settlement fields are missing.
    """
    grid = _parse_grid(state_json["grid"])
    settlements = _parse_settlements(state_json.get("settlements", []))
    return grid, settlements


def load_round(
    round_json: dict[str, Any],
) -> list[tuple[np.ndarray, list[Settlement]]]:
    """Load all seeds' initial states from a round response.

    Args:
        round_json: Full round detail from GET /astar-island/rounds/{id}.

    Returns:
        List of (grid, settlements) tuples, one per seed.

    Raises:
        ValueError: If grid dimensions don't match declared size.
    """
    expected_w = round_json.get("map_width")
    expected_h = round_json.get("map_height")
    initial_states = round_json.get("initial_states", [])

    results: list[tuple[np.ndarray, list[Settlement]]] = []
    for idx, state in enumerate(initial_states):
        grid, settlements = load_initial_state(state)
        _validate_dimensions(grid, expected_w, expected_h, idx)
        results.append((grid, settlements))

    logger.info(
        "Loaded %d seed states (%dx%d)",
        len(results),
        expected_w or 0,
        expected_h or 0,
    )
    return results


def _parse_grid(grid_data: list[list[int]]) -> np.ndarray:
    """Convert server grid (list of lists) to InternalTerrain ndarray.

    Args:
        grid_data: Height x Width list of integer terrain codes.

    Returns:
        np.ndarray of shape (height, width) with InternalTerrain values.

    Raises:
        ValueError: If grid_data is empty or contains unknown codes.
    """
    if not grid_data or not grid_data[0]:
        msg = "Grid data is empty"
        raise ValueError(msg)

    height = len(grid_data)
    width = len(grid_data[0])
    grid = np.empty((height, width), dtype=np.int8)

    for row_idx, row in enumerate(grid_data):
        for col_idx, code in enumerate(row):
            terrain = _map_terrain_code(code)
            grid[row_idx, col_idx] = terrain

    return grid


def _map_terrain_code(code: int) -> InternalTerrain:
    """Map a single server terrain code to InternalTerrain.

    Raises:
        ValueError: If code is not recognized.
    """
    terrain = _SERVER_TERRAIN_MAP.get(code)
    if terrain is None:
        msg = f"Unknown server terrain code: {code}"
        raise ValueError(msg)
    return terrain


def _parse_settlements(
    settlements_data: list[dict[str, Any]],
) -> list[Settlement]:
    """Convert server settlement list to Settlement objects.

    Server settlements have at minimum: x, y, owner_id, is_port.
    Other fields get sensible defaults from constants.
    """
    settlements: list[Settlement] = []
    for sdata in settlements_data:
        settlement = Settlement(
            x=sdata["x"],
            y=sdata["y"],
            owner_id=sdata.get("owner_id", 0),
            population=sdata.get("population", INITIAL_POPULATION),
            food=sdata.get("food", INITIAL_FOOD),
            wealth=sdata.get("wealth", 0),
            defense=sdata.get("defense", 10),
            tech_level=sdata.get("tech_level", 1),
            is_port=sdata.get("is_port", False),
            has_longship=sdata.get("has_longship", False),
        )
        settlements.append(settlement)
    return settlements


def _validate_dimensions(
    grid: np.ndarray,
    expected_w: int | None,
    expected_h: int | None,
    seed_idx: int,
) -> None:
    """Check grid dimensions match the round's declared size.

    Raises:
        ValueError: If dimensions don't match.
    """
    if expected_h is not None and grid.shape[0] != expected_h:
        msg = (
            f"Seed {seed_idx}: grid height {grid.shape[0]} "
            f"!= expected {expected_h}"
        )
        raise ValueError(msg)
    if expected_w is not None and grid.shape[1] != expected_w:
        msg = (
            f"Seed {seed_idx}: grid width {grid.shape[1]} "
            f"!= expected {expected_w}"
        )
        raise ValueError(msg)
