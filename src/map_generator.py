"""Procedural map generation from seed."""

from __future__ import annotations

import numpy as np

from src.constants import (
    DEFAULT_MAP_HEIGHT,
    DEFAULT_MAP_WIDTH,
    FJORD_LENGTH_RANGE,
    FOREST_PATCH_SIZE_RANGE,
    INITIAL_FOOD,
    INITIAL_POPULATION,
    MIN_SETTLEMENT_SPACING,
    MOUNTAIN_CHAIN_LENGTH_RANGE,
    MOUNTAIN_TURN_PROB,
    NUM_FJORDS_RANGE,
    NUM_FOREST_PATCHES,
    NUM_INITIAL_SETTLEMENTS_RANGE,
    NUM_MOUNTAIN_CHAINS,
    OCEAN_BORDER,
)
from src.settlement import Settlement
from src.terrain import InternalTerrain, neighbors_4


def generate_map(
    seed: int,
    width: int = DEFAULT_MAP_WIDTH,
    height: int = DEFAULT_MAP_HEIGHT,
) -> tuple[np.ndarray, list[Settlement]]:
    """Generate a full map and initial settlements from a seed.

    Returns:
        grid: height x width array of InternalTerrain values
        settlements: list of initial Settlement objects
    """
    rng = np.random.default_rng(seed)

    # Start with all plains
    grid = np.full((height, width), InternalTerrain.PLAINS, dtype=np.int8)

    # 1. Ocean border
    _place_ocean_border(grid, width, height)

    # 2. Fjords
    _place_fjords(grid, width, height, rng)

    # 3. Mountain chains
    _place_mountains(grid, width, height, rng)

    # 4. Forest patches
    _place_forests(grid, width, height, rng)

    # 5. Initial settlements
    settlements = _place_settlements(grid, width, height, rng)

    return grid, settlements


def _place_ocean_border(grid: np.ndarray, width: int, height: int) -> None:
    """Fill perimeter cells with ocean."""
    for b in range(OCEAN_BORDER):
        grid[b, :] = InternalTerrain.OCEAN
        grid[height - 1 - b, :] = InternalTerrain.OCEAN
        grid[:, b] = InternalTerrain.OCEAN
        grid[:, width - 1 - b] = InternalTerrain.OCEAN


def _place_fjords(grid: np.ndarray, width: int, height: int, rng: np.random.Generator) -> None:
    """Carve fjords inland from random edge positions."""
    num_fjords = rng.integers(NUM_FJORDS_RANGE[0], NUM_FJORDS_RANGE[1] + 1)

    for _ in range(num_fjords):
        # Pick a random edge
        edge = rng.integers(0, 4)  # 0=top, 1=bottom, 2=left, 3=right
        length = rng.integers(FJORD_LENGTH_RANGE[0], FJORD_LENGTH_RANGE[1] + 1)

        if edge == 0:
            x, y, dx, dy = int(rng.integers(2, width - 2)), 0, 0, 1
        elif edge == 1:
            x, y, dx, dy = int(rng.integers(2, width - 2)), height - 1, 0, -1
        elif edge == 2:
            x, y, dx, dy = 0, int(rng.integers(2, height - 2)), 1, 0
        else:
            x, y, dx, dy = width - 1, int(rng.integers(2, height - 2)), -1, 0

        for _step in range(length):
            if 0 <= y < height and 0 <= x < width:
                grid[y, x] = InternalTerrain.OCEAN
            x += dx
            y += dy
            # Slight random drift perpendicular to main direction
            if rng.random() < 0.3:
                if dx == 0:
                    x += int(rng.choice([-1, 1]))
                else:
                    y += int(rng.choice([-1, 1]))
            x = int(max(0, min(width - 1, x)))
            y = int(max(0, min(height - 1, y)))


def _place_mountains(grid: np.ndarray, width: int, height: int, rng: np.random.Generator) -> None:
    """Place mountain chains via random walks."""
    for _ in range(NUM_MOUNTAIN_CHAINS):
        x = int(rng.integers(3, width - 3))
        y = int(rng.integers(3, height - 3))
        dx, dy = int(rng.choice([-1, 0, 1])), int(rng.choice([-1, 0, 1]))
        if dx == 0 and dy == 0:
            dx = 1

        low, high = MOUNTAIN_CHAIN_LENGTH_RANGE
        length = int(rng.integers(low, high + 1))

        for _ in range(length):
            if (
                OCEAN_BORDER < y < height - OCEAN_BORDER - 1
                and OCEAN_BORDER < x < width - OCEAN_BORDER - 1
            ):
                grid[y, x] = InternalTerrain.MOUNTAIN
            x += dx
            y += dy
            # Random turn
            if rng.random() < MOUNTAIN_TURN_PROB:
                dx = int(rng.choice([-1, 0, 1]))
                dy = int(rng.choice([-1, 0, 1]))
                if dx == 0 and dy == 0:
                    dx = 1
            x = int(max(1, min(width - 2, x)))
            y = int(max(1, min(height - 2, y)))


def _place_forests(grid: np.ndarray, width: int, height: int, rng: np.random.Generator) -> None:
    """Place clustered forest patches."""
    for _ in range(NUM_FOREST_PATCHES):
        cx = int(rng.integers(2, width - 2))
        cy = int(rng.integers(2, height - 2))
        size = int(rng.integers(FOREST_PATCH_SIZE_RANGE[0], FOREST_PATCH_SIZE_RANGE[1] + 1))

        # BFS-like growth from center
        placed = 0
        frontier = [(cx, cy)]
        visited: set[tuple[int, int]] = set()

        while frontier and placed < size:
            idx = rng.integers(0, len(frontier))
            fx, fy = frontier.pop(int(idx))
            if (fx, fy) in visited:
                continue
            visited.add((fx, fy))

            if grid[fy, fx] == InternalTerrain.PLAINS:
                grid[fy, fx] = InternalTerrain.FOREST
                placed += 1
                for nx, ny in neighbors_4(fx, fy, width, height):
                    if (nx, ny) not in visited:
                        frontier.append((nx, ny))


def _is_land(grid: np.ndarray, x: int, y: int) -> bool:
    """Check if a cell is buildable land (plains or forest)."""
    t = InternalTerrain(grid[y, x])
    return t in (InternalTerrain.PLAINS, InternalTerrain.FOREST)


def _place_settlements(
    grid: np.ndarray, width: int, height: int, rng: np.random.Generator
) -> list[Settlement]:
    """Place initial settlements on land, spaced apart."""
    num = rng.integers(NUM_INITIAL_SETTLEMENTS_RANGE[0], NUM_INITIAL_SETTLEMENTS_RANGE[1] + 1)

    # Find all candidate positions (plains only -- don't destroy forests)
    candidates = []
    for y in range(height):
        for x in range(width):
            if grid[y, x] == InternalTerrain.PLAINS:
                candidates.append((x, y))

    rng.shuffle(candidates)  # type: ignore[arg-type]

    settlements: list[Settlement] = []
    faction_id = 0

    for cx, cy in candidates:
        if len(settlements) >= num:
            break
        # Check spacing
        too_close = False
        for s in settlements:
            dist = abs(s.x - cx) + abs(s.y - cy)
            if dist < MIN_SETTLEMENT_SPACING:
                too_close = True
                break
        if too_close:
            continue

        # Check if adjacent to ocean (potential port location)
        is_coastal = any(
            grid[ny, nx] == InternalTerrain.OCEAN for nx, ny in neighbors_4(cx, cy, width, height)
        )

        settlement = Settlement(
            x=cx,
            y=cy,
            owner_id=faction_id,
            population=INITIAL_POPULATION,
            food=INITIAL_FOOD,
            is_port=is_coastal,
        )
        settlements.append(settlement)
        grid[cy, cx] = InternalTerrain.PORT if is_coastal else InternalTerrain.SETTLEMENT
        faction_id += 1

    return settlements
