"""Top-level simulation runner -- run multiple seeds and collect results."""

from __future__ import annotations

import copy

import numpy as np

from src.constants import (
    DEFAULT_MAP_HEIGHT,
    DEFAULT_MAP_WIDTH,
    NUM_PREDICTION_CLASSES,
    PROBABILITY_FLOOR,
)
from src.map_generator import generate_map
from src.settlement import Settlement
from src.simulation import simulate
from src.terrain import grid_to_prediction


def run_single(
    map_seed: int,
    sim_seed: int,
    width: int = DEFAULT_MAP_WIDTH,
    height: int = DEFAULT_MAP_HEIGHT,
) -> np.ndarray:
    """Run one simulation and return the final prediction-class grid.

    Args:
        map_seed: seed for map generation (same across runs for one round)
        sim_seed: seed for simulation stochasticity (varies per run)

    Returns:
        height x width array of Terrain class indices (0-5)
    """
    grid, settlements = generate_map(map_seed, width, height)
    grid, _ = simulate(grid, settlements, sim_seed)
    return grid_to_prediction(grid)


def run_monte_carlo(
    map_seed: int,
    num_runs: int = 100,
    width: int = DEFAULT_MAP_WIDTH,
    height: int = DEFAULT_MAP_HEIGHT,
    base_sim_seed: int = 0,
) -> np.ndarray:
    """Run many simulations and produce a probability distribution.

    Returns:
        height x width x 6 probability tensor
    """
    counts = np.zeros((height, width, NUM_PREDICTION_CLASSES), dtype=np.float64)

    for i in range(num_runs):
        result = run_single(map_seed, base_sim_seed + i, width, height)
        for cls in range(NUM_PREDICTION_CLASSES):
            counts[:, :, cls] += (result == cls).astype(np.float64)

    # Normalize to probabilities
    probs = counts / num_runs

    # Apply probability floor to avoid infinite KL divergence
    probs = np.maximum(probs, PROBABILITY_FLOOR)
    probs = probs / probs.sum(axis=2, keepdims=True)

    return probs


def run_single_from_state(
    grid: np.ndarray,
    settlements: list[Settlement],
    sim_seed: int,
) -> np.ndarray:
    """Run one simulation from a given initial state.

    Deep copies inputs so simulate() doesn't mutate originals.

    Returns:
        H x W array of terrain class indices (0-5).
    """
    grid_copy = grid.copy()
    settlements_copy = copy.deepcopy(settlements)
    grid_copy, _ = simulate(grid_copy, settlements_copy, sim_seed)
    return grid_to_prediction(grid_copy)


def run_monte_carlo_from_state(
    grid: np.ndarray,
    settlements: list[Settlement],
    num_runs: int = 100,
    base_sim_seed: int = 0,
) -> np.ndarray:
    """Run MC simulations from a given initial state.

    Returns:
        H x W x 6 probability tensor.
    """
    h, w = grid.shape
    counts = np.zeros((h, w, NUM_PREDICTION_CLASSES), dtype=np.float64)

    for i in range(num_runs):
        result = run_single_from_state(
            grid,
            settlements,
            base_sim_seed + i,
        )
        for cls in range(NUM_PREDICTION_CLASSES):
            counts[:, :, cls] += (result == cls).astype(np.float64)

    probs = counts / num_runs
    probs = np.maximum(probs, PROBABILITY_FLOOR)
    probs = probs / probs.sum(axis=2, keepdims=True)
    return probs


def grid_to_ascii(grid: np.ndarray) -> str:
    """Render a prediction-class grid as ASCII for debugging.

    Symbols: . = empty, S = settlement, P = port, R = ruin, F = forest, M = mountain
    """
    symbols = {0: ".", 1: "S", 2: "P", 3: "R", 4: "F", 5: "M"}
    lines = []
    for row in grid:
        lines.append("".join(symbols.get(int(cell), "?") for cell in row))
    return "\n".join(lines)
