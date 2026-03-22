"""Norse world simulation engine -- 50-year lifecycle with 5 phases per year."""

from __future__ import annotations

import numpy as np

from src.constants import SIMULATION_YEARS
from src.settlement import Settlement
from src.sim_phases import (
    _phase_conflict,
    _phase_environment,
    _phase_growth,
    _phase_trade,
    _phase_winter,
    _reset_yearly,
)


def simulate(
    grid: np.ndarray,
    settlements: list[Settlement],
    seed: int,
    years: int = SIMULATION_YEARS,
) -> tuple[np.ndarray, list[Settlement]]:
    """Run the full simulation for the given number of years.

    Returns new grid and surviving settlements. Inputs are not mutated.

    Args:
        grid: height x width InternalTerrain grid
        settlements: list of settlements
        seed: random seed for stochastic decisions
        years: number of years to simulate

    Returns:
        final grid and surviving settlements
    """
    grid = grid.copy()
    settlements = [s.copy() for s in settlements]
    rng = np.random.default_rng(seed)
    height, width = grid.shape

    for _ in range(years):
        _reset_yearly(settlements)
        _phase_growth(grid, settlements, width, height, rng)
        _phase_conflict(grid, settlements, width, height, rng)
        _phase_trade(settlements, rng)
        _phase_winter(grid, settlements, width, height, rng)
        _phase_environment(grid, settlements, width, height, rng)

    return grid, [s for s in settlements if s.alive]
