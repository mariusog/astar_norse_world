"""Tests for runner -- simulation execution from external state."""

from __future__ import annotations

import numpy as np

from src.constants import NUM_PREDICTION_CLASSES, PROBABILITY_FLOOR
from src.runner import (
    run_monte_carlo_from_state,
    run_single_from_state,
)
from src.settlement import Settlement
from src.terrain import InternalTerrain


def _small_grid() -> np.ndarray:
    """Create a 5x5 grid with ocean border and plains interior."""
    grid = np.full((5, 5), InternalTerrain.OCEAN, dtype=np.int8)
    grid[1:4, 1:4] = InternalTerrain.PLAINS
    return grid


def _settlements() -> list[Settlement]:
    return [Settlement(x=2, y=2, owner_id=0)]


class TestRunSingleFromState:
    def test_returns_correct_shape(self) -> None:
        result = run_single_from_state(_small_grid(), _settlements(), sim_seed=42)
        assert result.shape == (5, 5)

    def test_returns_valid_classes(self) -> None:
        result = run_single_from_state(_small_grid(), _settlements(), sim_seed=42)
        assert result.min() >= 0
        assert result.max() < NUM_PREDICTION_CLASSES

    def test_does_not_mutate_inputs(self) -> None:
        grid = _small_grid()
        settlements = _settlements()
        grid_orig = grid.copy()
        pop_orig = settlements[0].population
        run_single_from_state(grid, settlements, sim_seed=42)
        np.testing.assert_array_equal(grid, grid_orig)
        assert settlements[0].population == pop_orig

    def test_deterministic_with_same_seed(self) -> None:
        r1 = run_single_from_state(_small_grid(), _settlements(), sim_seed=99)
        r2 = run_single_from_state(_small_grid(), _settlements(), sim_seed=99)
        np.testing.assert_array_equal(r1, r2)


class TestRunMonteCarloFromState:
    def test_returns_probability_tensor(self) -> None:
        probs = run_monte_carlo_from_state(
            _small_grid(),
            _settlements(),
            num_runs=5,
            base_sim_seed=0,
        )
        assert probs.shape == (5, 5, NUM_PREDICTION_CLASSES)

    def test_probabilities_sum_to_one(self) -> None:
        probs = run_monte_carlo_from_state(
            _small_grid(),
            _settlements(),
            num_runs=5,
            base_sim_seed=0,
        )
        np.testing.assert_allclose(probs.sum(axis=2), 1.0, atol=1e-10)

    def test_probability_floor_applied(self) -> None:
        probs = run_monte_carlo_from_state(
            _small_grid(),
            _settlements(),
            num_runs=5,
            base_sim_seed=0,
        )
        # After renormalization, values may be slightly below the floor
        assert probs.min() >= PROBABILITY_FLOOR * 0.9

    def test_does_not_mutate_inputs(self) -> None:
        grid = _small_grid()
        settlements = _settlements()
        grid_orig = grid.copy()
        run_monte_carlo_from_state(grid, settlements, num_runs=3)
        np.testing.assert_array_equal(grid, grid_orig)
