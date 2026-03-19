"""Tests for the simulation engine."""

import numpy as np

from src.map_generator import generate_map
from src.runner import grid_to_ascii, run_monte_carlo, run_single
from src.simulation import simulate
from src.terrain import InternalTerrain, Terrain


class TestSimulate:
    def test_returns_grid_and_settlements(self) -> None:
        grid, settlements = generate_map(seed=42, width=20, height=20)
        final_grid, survivors = simulate(grid, settlements, seed=100, years=10)
        assert final_grid.shape == (20, 20)
        assert isinstance(survivors, list)

    def test_deterministic_with_same_seed(self) -> None:
        g1, s1 = generate_map(seed=42, width=20, height=20)
        g2, s2 = generate_map(seed=42, width=20, height=20)
        fg1, _ = simulate(g1, s1, seed=100, years=20)
        fg2, _ = simulate(g2, s2, seed=100, years=20)
        np.testing.assert_array_equal(fg1, fg2)

    def test_different_sim_seeds_produce_different_results(self) -> None:
        g1, s1 = generate_map(seed=42, width=20, height=20)
        g2, s2 = generate_map(seed=42, width=20, height=20)
        fg1, _ = simulate(g1, s1, seed=100, years=20)
        fg2, _ = simulate(g2, s2, seed=200, years=20)
        # Stochastic -- very likely different
        assert not np.array_equal(fg1, fg2)

    def test_mountains_never_change(self) -> None:
        grid, settlements = generate_map(seed=42, width=20, height=20)
        mountains_before = (grid == InternalTerrain.MOUNTAIN).copy()
        simulate(grid, settlements, seed=100, years=50)
        mountains_after = grid == InternalTerrain.MOUNTAIN
        np.testing.assert_array_equal(mountains_before, mountains_after)

    def test_ocean_border_never_changes(self) -> None:
        grid, settlements = generate_map(seed=42, width=20, height=20)
        # Perimeter should stay ocean
        border_before = grid[0, :].copy()
        simulate(grid, settlements, seed=100, years=50)
        np.testing.assert_array_equal(border_before, grid[0, :])

    def test_simulation_changes_terrain(self) -> None:
        """Over 50 years, the map should change from initial state."""
        grid, settlements = generate_map(seed=42, width=30, height=30)
        initial = grid.copy()
        simulate(grid, settlements, seed=100, years=50)
        # Some cells must have changed (settlements expand, collapse, etc.)
        assert not np.array_equal(initial, grid)


class TestRunSingle:
    def test_returns_prediction_classes(self) -> None:
        result = run_single(map_seed=42, sim_seed=100, width=20, height=20)
        assert result.shape == (20, 20)
        assert np.all(result >= 0)
        assert np.all(result < Terrain.NUM_CLASSES)


class TestRunMonteCarlo:
    def test_returns_valid_probabilities(self) -> None:
        probs = run_monte_carlo(map_seed=42, num_runs=10, width=15, height=15)
        assert probs.shape == (15, 15, 6)
        # Each cell sums to ~1.0
        sums = probs.sum(axis=2)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)
        # No zeros (probability floor applied)
        assert np.all(probs > 0)


class TestGridToAscii:
    def test_renders_all_terrain_types(self) -> None:
        grid = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int8)
        result = grid_to_ascii(grid)
        assert ".SP" in result
        assert "RFM" in result
