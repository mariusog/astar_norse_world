"""Tests for procedural map generation."""

import numpy as np

from src.map_generator import generate_map
from src.terrain import InternalTerrain


class TestGenerateMap:
    def test_returns_correct_shape(self) -> None:
        grid, _settlements = generate_map(seed=42, width=20, height=20)
        assert grid.shape == (20, 20)

    def test_deterministic_with_same_seed(self) -> None:
        g1, s1 = generate_map(seed=42, width=20, height=20)
        g2, s2 = generate_map(seed=42, width=20, height=20)
        np.testing.assert_array_equal(g1, g2)
        assert len(s1) == len(s2)

    def test_different_seeds_produce_different_maps(self) -> None:
        g1, _ = generate_map(seed=42, width=20, height=20)
        g2, _ = generate_map(seed=99, width=20, height=20)
        assert not np.array_equal(g1, g2)

    def test_ocean_border_exists(self) -> None:
        grid, _ = generate_map(seed=42, width=20, height=20)
        # Top and bottom rows should be ocean
        assert np.all(grid[0, :] == InternalTerrain.OCEAN)
        assert np.all(grid[-1, :] == InternalTerrain.OCEAN)
        # Left and right columns should be ocean
        assert np.all(grid[:, 0] == InternalTerrain.OCEAN)
        assert np.all(grid[:, -1] == InternalTerrain.OCEAN)

    def test_settlements_placed(self) -> None:
        _, settlements = generate_map(seed=42, width=20, height=20)
        assert len(settlements) >= 1

    def test_settlements_on_valid_terrain(self) -> None:
        grid, settlements = generate_map(seed=42, width=20, height=20)
        for s in settlements:
            terrain = InternalTerrain(grid[s.y, s.x])
            assert terrain in (InternalTerrain.SETTLEMENT, InternalTerrain.PORT)

    def test_contains_mountains(self) -> None:
        grid, _ = generate_map(seed=42, width=30, height=30)
        assert np.any(grid == InternalTerrain.MOUNTAIN)

    def test_contains_forests(self) -> None:
        grid, _ = generate_map(seed=42, width=30, height=30)
        assert np.any(grid == InternalTerrain.FOREST)
