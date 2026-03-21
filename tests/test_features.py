"""Tests for spatial feature extraction."""

import numpy as np
import pytest

from src.features import (
    compute_coastal_mask,
    compute_feature_grid,
    compute_forest_density,
    compute_ocean_distance,
    compute_settlement_distance,
)
from src.terrain import InternalTerrain


@pytest.fixture
def simple_grid() -> np.ndarray:
    """5x5 grid: ocean border, settlement at (2,2), forest at (3,2)."""
    grid = np.full((5, 5), InternalTerrain.PLAINS, dtype=np.int8)
    grid[0, :] = InternalTerrain.OCEAN
    grid[-1, :] = InternalTerrain.OCEAN
    grid[:, 0] = InternalTerrain.OCEAN
    grid[:, -1] = InternalTerrain.OCEAN
    grid[2, 2] = InternalTerrain.SETTLEMENT
    grid[2, 3] = InternalTerrain.FOREST
    return grid


class TestComputeSettlementDistance:
    """Tests for compute_settlement_distance."""

    def test_settlement_cell_has_zero_distance(self, simple_grid: np.ndarray) -> None:
        dist = compute_settlement_distance(simple_grid)
        assert dist[2, 2] == 0

    def test_adjacent_cell_has_distance_one(self, simple_grid: np.ndarray) -> None:
        dist = compute_settlement_distance(simple_grid)
        # Cell (2,1) is one step from settlement at (2,2)
        assert dist[2, 1] == 1

    def test_distance_increases_away_from_settlement(self, simple_grid: np.ndarray) -> None:
        dist = compute_settlement_distance(simple_grid)
        assert dist[1, 1] == 2  # Manhattan dist from (2,2)

    def test_no_settlements_returns_max_distance(self) -> None:
        grid = np.full((4, 4), InternalTerrain.PLAINS, dtype=np.int8)
        dist = compute_settlement_distance(grid)
        assert np.all(dist == 8)  # width + height = 4 + 4

    def test_port_counts_as_settlement(self) -> None:
        grid = np.full((3, 3), InternalTerrain.PLAINS, dtype=np.int8)
        grid[1, 1] = InternalTerrain.PORT
        dist = compute_settlement_distance(grid)
        assert dist[1, 1] == 0
        assert dist[0, 1] == 1


class TestComputeCoastalMask:
    """Tests for compute_coastal_mask."""

    def test_land_adjacent_to_ocean_is_coastal(self, simple_grid: np.ndarray) -> None:
        mask = compute_coastal_mask(simple_grid)
        # (1,1) is plains adjacent to ocean at (0,1) and (1,0)
        assert mask[1, 1] is np.True_

    def test_ocean_cells_are_not_coastal(self, simple_grid: np.ndarray) -> None:
        mask = compute_coastal_mask(simple_grid)
        assert mask[0, 0] is np.False_

    def test_interior_land_is_not_coastal(self) -> None:
        grid = np.full((7, 7), InternalTerrain.PLAINS, dtype=np.int8)
        grid[0, :] = InternalTerrain.OCEAN
        grid[-1, :] = InternalTerrain.OCEAN
        grid[:, 0] = InternalTerrain.OCEAN
        grid[:, -1] = InternalTerrain.OCEAN
        mask = compute_coastal_mask(grid)
        # Center cell (3,3) is far from ocean
        assert mask[3, 3] is np.False_

    def test_all_ocean_returns_empty_mask(self) -> None:
        grid = np.full((3, 3), InternalTerrain.OCEAN, dtype=np.int8)
        mask = compute_coastal_mask(grid)
        assert not mask.any()


class TestComputeOceanDistance:
    """Tests for compute_ocean_distance."""

    def test_ocean_cell_has_zero_distance(self, simple_grid: np.ndarray) -> None:
        dist = compute_ocean_distance(simple_grid)
        assert dist[0, 0] == 0

    def test_center_cell_has_correct_distance(self, simple_grid: np.ndarray) -> None:
        dist = compute_ocean_distance(simple_grid)
        # (2,2) is 2 steps from nearest ocean border
        assert dist[2, 2] == 2

    def test_no_ocean_returns_max_distance(self) -> None:
        grid = np.full((3, 3), InternalTerrain.PLAINS, dtype=np.int8)
        dist = compute_ocean_distance(grid)
        assert np.all(dist == 6)  # 3 + 3


class TestComputeForestDensity:
    """Tests for compute_forest_density."""

    def test_forest_cell_counted_in_own_density(self) -> None:
        grid = np.full((5, 5), InternalTerrain.PLAINS, dtype=np.int8)
        grid[2, 2] = InternalTerrain.FOREST
        density = compute_forest_density(grid, radius=1)
        assert density[2, 2] >= 1

    def test_no_forest_returns_zeros(self) -> None:
        grid = np.full((5, 5), InternalTerrain.PLAINS, dtype=np.int8)
        density = compute_forest_density(grid, radius=3)
        assert np.all(density == 0)

    def test_cluster_has_higher_density(self) -> None:
        grid = np.full((7, 7), InternalTerrain.PLAINS, dtype=np.int8)
        grid[2, 2] = InternalTerrain.FOREST
        grid[2, 3] = InternalTerrain.FOREST
        grid[3, 2] = InternalTerrain.FOREST
        density = compute_forest_density(grid, radius=2)
        # Center of cluster should have higher density
        assert density[2, 2] >= 3


class TestComputeFeatureGrid:
    """Tests for compute_feature_grid."""

    def test_returns_all_expected_keys(self, simple_grid: np.ndarray) -> None:
        features = compute_feature_grid(simple_grid)
        expected_keys = {
            "settlement_dist",
            "settlement_density",
            "coastal_mask",
            "ocean_dist",
            "forest_density",
        }
        assert set(features.keys()) == expected_keys

    def test_all_features_match_grid_shape(self, simple_grid: np.ndarray) -> None:
        features = compute_feature_grid(simple_grid)
        h, w = simple_grid.shape
        for name, arr in features.items():
            assert arr.shape[:2] == (h, w), f"{name} shape mismatch"
