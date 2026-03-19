"""Tests for terrain types and grid utilities."""

import numpy as np

from src.terrain import InternalTerrain, Terrain, grid_to_prediction, neighbors_4, neighbors_8


class TestTerrainMapping:
    def test_ocean_maps_to_empty(self) -> None:
        assert InternalTerrain.OCEAN.to_prediction_class() == Terrain.EMPTY

    def test_plains_maps_to_empty(self) -> None:
        assert InternalTerrain.PLAINS.to_prediction_class() == Terrain.EMPTY

    def test_settlement_maps_to_settlement(self) -> None:
        assert InternalTerrain.SETTLEMENT.to_prediction_class() == Terrain.SETTLEMENT

    def test_port_maps_to_port(self) -> None:
        assert InternalTerrain.PORT.to_prediction_class() == Terrain.PORT

    def test_ruin_maps_to_ruin(self) -> None:
        assert InternalTerrain.RUIN.to_prediction_class() == Terrain.RUIN

    def test_forest_maps_to_forest(self) -> None:
        assert InternalTerrain.FOREST.to_prediction_class() == Terrain.FOREST

    def test_mountain_maps_to_mountain(self) -> None:
        assert InternalTerrain.MOUNTAIN.to_prediction_class() == Terrain.MOUNTAIN


class TestNeighbors:
    def test_neighbors_4_center(self) -> None:
        result = neighbors_4(5, 5, 10, 10)
        assert len(result) == 4
        assert (4, 5) in result
        assert (6, 5) in result
        assert (5, 4) in result
        assert (5, 6) in result

    def test_neighbors_4_corner(self) -> None:
        result = neighbors_4(0, 0, 10, 10)
        assert len(result) == 2
        assert (1, 0) in result
        assert (0, 1) in result

    def test_neighbors_8_center(self) -> None:
        result = neighbors_8(5, 5, 10, 10)
        assert len(result) == 8

    def test_neighbors_8_corner(self) -> None:
        result = neighbors_8(0, 0, 10, 10)
        assert len(result) == 3


class TestGridToPrediction:
    def test_converts_internal_to_prediction(self) -> None:
        grid = np.array(
            [
                [InternalTerrain.OCEAN, InternalTerrain.PLAINS],
                [InternalTerrain.MOUNTAIN, InternalTerrain.FOREST],
            ],
            dtype=np.int8,
        )
        result = grid_to_prediction(grid)
        assert result[0, 0] == Terrain.EMPTY
        assert result[0, 1] == Terrain.EMPTY
        assert result[1, 0] == Terrain.MOUNTAIN
        assert result[1, 1] == Terrain.FOREST
