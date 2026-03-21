"""Tests for terrain types and grid utilities."""

import numpy as np
import pytest

from src.terrain import (
    InternalTerrain,
    Terrain,
    grid_to_prediction,
    map_server_codes,
    neighbors_4,
    neighbors_8,
)


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


class TestMapServerCodes:
    def test_identity_0_through_5(self) -> None:
        """Server codes 0-5 map to prediction classes 0-5."""
        patch = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        result = map_server_codes(patch)
        expected = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)

    def test_ocean_plains_map_to_empty(self) -> None:
        """Server codes 10 (ocean) and 11 (plains) map to class 0."""
        patch = np.array([[10, 11]], dtype=np.int32)
        result = map_server_codes(patch)
        np.testing.assert_array_equal(result, np.array([[0, 0]], dtype=np.int8))

    def test_mixed_codes(self) -> None:
        """Mixed server codes are mapped correctly."""
        patch = np.array([[0, 10, 5], [11, 3, 1]], dtype=np.int32)
        result = map_server_codes(patch)
        expected = np.array([[0, 0, 5], [0, 3, 1]], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)

    def test_invalid_code_raises(self) -> None:
        """Unmapped code 7 raises ValueError."""
        patch = np.array([[7]], dtype=np.int32)
        with pytest.raises(ValueError, match="Unmapped server codes"):
            map_server_codes(patch)

    def test_negative_code_raises(self) -> None:
        """Negative code raises ValueError."""
        patch = np.array([[-1]], dtype=np.int32)
        with pytest.raises(ValueError, match="out of range"):
            map_server_codes(patch)

    def test_out_of_range_code_raises(self) -> None:
        """Code beyond max known server code raises ValueError."""
        patch = np.array([[99]], dtype=np.int32)
        with pytest.raises(ValueError, match="out of range"):
            map_server_codes(patch)


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


class TestServerGridToInternal:
    def test_maps_server_codes_correctly(self) -> None:
        from src.terrain import server_grid_to_internal

        grid_data = [[10, 11, 0], [1, 4, 5]]
        result = server_grid_to_internal(grid_data)
        assert result[0, 0] == InternalTerrain.OCEAN
        assert result[0, 1] == InternalTerrain.PLAINS
        assert result[0, 2] == InternalTerrain.PLAINS  # server 0 = Empty -> Plains
        assert result[1, 0] == InternalTerrain.SETTLEMENT
        assert result[1, 1] == InternalTerrain.FOREST
        assert result[1, 2] == InternalTerrain.MOUNTAIN
