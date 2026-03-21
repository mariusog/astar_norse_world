"""Tests for feature-based per-cell predictor."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.constants import (
    DIST_BIN_EDGES,
    NUM_PREDICTION_CLASSES,
    SETTLEMENT_DENSITY_MAX_BIN,
)
from src.feature_predictor import (
    FeatureLookup,
    _apply_static_overrides,
    _digitize_density,
    _digitize_distances,
    _floor_and_normalize,
    _lookup_with_fallback,
    build_feature_lookup,
    predict_from_features,
)
from src.features import compute_settlement_density
from src.terrain import InternalTerrain


@pytest.fixture
def simple_grid() -> np.ndarray:
    """5x5 grid with mixed terrain."""
    grid = np.full((5, 5), InternalTerrain.PLAINS, dtype=np.int8)
    grid[0, :] = InternalTerrain.OCEAN
    grid[2, 2] = InternalTerrain.SETTLEMENT
    grid[4, 4] = InternalTerrain.MOUNTAIN
    grid[3, 1] = InternalTerrain.FOREST
    return grid


@pytest.fixture
def simple_lookup() -> FeatureLookup:
    """Minimal lookup for testing."""
    return {
        (1, 0, 0): np.array([0.5, 0.2, 0.0, 0.1, 0.2, 0.0]),
        (1, 1, 0): np.array([0.4, 0.3, 0.0, 0.1, 0.2, 0.0]),
        (1, 2, 0): np.array([0.6, 0.1, 0.0, 0.1, 0.2, 0.0]),
        (5, 1, 0): np.array([0.1, 0.0, 0.0, 0.0, 0.9, 0.0]),
    }


@pytest.fixture
def mock_data_dir(tmp_path: Path) -> Path:
    """Create a mock data directory with one round."""
    round_dir = tmp_path / "round1"
    round_dir.mkdir()
    grid = np.full((5, 5), InternalTerrain.PLAINS, dtype=np.int8)
    grid[0, :] = InternalTerrain.OCEAN
    grid[2, 2] = InternalTerrain.SETTLEMENT
    gt = np.full((5, 5, 6), 1.0 / 6, dtype=np.float64)
    gt[0, :] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # ocean
    gt[2, 2] = [0.0, 0.8, 0.0, 0.1, 0.1, 0.0]  # settlement
    seed_dir = round_dir / "seed_0"
    seed_dir.mkdir()
    np.save(seed_dir / "ground_truth.npy", gt)
    np.save(seed_dir / "initial_grid.npy", grid)
    rd = {"round_number": 1, "initial_states": [{"grid": grid.tolist()}]}
    with open(round_dir / "round.json", "w") as f:
        json.dump(rd, f)
    return tmp_path


class TestComputeDensityMap:
    """Tests for settlement density computation."""

    def test_empty_grid_zero_density(self) -> None:
        grid = np.full((10, 10), InternalTerrain.PLAINS, dtype=np.int8)
        density = compute_settlement_density(grid)
        assert density.shape == (10, 10)
        assert np.all(density == 0)

    def test_single_settlement_nonzero_density(self) -> None:
        grid = np.full((15, 15), InternalTerrain.PLAINS, dtype=np.int8)
        grid[7, 7] = InternalTerrain.SETTLEMENT
        density = compute_settlement_density(grid)
        # Cell at (7,7) is fully inside filter window, density >= 1
        assert density[7, 7] >= 1
        assert density[0, 0] == 0  # far corner

    def test_port_counted_as_settlement(self) -> None:
        grid = np.full((15, 15), InternalTerrain.PLAINS, dtype=np.int8)
        grid[7, 7] = InternalTerrain.PORT
        density = compute_settlement_density(grid)
        assert density[7, 7] >= 1


class TestDigitizeDistances:
    """Tests for distance binning."""

    def test_zero_distance_bin_zero(self) -> None:
        dist = np.array([[0]], dtype=np.int32)
        result = _digitize_distances(dist)
        assert result[0, 0] == 0

    def test_large_distance_max_bin(self) -> None:
        dist = np.array([[100]], dtype=np.int32)
        result = _digitize_distances(dist)
        # np.digitize with edges [1,2,3,4,5,7,10,15,999] gives max bin 8
        assert result[0, 0] == len(DIST_BIN_EDGES) - 2

    def test_bin_edges_monotonic(self) -> None:
        dists = np.array([DIST_BIN_EDGES], dtype=np.int32)
        bins = _digitize_distances(dists)
        assert np.all(np.diff(bins[0]) >= 0)


class TestDigitizeDensity:
    """Tests for density binning."""

    def test_caps_at_max(self) -> None:
        density = np.array([[10]], dtype=np.int32)
        result = _digitize_density(density)
        assert result[0, 0] == SETTLEMENT_DENSITY_MAX_BIN

    def test_preserves_low_values(self) -> None:
        density = np.array([[0, 1, 2]], dtype=np.int32)
        result = _digitize_density(density)
        np.testing.assert_array_equal(result[0], [0, 1, 2])


class TestLookupWithFallback:
    """Tests for cascading fallback lookup."""

    def test_exact_match(self, simple_lookup: FeatureLookup) -> None:
        vec = _lookup_with_fallback(1, 0, 0, simple_lookup)
        assert vec is not None
        np.testing.assert_array_almost_equal(vec, [0.5, 0.2, 0.0, 0.1, 0.2, 0.0])

    def test_fallback_aggregates_density(self, simple_lookup: FeatureLookup) -> None:
        vec = _lookup_with_fallback(1, 0, 99, simple_lookup)
        assert vec is not None
        # Should aggregate over density bins for terrain=1, dist=0
        np.testing.assert_array_almost_equal(vec, [0.5, 0.2, 0.0, 0.1, 0.2, 0.0])

    def test_fallback_aggregates_dist(self, simple_lookup: FeatureLookup) -> None:
        vec = _lookup_with_fallback(1, 99, 99, simple_lookup)
        assert vec is not None
        # Should average over all bins for terrain=1
        assert vec.sum() == pytest.approx(1.0, abs=0.01)

    def test_unknown_terrain_returns_none(self, simple_lookup: FeatureLookup) -> None:
        vec = _lookup_with_fallback(99, 0, 0, simple_lookup)
        assert vec is None


class TestStaticOverrides:
    """Tests for ocean and mountain overrides."""

    def test_ocean_override(self, simple_grid: np.ndarray) -> None:
        tensor = np.full((5, 5, 6), 1.0 / 6)
        _apply_static_overrides(tensor, simple_grid)
        np.testing.assert_array_equal(tensor[0, 0], [1, 0, 0, 0, 0, 0])

    def test_mountain_override(self, simple_grid: np.ndarray) -> None:
        tensor = np.full((5, 5, 6), 1.0 / 6)
        _apply_static_overrides(tensor, simple_grid)
        np.testing.assert_array_equal(tensor[4, 4], [0, 0, 0, 0, 0, 1])


class TestFloorAndNormalize:
    """Tests for probability floor and normalization."""

    def test_no_zeros_after_floor(self) -> None:
        tensor = np.array([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        _floor_and_normalize(tensor)
        # After floor + renormalize, minimum is floor/sum ~ 0.01/1.05
        assert np.all(tensor > 0.0)
        assert tensor[0, 0].min() > 0.005

    def test_sums_to_one(self) -> None:
        tensor = np.array([[[0.5, 0.3, 0.1, 0.05, 0.03, 0.02]]])
        _floor_and_normalize(tensor)
        assert tensor[0, 0].sum() == pytest.approx(1.0, abs=1e-10)


class TestBuildFeatureLookup:
    """Tests for building lookup from historical data."""

    def test_builds_from_mock_data(self, mock_data_dir: Path) -> None:
        lookup = build_feature_lookup(mock_data_dir)
        assert len(lookup) > 0
        for key, vec in lookup.items():
            assert len(key) == 3
            assert vec.shape == (NUM_PREDICTION_CLASSES,)

    def test_regime_weights_applied(self, mock_data_dir: Path) -> None:
        lookup_1x = build_feature_lookup(mock_data_dir, {1: 1.0})
        lookup_2x = build_feature_lookup(mock_data_dir, {1: 2.0})
        # Same keys, same values (only one round so weights scale uniformly)
        assert set(lookup_1x.keys()) == set(lookup_2x.keys())
        for key in lookup_1x:
            np.testing.assert_array_almost_equal(
                lookup_1x[key],
                lookup_2x[key],
                decimal=5,
            )

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        lookup = build_feature_lookup(tmp_path)
        assert len(lookup) == 0


class TestPredictFromFeatures:
    """Tests for end-to-end prediction."""

    def test_output_shape(
        self,
        simple_grid: np.ndarray,
        simple_lookup: FeatureLookup,
    ) -> None:
        tensor = predict_from_features(simple_grid, simple_lookup)
        assert tensor.shape == (5, 5, NUM_PREDICTION_CLASSES)

    def test_probabilities_sum_to_one(
        self,
        simple_grid: np.ndarray,
        simple_lookup: FeatureLookup,
    ) -> None:
        tensor = predict_from_features(simple_grid, simple_lookup)
        sums = tensor.sum(axis=2)
        np.testing.assert_array_almost_equal(sums, np.ones((5, 5)))

    def test_ocean_cells_deterministic(
        self,
        simple_grid: np.ndarray,
        simple_lookup: FeatureLookup,
    ) -> None:
        tensor = predict_from_features(simple_grid, simple_lookup)
        # Ocean row should have high prob for class 0
        assert tensor[0, 0, 0] > 0.9

    def test_mountain_cells_deterministic(
        self,
        simple_grid: np.ndarray,
        simple_lookup: FeatureLookup,
    ) -> None:
        tensor = predict_from_features(simple_grid, simple_lookup)
        assert tensor[4, 4, 5] > 0.9

    def test_no_zeros_in_output(
        self,
        simple_grid: np.ndarray,
        simple_lookup: FeatureLookup,
    ) -> None:
        tensor = predict_from_features(simple_grid, simple_lookup)
        # All cells should have positive probabilities after floor
        assert np.all(tensor[1, 1] > 0.0)
