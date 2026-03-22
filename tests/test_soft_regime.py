"""Tests for soft regime blending module."""

import numpy as np
import pytest

from src.constants import NUM_PREDICTION_CLASSES
from src.observation import ObservationStore
from src.soft_regime import (
    build_regime_predictions,
    estimate_regime_confidence,
    score_soft_blend,
    soft_blend_predictions,
)
from src.terrain import InternalTerrain, Terrain

C = NUM_PREDICTION_CLASSES


@pytest.fixture
def small_grid() -> np.ndarray:
    """10x10 grid with settlements, ports, and plains."""
    grid = np.full((10, 10), int(InternalTerrain.PLAINS), dtype=np.int8)
    grid[0, :] = int(InternalTerrain.OCEAN)
    grid[:, 0] = int(InternalTerrain.OCEAN)
    grid[3, 3] = int(InternalTerrain.SETTLEMENT)
    grid[3, 4] = int(InternalTerrain.SETTLEMENT)
    grid[5, 5] = int(InternalTerrain.PORT)
    return grid


@pytest.fixture
def obs_store_survive(small_grid: np.ndarray) -> ObservationStore:
    """Survive regime: settlements persist."""
    h, w = small_grid.shape
    store = ObservationStore(height=h, width=w, num_seeds=1)
    store.add_observation(0, 3, 3, np.array([[1, 1]], dtype=np.int32))
    store.add_observation(0, 5, 5, np.array([[2]], dtype=np.int32))
    return store


@pytest.fixture
def obs_store_collapse(small_grid: np.ndarray) -> ObservationStore:
    """Collapse regime: settlements gone."""
    h, w = small_grid.shape
    store = ObservationStore(height=h, width=w, num_seeds=1)
    store.add_observation(0, 3, 3, np.array([[3, 3]], dtype=np.int32))
    store.add_observation(0, 5, 5, np.array([[0]], dtype=np.int32))
    return store


def _make_priors(terrain: InternalTerrain, target: Terrain, prob: float) -> np.ndarray:
    """Build priors with one terrain type having high prob for target class."""
    priors = np.full((7, C), 0.02, dtype=np.float64)
    priors[int(terrain), int(target)] = prob
    priors[int(terrain)] /= priors[int(terrain)].sum()
    return priors


@pytest.fixture
def survive_priors() -> np.ndarray:
    """Priors favoring settlement survival."""
    p = np.full((7, C), 0.02, dtype=np.float64)
    p[int(InternalTerrain.SETTLEMENT), int(Terrain.SETTLEMENT)] = 0.80
    p[int(InternalTerrain.SETTLEMENT)] /= p[int(InternalTerrain.SETTLEMENT)].sum()
    p[int(InternalTerrain.PORT), int(Terrain.PORT)] = 0.70
    p[int(InternalTerrain.PORT)] /= p[int(InternalTerrain.PORT)].sum()
    return p


@pytest.fixture
def collapse_priors() -> np.ndarray:
    """Priors favoring settlement collapse to ruin."""
    p = np.full((7, C), 0.02, dtype=np.float64)
    p[int(InternalTerrain.SETTLEMENT), int(Terrain.RUIN)] = 0.80
    p[int(InternalTerrain.SETTLEMENT)] /= p[int(InternalTerrain.SETTLEMENT)].sum()
    p[int(InternalTerrain.PORT), int(Terrain.EMPTY)] = 0.70
    p[int(InternalTerrain.PORT)] /= p[int(InternalTerrain.PORT)].sum()
    return p


class TestEstimateRegimeConfidence:
    """Tests for estimate_regime_confidence."""

    def test_survive_gives_high_confidence(
        self,
        small_grid: np.ndarray,
        obs_store_survive: ObservationStore,
    ) -> None:
        confidence = estimate_regime_confidence([small_grid], [obs_store_survive])
        assert confidence >= 0.8

    def test_collapse_gives_low_confidence(
        self,
        small_grid: np.ndarray,
        obs_store_collapse: ObservationStore,
    ) -> None:
        confidence = estimate_regime_confidence([small_grid], [obs_store_collapse])
        assert confidence <= 0.2

    def test_no_observations_returns_default(self) -> None:
        grid = np.full((5, 5), int(InternalTerrain.PLAINS), dtype=np.int8)
        store = ObservationStore(height=5, width=5, num_seeds=1)
        assert estimate_regime_confidence([grid], [store]) == pytest.approx(0.6)

    def test_no_settlement_cells_returns_default(self) -> None:
        grid = np.full((5, 5), int(InternalTerrain.FOREST), dtype=np.int8)
        store = ObservationStore(height=5, width=5, num_seeds=1)
        store.add_observation(0, 2, 2, np.array([[4]], dtype=np.int32))
        assert estimate_regime_confidence([grid], [store]) == pytest.approx(0.6)

    def test_confidence_bounded_zero_to_one(
        self,
        small_grid: np.ndarray,
        obs_store_survive: ObservationStore,
    ) -> None:
        confidence = estimate_regime_confidence([small_grid], [obs_store_survive])
        assert 0.0 <= confidence <= 1.0

    def test_multiple_seeds_aggregated(self) -> None:
        grid = np.full((5, 5), int(InternalTerrain.SETTLEMENT), dtype=np.int8)
        store1 = ObservationStore(height=5, width=5, num_seeds=1)
        store2 = ObservationStore(height=5, width=5, num_seeds=1)
        store1.add_observation(0, 0, 2, np.array([[1, 3, 3, 3, 3]], dtype=np.int32))
        store2.add_observation(0, 0, 2, np.array([[3, 3, 3, 3, 3]], dtype=np.int32))
        confidence = estimate_regime_confidence([grid, grid], [store1, store2])
        assert 0.1 < confidence < 0.9


class TestSoftBlendPredictions:
    """Tests for soft_blend_predictions."""

    def test_full_confidence_returns_survive(self) -> None:
        rng = np.random.RandomState(42)
        survive = rng.dirichlet(np.ones(C), (5, 5))
        collapse = rng.dirichlet(np.ones(C), (5, 5))
        result = soft_blend_predictions(survive, collapse, 1.0, np.zeros((5, 5), dtype=bool))
        np.testing.assert_array_almost_equal(result, survive)

    def test_zero_confidence_returns_collapse(self) -> None:
        rng_s, rng_c = np.random.RandomState(42), np.random.RandomState(99)
        survive = rng_s.dirichlet(np.ones(C), (5, 5))
        collapse = rng_c.dirichlet(np.ones(C), (5, 5))
        result = soft_blend_predictions(survive, collapse, 0.0, np.zeros((5, 5), dtype=bool))
        np.testing.assert_array_almost_equal(result, collapse)

    def test_observed_cells_keep_survive(self) -> None:
        rng = np.random.RandomState(42)
        survive = rng.dirichlet(np.ones(C), (5, 5))
        collapse = rng.dirichlet(np.ones(C), (5, 5))
        result = soft_blend_predictions(survive, collapse, 0.0, np.ones((5, 5), dtype=bool))
        np.testing.assert_array_almost_equal(result, survive)

    def test_half_confidence_blends_equally(self) -> None:
        survive = np.full((5, 5, C), 0.2)
        collapse = np.full((5, 5, C), 0.1)
        survive[:, :, 0], collapse[:, :, 0] = 0.8, 0.3
        result = soft_blend_predictions(survive, collapse, 0.5, np.zeros((5, 5), dtype=bool))
        np.testing.assert_array_almost_equal(result, 0.5 * survive + 0.5 * collapse)

    def test_mixed_observed_unobserved(self) -> None:
        rng = np.random.RandomState(7)
        survive = rng.dirichlet(np.ones(C), (4, 4))
        collapse = rng.dirichlet(np.ones(C), (4, 4))
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, :] = True
        result = soft_blend_predictions(survive, collapse, 0.3, mask)
        np.testing.assert_array_almost_equal(result[0], survive[0])
        np.testing.assert_array_almost_equal(result[1:], 0.3 * survive[1:] + 0.7 * collapse[1:])


class TestBuildRegimePredictions:
    """Tests for build_regime_predictions."""

    def test_output_shapes(
        self,
        small_grid: np.ndarray,
        survive_priors: np.ndarray,
        collapse_priors: np.ndarray,
    ) -> None:
        s_pred, c_pred = build_regime_predictions(small_grid, survive_priors, collapse_priors)
        h, w = small_grid.shape
        assert s_pred.shape == (h, w, C)
        assert c_pred.shape == (h, w, C)

    def test_settlement_cells_get_correct_priors(
        self,
        small_grid: np.ndarray,
        survive_priors: np.ndarray,
        collapse_priors: np.ndarray,
    ) -> None:
        s_pred, c_pred = build_regime_predictions(small_grid, survive_priors, collapse_priors)
        idx = int(InternalTerrain.SETTLEMENT)
        np.testing.assert_array_almost_equal(s_pred[3, 3], survive_priors[idx])
        np.testing.assert_array_almost_equal(c_pred[3, 3], collapse_priors[idx])


class TestScoreSoftBlend:
    """Tests for score_soft_blend convenience function."""

    def test_returns_valid_tensor(
        self,
        small_grid: np.ndarray,
        obs_store_survive: ObservationStore,
        survive_priors: np.ndarray,
        collapse_priors: np.ndarray,
    ) -> None:
        s_pred, c_pred = build_regime_predictions(small_grid, survive_priors, collapse_priors)
        result = score_soft_blend(small_grid, obs_store_survive, 0, s_pred, c_pred, 0.8)
        assert result.shape == (*small_grid.shape, C)
        assert np.all(result >= 0.0)


class TestRegimeDetectionIntegration:
    """End-to-end regime detection from observations."""

    def test_survive_regime_favors_settlement(
        self,
        small_grid: np.ndarray,
        obs_store_survive: ObservationStore,
        survive_priors: np.ndarray,
        collapse_priors: np.ndarray,
    ) -> None:
        confidence = estimate_regime_confidence([small_grid], [obs_store_survive])
        s_pred, c_pred = build_regime_predictions(small_grid, survive_priors, collapse_priors)
        blended = score_soft_blend(small_grid, obs_store_survive, 0, s_pred, c_pred, confidence)
        assert blended[3, 3, int(Terrain.SETTLEMENT)] > blended[3, 3, int(Terrain.RUIN)]

    def test_collapse_regime_shifts_to_ruin(
        self,
        small_grid: np.ndarray,
        obs_store_collapse: ObservationStore,
        survive_priors: np.ndarray,
        collapse_priors: np.ndarray,
    ) -> None:
        confidence = estimate_regime_confidence([small_grid], [obs_store_collapse])
        grid2 = small_grid.copy()
        grid2[7, 7] = int(InternalTerrain.SETTLEMENT)
        s_pred2, c_pred2 = build_regime_predictions(grid2, survive_priors, collapse_priors)
        coverage = obs_store_collapse.get_coverage_mask(0)
        blended = soft_blend_predictions(s_pred2, c_pred2, confidence, coverage)
        assert blended[7, 7, int(Terrain.RUIN)] > blended[7, 7, int(Terrain.SETTLEMENT)]
