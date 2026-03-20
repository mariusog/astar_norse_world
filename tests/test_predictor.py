"""Tests for prediction tensor generator (T21)."""

from __future__ import annotations

import numpy as np
import pytest

from src.constants import (
    NUM_PREDICTION_CLASSES,
    OBSERVATION_WEIGHT,
    SIMULATION_WEIGHT,
    STATIC_TERRAIN_CONFIDENCE,
)
from src.observation import ObservationStore
from src.predictor import (
    Predictor,
    _apply_floor_and_normalize,
    _apply_static_terrain,
    _blend_probabilities,
)
from src.settlement import Settlement
from src.terrain import InternalTerrain

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_grid() -> np.ndarray:
    """10x10 grid with ocean border, mountains, and a settlement."""
    grid = np.full((10, 10), InternalTerrain.PLAINS, dtype=np.int8)
    grid[0, :] = InternalTerrain.OCEAN
    grid[-1, :] = InternalTerrain.OCEAN
    grid[:, 0] = InternalTerrain.OCEAN
    grid[:, -1] = InternalTerrain.OCEAN
    grid[5, 5] = InternalTerrain.SETTLEMENT
    grid[3, 3] = InternalTerrain.MOUNTAIN
    grid[3, 4] = InternalTerrain.MOUNTAIN
    return grid


@pytest.fixture
def settlements() -> list[Settlement]:
    """A single settlement for testing."""
    return [
        Settlement(
            x=5,
            y=5,
            owner_id=0,
            population=50,
            food=100,
            wealth=0,
            defense=10,
            tech_level=1,
            is_port=False,
            has_longship=False,
        )
    ]


@pytest.fixture
def obs_store() -> ObservationStore:
    """10x10 observation store with some data."""
    store = ObservationStore(height=10, width=10, num_seeds=5)
    # Add observation for seed 0, covering center area
    patch = np.array(
        [
            [1, 1, 1, 1, 1],  # plains=1 in prediction classes
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.int8,
    )
    store.add_observation(0, viewport_x=3, viewport_y=3, grid_patch=patch)
    return store


@pytest.fixture
def uniform_sim_probs() -> np.ndarray:
    """10x10x6 uniform simulation probs."""
    return np.full((10, 10, 6), 1.0 / 6.0)


# ---------------------------------------------------------------------------
# _blend_probabilities
# ---------------------------------------------------------------------------


class TestBlendProbabilities:
    """Tests for probability blending."""

    def test_unobserved_uses_sim_only(
        self,
        uniform_sim_probs: np.ndarray,
    ) -> None:
        obs_probs = np.full((10, 10, 6), np.nan)
        coverage = np.zeros((10, 10), dtype=bool)
        result = _blend_probabilities(
            uniform_sim_probs,
            obs_probs,
            coverage,
            OBSERVATION_WEIGHT,
            SIMULATION_WEIGHT,
        )
        np.testing.assert_array_almost_equal(result, uniform_sim_probs)

    def test_observed_cells_are_blended(self) -> None:
        sim = np.full((5, 5, 6), 1.0 / 6.0)
        obs = np.full((5, 5, 6), np.nan)
        obs[2, 2] = [0.9, 0.02, 0.02, 0.02, 0.02, 0.02]
        coverage = np.zeros((5, 5), dtype=bool)
        coverage[2, 2] = True
        obs_counts = np.zeros((5, 5), dtype=np.int32)
        obs_counts[2, 2] = 10  # high count = strong obs weight

        result = _blend_probabilities(sim, obs, coverage, obs_counts, 0.8)
        # With count=10 and K=5, confidence=0.667, w_obs=0.533
        assert result[2, 2, 0] > 1.0 / 6.0  # leaning toward obs

    def test_blend_does_not_modify_sim(
        self,
        uniform_sim_probs: np.ndarray,
    ) -> None:
        original = uniform_sim_probs.copy()
        obs_probs = np.full((10, 10, 6), np.nan)
        coverage = np.zeros((10, 10), dtype=bool)
        obs_counts = np.zeros((10, 10), dtype=np.int32)
        _blend_probabilities(
            uniform_sim_probs,
            obs_probs,
            coverage,
            obs_counts,
            OBSERVATION_WEIGHT,
        )
        np.testing.assert_array_equal(uniform_sim_probs, original)


# ---------------------------------------------------------------------------
# _apply_static_terrain
# ---------------------------------------------------------------------------


class TestApplyStaticTerrain:
    """Tests for static terrain certainty override."""

    def test_mountain_gets_high_confidence(
        self,
        simple_grid: np.ndarray,
    ) -> None:
        probs = np.full((10, 10, 6), 1.0 / 6.0)
        result = _apply_static_terrain(probs, simple_grid)
        # Mountain class = 5
        assert result[3, 3, 5] == pytest.approx(
            STATIC_TERRAIN_CONFIDENCE,
            abs=1e-6,
        )

    def test_ocean_gets_high_confidence(
        self,
        simple_grid: np.ndarray,
    ) -> None:
        probs = np.full((10, 10, 6), 1.0 / 6.0)
        result = _apply_static_terrain(probs, simple_grid)
        # Ocean -> EMPTY class = 0
        assert result[0, 5, 0] == pytest.approx(
            STATIC_TERRAIN_CONFIDENCE,
            abs=1e-6,
        )

    def test_static_terrain_sums_to_one(
        self,
        simple_grid: np.ndarray,
    ) -> None:
        probs = np.full((10, 10, 6), 1.0 / 6.0)
        result = _apply_static_terrain(probs, simple_grid)
        # Check mountain cell sums to 1
        assert result[3, 3].sum() == pytest.approx(1.0, abs=1e-6)
        # Check ocean cell sums to 1
        assert result[0, 5].sum() == pytest.approx(1.0, abs=1e-6)

    def test_non_static_terrain_unchanged(
        self,
        simple_grid: np.ndarray,
    ) -> None:
        probs = np.full((10, 10, 6), 1.0 / 6.0)
        result = _apply_static_terrain(probs, simple_grid)
        # Plains cell (4,4) should be unchanged
        np.testing.assert_array_almost_equal(
            result[4, 4],
            np.full(6, 1.0 / 6.0),
        )


# ---------------------------------------------------------------------------
# _apply_floor_and_normalize
# ---------------------------------------------------------------------------


class TestApplyFloorAndNormalize:
    """Tests for probability floor and normalization."""

    def test_no_zeros_in_output(self) -> None:
        probs = np.zeros((5, 5, 6))
        probs[:, :, 0] = 1.0
        result = _apply_floor_and_normalize(probs)
        # After floor + renormalize, all values are positive
        assert result.min() > 0.0

    def test_cells_sum_to_one(self) -> None:
        probs = np.random.default_rng(42).random((10, 10, 6))
        result = _apply_floor_and_normalize(probs)
        sums = result.sum(axis=2)
        np.testing.assert_array_almost_equal(sums, np.ones((10, 10)))

    def test_already_normalized_stays_close(self) -> None:
        probs = np.full((3, 3, 6), 1.0 / 6.0)
        result = _apply_floor_and_normalize(probs)
        np.testing.assert_array_almost_equal(result, probs, decimal=2)


# ---------------------------------------------------------------------------
# Predictor.predict_from_sim
# ---------------------------------------------------------------------------


class TestPredictorPredictFromSim:
    """Tests for Predictor with pre-computed sim probs."""

    def test_output_shape(
        self,
        simple_grid: np.ndarray,
        settlements: list[Settlement],
        obs_store: ObservationStore,
        uniform_sim_probs: np.ndarray,
    ) -> None:
        pred = Predictor(simple_grid, settlements, obs_store)
        result = pred.predict_from_sim(uniform_sim_probs, seed_index=0)
        assert result.shape == (10, 10, NUM_PREDICTION_CLASSES)

    def test_all_cells_sum_to_one(
        self,
        simple_grid: np.ndarray,
        settlements: list[Settlement],
        obs_store: ObservationStore,
        uniform_sim_probs: np.ndarray,
    ) -> None:
        pred = Predictor(simple_grid, settlements, obs_store)
        result = pred.predict_from_sim(uniform_sim_probs, seed_index=0)
        sums = result.sum(axis=2)
        np.testing.assert_array_almost_equal(
            sums,
            np.ones((10, 10)),
            decimal=5,
        )

    def test_mountain_has_high_prob(
        self,
        simple_grid: np.ndarray,
        settlements: list[Settlement],
        obs_store: ObservationStore,
        uniform_sim_probs: np.ndarray,
    ) -> None:
        pred = Predictor(simple_grid, settlements, obs_store)
        result = pred.predict_from_sim(uniform_sim_probs, seed_index=0)
        # Mountain at (3,3), class 5
        assert result[3, 3, 5] > 0.9

    def test_no_zero_probabilities(
        self,
        simple_grid: np.ndarray,
        settlements: list[Settlement],
        obs_store: ObservationStore,
        uniform_sim_probs: np.ndarray,
    ) -> None:
        pred = Predictor(simple_grid, settlements, obs_store)
        result = pred.predict_from_sim(uniform_sim_probs, seed_index=0)
        # After floor + renormalize, all values are positive
        assert result.min() > 0.0

    def test_observed_cells_weighted_toward_obs(
        self,
        simple_grid: np.ndarray,
        settlements: list[Settlement],
        obs_store: ObservationStore,
    ) -> None:
        sim_probs = np.full((10, 10, 6), 1.0 / 6.0)
        pred = Predictor(simple_grid, settlements, obs_store)
        prediction = pred.predict_from_sim(sim_probs, seed_index=0)
        # Cell (4,4) is observed (in the 3,3 -> 8,8 viewport)
        coverage = obs_store.get_coverage_mask(0)
        assert coverage[4, 4]  # Confirm it's observed
        # Prediction should differ from uniform for observed cells
        assert not np.allclose(prediction[4, 4], 1.0 / 6.0)
