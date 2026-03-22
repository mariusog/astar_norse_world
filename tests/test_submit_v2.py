"""Tests for V2 submission pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.submit_v2 import (
    _apply_static_overrides,
    _blend_observations,
    _build_prediction,
    _detect_regime,
    _floor_and_normalize,
)
from src.constants import NUM_PREDICTION_CLASSES
from src.feature_predictor import FeatureLookup
from src.observation import ObservationStore
from src.terrain import InternalTerrain


@pytest.fixture
def small_grid() -> np.ndarray:
    """5x5 test grid with ocean border and settlement."""
    grid = np.full((5, 5), InternalTerrain.PLAINS, dtype=np.int8)
    grid[0, :] = InternalTerrain.OCEAN
    grid[2, 2] = InternalTerrain.SETTLEMENT
    grid[4, 4] = InternalTerrain.MOUNTAIN
    return grid


@pytest.fixture
def empty_obs_store() -> ObservationStore:
    """Observation store with no observations."""
    return ObservationStore(height=5, width=5, num_seeds=1)


@pytest.fixture
def simple_lookup() -> FeatureLookup:
    """Minimal lookup for prediction tests."""
    return {
        (1, 0, 0): np.array([0.5, 0.2, 0.0, 0.1, 0.2, 0.0]),
        (1, 1, 0): np.array([0.4, 0.3, 0.0, 0.1, 0.2, 0.0]),
    }


class TestDetectRegime:
    """Tests for regime detection from round metadata."""

    def test_early_rounds_survive(self) -> None:
        assert _detect_regime({"round_number": 1}) == "survive"
        assert _detect_regime({"round_number": 2}) == "survive"

    def test_collapse_rounds(self) -> None:
        assert _detect_regime({"round_number": 3}) == "collapse"
        assert _detect_regime({"round_number": 4}) == "collapse"

    def test_aggressive_rounds(self) -> None:
        assert _detect_regime({"round_number": 5}) == "aggressive"
        assert _detect_regime({"round_number": 6}) == "aggressive"

    def test_missing_round_number_defaults_survive(self) -> None:
        assert _detect_regime({}) == "survive"


class TestBuildPrediction:
    """Tests for the full prediction builder."""

    def test_output_shape(
        self,
        small_grid: np.ndarray,
        empty_obs_store: ObservationStore,
        simple_lookup: FeatureLookup,
    ) -> None:
        pred = _build_prediction(small_grid, 0, empty_obs_store, simple_lookup)
        assert pred.shape == (5, 5, NUM_PREDICTION_CLASSES)

    def test_probabilities_sum_to_one(
        self,
        small_grid: np.ndarray,
        empty_obs_store: ObservationStore,
        simple_lookup: FeatureLookup,
    ) -> None:
        pred = _build_prediction(small_grid, 0, empty_obs_store, simple_lookup)
        sums = pred.sum(axis=2)
        np.testing.assert_array_almost_equal(sums, np.ones((5, 5)))

    def test_ocean_high_prob_class0(
        self,
        small_grid: np.ndarray,
        empty_obs_store: ObservationStore,
        simple_lookup: FeatureLookup,
    ) -> None:
        pred = _build_prediction(small_grid, 0, empty_obs_store, simple_lookup)
        assert pred[0, 0, 0] > 0.9

    def test_mountain_high_prob_class5(
        self,
        small_grid: np.ndarray,
        empty_obs_store: ObservationStore,
        simple_lookup: FeatureLookup,
    ) -> None:
        pred = _build_prediction(small_grid, 0, empty_obs_store, simple_lookup)
        assert pred[4, 4, 5] > 0.9


class TestBlendObservations:
    """Tests for observation blending."""

    def test_no_observations_returns_input(
        self,
        empty_obs_store: ObservationStore,
    ) -> None:
        tensor = np.full((5, 5, 6), 1.0 / 6)
        result = _blend_observations(tensor, empty_obs_store, 0)
        np.testing.assert_array_equal(result, tensor)


class TestApplyStaticOverrides:
    """Tests for static terrain overrides."""

    def test_ocean_overridden(self, small_grid: np.ndarray) -> None:
        tensor = np.full((5, 5, 6), 1.0 / 6)
        result = _apply_static_overrides(tensor, small_grid)
        assert result[0, 0, 0] > 0.9

    def test_mountain_overridden(self, small_grid: np.ndarray) -> None:
        tensor = np.full((5, 5, 6), 1.0 / 6)
        result = _apply_static_overrides(tensor, small_grid)
        assert result[4, 4, 5] > 0.9


class TestFloorAndNormalize:
    """Tests for floor and normalize."""

    def test_no_zeros(self) -> None:
        tensor = np.array([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        result = _floor_and_normalize(tensor)
        assert np.all(result > 0)

    def test_sums_to_one(self) -> None:
        tensor = np.array([[[0.5, 0.3, 0.1, 0.05, 0.03, 0.02]]])
        result = _floor_and_normalize(tensor)
        assert result[0, 0].sum() == pytest.approx(1.0, abs=1e-10)
