"""Tests for shared prediction utilities."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.constants import (
    NUM_PREDICTION_CLASSES,
    OBS_CONFIDENCE_K,
    STATIC_TERRAIN_CONFIDENCE,
)
from src.prediction_utils import (
    apply_static_overrides,
    blend_observations,
    floor_and_normalize,
)
from src.terrain import InternalTerrain

# ---------------------------------------------------------------------------
# floor_and_normalize
# ---------------------------------------------------------------------------


class TestFloorAndNormalize:
    """Tests for floor_and_normalize."""

    def test_zeros_become_uniform(self) -> None:
        """All-zero input produces uniform distribution."""
        tensor = np.zeros((2, 2, NUM_PREDICTION_CLASSES))
        result = floor_and_normalize(tensor)
        expected = 1.0 / NUM_PREDICTION_CLASSES
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_cells_sum_to_one(self) -> None:
        """Every cell sums to 1 after normalization."""
        rng = np.random.default_rng(42)
        tensor = rng.random((5, 5, NUM_PREDICTION_CLASSES))
        result = floor_and_normalize(tensor)
        sums = result.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-10)

    def test_no_value_below_floor(self) -> None:
        """All values are at least floor after normalization."""
        tensor = np.zeros((3, 3, NUM_PREDICTION_CLASSES))
        tensor[:, :, 0] = 1.0
        result = floor_and_normalize(tensor)
        assert result.min() > 0.0

    def test_dominant_class_preserved(self) -> None:
        """A dominant class remains dominant."""
        tensor = np.zeros((1, 1, NUM_PREDICTION_CLASSES))
        tensor[0, 0, 3] = 1.0
        result = floor_and_normalize(tensor)
        assert result[0, 0, 3] > 0.9

    def test_custom_floor(self) -> None:
        """Custom floor value is respected."""
        tensor = np.zeros((1, 1, NUM_PREDICTION_CLASSES))
        tensor[0, 0, 0] = 1.0
        result = floor_and_normalize(tensor, floor=0.1)
        # After floor: [1.0, 0.1, 0.1, 0.1, 0.1, 0.1], sum=1.5
        # Non-dominant classes get 0.1/1.5 ~ 0.0667
        expected_min = 0.1 / 1.5
        assert result[0, 0, 1] == pytest.approx(expected_min, abs=1e-10)


# ---------------------------------------------------------------------------
# apply_static_overrides
# ---------------------------------------------------------------------------


class TestApplyStaticOverrides:
    """Tests for apply_static_overrides."""

    def test_ocean_gets_high_empty_prob(self) -> None:
        """Ocean cells get near-certain Empty (class 0)."""
        grid = np.array([[InternalTerrain.OCEAN]], dtype=np.int8)
        tensor = np.full((1, 1, NUM_PREDICTION_CLASSES), 1.0 / 6)
        result = apply_static_overrides(tensor, grid)
        assert result[0, 0, 0] == pytest.approx(STATIC_TERRAIN_CONFIDENCE, abs=1e-6)

    def test_mountain_gets_high_mountain_prob(self) -> None:
        """Mountain cells get near-certain Mountain (class 5)."""
        grid = np.array([[InternalTerrain.MOUNTAIN]], dtype=np.int8)
        tensor = np.full((1, 1, NUM_PREDICTION_CLASSES), 1.0 / 6)
        result = apply_static_overrides(tensor, grid)
        assert result[0, 0, 5] == pytest.approx(STATIC_TERRAIN_CONFIDENCE, abs=1e-6)

    def test_static_cells_sum_to_one(self) -> None:
        """Static terrain cells still sum to 1."""
        grid = np.array(
            [[InternalTerrain.OCEAN, InternalTerrain.MOUNTAIN]],
            dtype=np.int8,
        )
        tensor = np.full((1, 2, NUM_PREDICTION_CLASSES), 1.0 / 6)
        result = apply_static_overrides(tensor, grid)
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-10)

    def test_non_static_unchanged(self) -> None:
        """Non-static terrain is not modified."""
        grid = np.array([[InternalTerrain.PLAINS]], dtype=np.int8)
        tensor = np.full((1, 1, NUM_PREDICTION_CLASSES), 1.0 / 6)
        result = apply_static_overrides(tensor, grid)
        np.testing.assert_array_almost_equal(result, tensor)

    def test_does_not_modify_input(self) -> None:
        """Input tensor is not mutated."""
        grid = np.array([[InternalTerrain.OCEAN]], dtype=np.int8)
        tensor = np.full((1, 1, NUM_PREDICTION_CLASSES), 1.0 / 6)
        original = tensor.copy()
        apply_static_overrides(tensor, grid)
        np.testing.assert_array_equal(tensor, original)


# ---------------------------------------------------------------------------
# blend_observations
# ---------------------------------------------------------------------------


class TestBlendObservations:
    """Tests for blend_observations."""

    def _make_store(
        self,
        shape: tuple[int, int],
        obs_probs: np.ndarray,
        coverage: np.ndarray,
        counts: np.ndarray,
    ) -> MagicMock:
        """Create a mock observation store."""
        store = MagicMock()
        store.get_observed_probs.return_value = obs_probs
        store.get_coverage_mask.return_value = coverage
        store.observation_count.return_value = counts
        return store

    def test_no_observations_returns_tensor(self) -> None:
        """With no observed cells, tensor is returned unchanged."""
        h, w = 3, 3
        tensor = np.full((h, w, NUM_PREDICTION_CLASSES), 1.0 / 6)
        obs_probs = np.full((h, w, NUM_PREDICTION_CLASSES), np.nan)
        coverage = np.zeros((h, w), dtype=bool)
        counts = np.zeros((h, w), dtype=np.int32)
        store = self._make_store((h, w), obs_probs, coverage, counts)

        result = blend_observations(tensor, store, seed_idx=0, max_weight=0.8)
        np.testing.assert_array_equal(result, tensor)

    def test_observed_cell_shifts_toward_obs(self) -> None:
        """Observed cells are shifted toward observation values."""
        h, w = 3, 3
        tensor = np.full((h, w, NUM_PREDICTION_CLASSES), 1.0 / 6)
        obs_probs = np.full((h, w, NUM_PREDICTION_CLASSES), np.nan)
        obs_probs[1, 1] = [0.9, 0.02, 0.02, 0.02, 0.02, 0.02]
        coverage = np.zeros((h, w), dtype=bool)
        coverage[1, 1] = True
        counts = np.zeros((h, w), dtype=np.int32)
        counts[1, 1] = 10
        store = self._make_store((h, w), obs_probs, coverage, counts)

        result = blend_observations(
            tensor,
            store,
            seed_idx=0,
            max_weight=0.8,
            k=OBS_CONFIDENCE_K,
        )
        assert result[1, 1, 0] > 1.0 / 6.0

    def test_does_not_modify_input(self) -> None:
        """Input tensor is not mutated."""
        h, w = 3, 3
        tensor = np.full((h, w, NUM_PREDICTION_CLASSES), 1.0 / 6)
        original = tensor.copy()
        obs_probs = np.full((h, w, NUM_PREDICTION_CLASSES), np.nan)
        obs_probs[1, 1] = [0.9, 0.02, 0.02, 0.02, 0.02, 0.02]
        coverage = np.zeros((h, w), dtype=bool)
        coverage[1, 1] = True
        counts = np.zeros((h, w), dtype=np.int32)
        counts[1, 1] = 5
        store = self._make_store((h, w), obs_probs, coverage, counts)

        blend_observations(tensor, store, seed_idx=0, max_weight=0.8)
        np.testing.assert_array_equal(tensor, original)
