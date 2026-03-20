"""Tests for PriorPredictor and prior building utilities."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.constants import NUM_PREDICTION_CLASSES, PROBABILITY_FLOOR
from src.predictor_v2 import (
    DEFAULT_TERRAIN_PRIORS,
    PriorPredictor,
    _apply_static_overrides,
    _floor_and_normalize,
)
from src.terrain import InternalTerrain

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_grid() -> np.ndarray:
    """5x5 grid with ocean border and mixed interior."""
    grid = np.full((5, 5), InternalTerrain.PLAINS, dtype=np.int8)
    grid[0, :] = InternalTerrain.OCEAN
    grid[-1, :] = InternalTerrain.OCEAN
    grid[:, 0] = InternalTerrain.OCEAN
    grid[:, -1] = InternalTerrain.OCEAN
    grid[2, 2] = InternalTerrain.SETTLEMENT
    grid[1, 2] = InternalTerrain.FOREST
    grid[3, 2] = InternalTerrain.MOUNTAIN
    return grid


@pytest.fixture
def predictor(simple_grid: np.ndarray) -> PriorPredictor:
    """PriorPredictor with no observations."""
    settlements = [{"x": 2, "y": 2}]
    return PriorPredictor(
        grid=simple_grid,
        settlements=settlements,
        observation_store=None,
        priors=None,
    )


# ---------------------------------------------------------------------------
# Output shape tests
# ---------------------------------------------------------------------------


def test_predict_output_shape(predictor: PriorPredictor) -> None:
    """Prediction tensor has correct H x W x 6 shape."""
    result = predictor.predict(seed_index=0)
    assert result.shape == (5, 5, NUM_PREDICTION_CLASSES)


def test_predict_output_dtype(predictor: PriorPredictor) -> None:
    """Prediction tensor uses float64."""
    result = predictor.predict(seed_index=0)
    assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------


def test_predict_rows_sum_to_one(predictor: PriorPredictor) -> None:
    """Every cell's probability vector sums to 1.0."""
    result = predictor.predict(seed_index=0)
    row_sums = result.sum(axis=2)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


def test_predict_all_values_positive(predictor: PriorPredictor) -> None:
    """All probabilities are strictly positive after floor + normalize."""
    result = predictor.predict(seed_index=0)
    # Floor is applied before normalization, so minimum is floor/(1+5*floor)
    min_expected = PROBABILITY_FLOOR / (1.0 + 5 * PROBABILITY_FLOOR)
    assert np.all(result >= min_expected - 1e-12)


# ---------------------------------------------------------------------------
# Static terrain tests
# ---------------------------------------------------------------------------


def test_ocean_cells_have_high_empty_prob(
    simple_grid: np.ndarray,
) -> None:
    """Ocean cells should predict Empty class with high confidence."""
    pred = PriorPredictor(simple_grid, [], None, None)
    result = pred.predict(seed_index=0)
    ocean_mask = simple_grid == InternalTerrain.OCEAN
    empty_probs = result[ocean_mask, 0]
    assert np.all(empty_probs > 0.9)


def test_mountain_cells_have_high_mountain_prob(
    simple_grid: np.ndarray,
) -> None:
    """Mountain cells should predict Mountain class with high confidence."""
    pred = PriorPredictor(simple_grid, [], None, None)
    result = pred.predict(seed_index=0)
    mountain_mask = simple_grid == InternalTerrain.MOUNTAIN
    mountain_probs = result[mountain_mask, 5]
    assert np.all(mountain_probs > 0.9)


def test_apply_static_overrides_ocean() -> None:
    """Static override sets ocean to near-certain Empty."""
    grid = np.array([[InternalTerrain.OCEAN]], dtype=np.int8)
    tensor = np.ones((1, 1, NUM_PREDICTION_CLASSES)) / NUM_PREDICTION_CLASSES
    result = _apply_static_overrides(tensor, grid)
    assert result[0, 0, 0] > 0.9


def test_apply_static_overrides_mountain() -> None:
    """Static override sets mountain to near-certain Mountain."""
    grid = np.array([[InternalTerrain.MOUNTAIN]], dtype=np.int8)
    tensor = np.ones((1, 1, NUM_PREDICTION_CLASSES)) / NUM_PREDICTION_CLASSES
    result = _apply_static_overrides(tensor, grid)
    assert result[0, 0, 5] > 0.9


# ---------------------------------------------------------------------------
# Observation blending tests
# ---------------------------------------------------------------------------


def test_blend_observations_with_mock_store(
    simple_grid: np.ndarray,
) -> None:
    """Observations should shift predictions toward observed values."""
    store = MagicMock()
    obs = np.full((5, 5, NUM_PREDICTION_CLASSES), np.nan)
    obs[2, 2] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    store.get_observed_probs.return_value = obs
    coverage = np.zeros((5, 5), dtype=bool)
    coverage[2, 2] = True
    store.get_coverage_mask.return_value = coverage
    counts = np.zeros((5, 5), dtype=np.int32)
    counts[2, 2] = 10
    store.observation_count.return_value = counts
    store.coverage_fraction.return_value = 1.0 / 25.0

    pred = PriorPredictor(simple_grid, [{"x": 2, "y": 2}], store, None)
    result = pred.predict(seed_index=0)
    # With 10 observations, weight is 0.8 * 10/(10+5) = 0.533
    # Settlement class (1) should be boosted
    assert result[2, 2, 1] > 0.3


def test_no_observations_returns_prior_based(
    predictor: PriorPredictor,
) -> None:
    """With no observation store, prediction relies only on priors."""
    result = predictor.predict(seed_index=0)
    # Plains cells should have non-zero Empty probability
    assert result[1, 1, 0] > 0.0


# ---------------------------------------------------------------------------
# Floor and normalize
# ---------------------------------------------------------------------------


def test_floor_and_normalize_zeros() -> None:
    """All-zero input gets floored and normalized to uniform."""
    tensor = np.zeros((2, 2, NUM_PREDICTION_CLASSES))
    result = _floor_and_normalize(tensor)
    expected = 1.0 / NUM_PREDICTION_CLASSES
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_floor_and_normalize_preserves_dominant() -> None:
    """A dominant class remains dominant after flooring."""
    tensor = np.zeros((1, 1, NUM_PREDICTION_CLASSES))
    tensor[0, 0, 3] = 1.0
    result = _floor_and_normalize(tensor)
    assert result[0, 0, 3] > 0.9


# ---------------------------------------------------------------------------
# Default priors validation
# ---------------------------------------------------------------------------


def test_default_priors_cover_all_terrains() -> None:
    """DEFAULT_TERRAIN_PRIORS has entries for all 7 internal terrains."""
    for t in range(7):
        assert t in DEFAULT_TERRAIN_PRIORS
        assert len(DEFAULT_TERRAIN_PRIORS[t]) == NUM_PREDICTION_CLASSES


def test_default_priors_ocean_is_certain_empty() -> None:
    """Ocean prior is [1, 0, 0, 0, 0, 0]."""
    ocean_prior = DEFAULT_TERRAIN_PRIORS[InternalTerrain.OCEAN]
    assert ocean_prior[0] == 1.0
    assert sum(ocean_prior[1:]) == 0.0


def test_default_priors_mountain_is_certain_mountain() -> None:
    """Mountain prior is [0, 0, 0, 0, 0, 1]."""
    mtn_prior = DEFAULT_TERRAIN_PRIORS[InternalTerrain.MOUNTAIN]
    assert mtn_prior[5] == 1.0
    assert sum(mtn_prior[:5]) == 0.0
