"""Tests for SettlementDistanceModel and position-aware prediction."""

from __future__ import annotations

import numpy as np
import pytest

from src.constants import NUM_PREDICTION_CLASSES
from src.position_priors import (
    SettlementDistanceModel,
    _apply_flat_priors,
    _compute_distances_from_positions,
    predict_from_position,
)
from src.terrain import InternalTerrain

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_distance_priors() -> dict[int, dict[int, list[float]]]:
    """Distance priors for plains terrain at bands 1-3."""
    plains_val = int(InternalTerrain.PLAINS)
    return {
        plains_val: {
            1: [0.3, 0.4, 0.02, 0.03, 0.2, 0.05],
            2: [0.5, 0.25, 0.01, 0.02, 0.15, 0.07],
            3: [0.6, 0.2, 0.01, 0.01, 0.12, 0.06],
        },
    }


@pytest.fixture
def distance_model(
    sample_distance_priors: dict[int, dict[int, list[float]]],
) -> SettlementDistanceModel:
    """Model built from sample priors."""
    return SettlementDistanceModel(sample_distance_priors)


@pytest.fixture
def small_grid() -> np.ndarray:
    """8x8 grid with ocean border and settlement at (4,4)."""
    grid = np.full((8, 8), InternalTerrain.PLAINS, dtype=np.int8)
    grid[0, :] = InternalTerrain.OCEAN
    grid[-1, :] = InternalTerrain.OCEAN
    grid[:, 0] = InternalTerrain.OCEAN
    grid[:, -1] = InternalTerrain.OCEAN
    grid[4, 4] = InternalTerrain.SETTLEMENT
    grid[2, 2] = InternalTerrain.FOREST
    grid[6, 6] = InternalTerrain.MOUNTAIN
    return grid


# ---------------------------------------------------------------------------
# SettlementDistanceModel tests
# ---------------------------------------------------------------------------


def test_model_get_prior_exists(
    distance_model: SettlementDistanceModel,
) -> None:
    """Returns a probability vector for known terrain and distance."""
    plains = int(InternalTerrain.PLAINS)
    result = distance_model.get_prior(plains, 1)
    assert result is not None
    assert result.shape == (NUM_PREDICTION_CLASSES,)


def test_model_get_prior_sums_to_one(
    distance_model: SettlementDistanceModel,
) -> None:
    """Returned prior vector sums to approximately 1."""
    plains = int(InternalTerrain.PLAINS)
    result = distance_model.get_prior(plains, 1)
    assert result is not None
    np.testing.assert_allclose(result.sum(), 1.0, atol=1e-10)


def test_model_get_prior_unknown_terrain(
    distance_model: SettlementDistanceModel,
) -> None:
    """Returns None for terrain types not in the model."""
    result = distance_model.get_prior(99, 1)
    assert result is None


def test_model_get_prior_unknown_distance(
    distance_model: SettlementDistanceModel,
) -> None:
    """Returns None for distance bands not in the model."""
    plains = int(InternalTerrain.PLAINS)
    # Only bands 1-3 in fixture; band 4 is absent
    result = distance_model.get_prior(plains, 4)
    assert result is None


def test_model_clamps_distance_to_max(
    distance_model: SettlementDistanceModel,
) -> None:
    """Distances beyond MAX_DISTANCE_BAND are clamped."""
    plains = int(InternalTerrain.PLAINS)
    # Distance 100 clamps to MAX_DISTANCE_BAND=5, not in fixture -> None
    result = distance_model.get_prior(plains, 100)
    assert result is None


def test_model_terrain_types(
    distance_model: SettlementDistanceModel,
) -> None:
    """terrain_types returns list of available terrain types."""
    types = distance_model.terrain_types
    assert int(InternalTerrain.PLAINS) in types


# ---------------------------------------------------------------------------
# Distance computation tests
# ---------------------------------------------------------------------------


def test_compute_distances_from_single_point() -> None:
    """Manhattan distance from a single settlement position."""
    positions = [(2, 2)]
    dist_map = _compute_distances_from_positions(positions, 5, 5)
    assert dist_map[2, 2] == 0.0
    assert dist_map[2, 3] == 1.0
    assert dist_map[0, 0] == 4.0


def test_compute_distances_from_two_points() -> None:
    """Distance is minimum over multiple settlement positions."""
    positions = [(0, 0), (4, 4)]
    dist_map = _compute_distances_from_positions(positions, 5, 5)
    assert dist_map[0, 0] == 0.0
    assert dist_map[4, 4] == 0.0
    assert dist_map[2, 2] == 4.0


def test_compute_distances_empty_positions() -> None:
    """With no settlements, all cells get max distance."""
    dist_map = _compute_distances_from_positions([], 3, 3)
    assert np.all(dist_map == 6.0)  # height + width = 3 + 3 = 6


# ---------------------------------------------------------------------------
# Flat priors tests
# ---------------------------------------------------------------------------


def test_apply_flat_priors_shape(small_grid: np.ndarray) -> None:
    """Flat priors tensor has correct shape."""
    priors = {int(InternalTerrain.PLAINS): [0.8, 0.1, 0.0, 0.0, 0.1, 0.0]}
    tensor = _apply_flat_priors(small_grid, priors, 8, 8)
    assert tensor.shape == (8, 8, NUM_PREDICTION_CLASSES)


def test_apply_flat_priors_applies_correctly(
    small_grid: np.ndarray,
) -> None:
    """Flat priors are applied to matching terrain cells."""
    plains_val = int(InternalTerrain.PLAINS)
    priors = {plains_val: [0.8, 0.1, 0.0, 0.0, 0.1, 0.0]}
    tensor = _apply_flat_priors(small_grid, priors, 8, 8)
    # Cell (1,1) is plains
    np.testing.assert_allclose(tensor[1, 1], [0.8, 0.1, 0.0, 0.0, 0.1, 0.0])


def test_apply_flat_priors_unmatched_terrain() -> None:
    """Cells with terrain not in priors dict stay at zero."""
    grid = np.full((3, 3), InternalTerrain.FOREST, dtype=np.int8)
    priors = {int(InternalTerrain.PLAINS): [0.8, 0.1, 0.0, 0.0, 0.1, 0.0]}
    tensor = _apply_flat_priors(grid, priors, 3, 3)
    np.testing.assert_allclose(tensor, 0.0)


# ---------------------------------------------------------------------------
# predict_from_position integration tests
# ---------------------------------------------------------------------------


def test_predict_from_position_shape(
    small_grid: np.ndarray,
    distance_model: SettlementDistanceModel,
) -> None:
    """Output tensor has correct shape."""
    base_priors = {
        int(InternalTerrain.PLAINS): [0.7, 0.15, 0.01, 0.02, 0.1, 0.02],
        int(InternalTerrain.OCEAN): [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        int(InternalTerrain.FOREST): [0.1, 0.1, 0.01, 0.01, 0.7, 0.08],
        int(InternalTerrain.MOUNTAIN): [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        int(InternalTerrain.SETTLEMENT): [0.3, 0.4, 0.01, 0.03, 0.2, 0.06],
    }
    settlements = [{"x": 4, "y": 4}]
    result = predict_from_position(
        small_grid,
        settlements,
        base_priors,
        distance_model,
    )
    assert result.shape == (8, 8, NUM_PREDICTION_CLASSES)


def test_predict_from_position_modifies_near_cells(
    small_grid: np.ndarray,
    distance_model: SettlementDistanceModel,
) -> None:
    """Cells near settlements differ from flat priors."""
    plains_val = int(InternalTerrain.PLAINS)
    flat_prior = [0.7, 0.15, 0.01, 0.02, 0.1, 0.02]
    base_priors = {plains_val: flat_prior}
    settlements = [{"x": 4, "y": 4}]
    result = predict_from_position(
        small_grid,
        settlements,
        base_priors,
        distance_model,
        blend_weight=1.0,
    )
    # Cell (4,3) is plains at distance 1 from settlement
    # With weight=1.0, it should be fully replaced by distance prior
    cell = result[4, 3]
    flat = np.array(flat_prior)
    # Should differ from flat prior (distance model overrides)
    assert not np.allclose(cell, flat)


def test_predict_from_position_zero_weight_is_flat(
    small_grid: np.ndarray,
    distance_model: SettlementDistanceModel,
) -> None:
    """With blend_weight=0, output matches flat priors exactly."""
    plains_val = int(InternalTerrain.PLAINS)
    flat_prior = [0.7, 0.15, 0.01, 0.02, 0.1, 0.02]
    base_priors = {plains_val: flat_prior}
    settlements = [{"x": 4, "y": 4}]
    result = predict_from_position(
        small_grid,
        settlements,
        base_priors,
        distance_model,
        blend_weight=0.0,
    )
    # Plains cells should match flat prior exactly
    plains_mask = small_grid == plains_val
    for row, col in np.argwhere(plains_mask):
        np.testing.assert_allclose(
            result[row, col],
            flat_prior,
            atol=1e-12,
        )


@pytest.mark.slow()
def test_build_distance_model_from_round_data() -> None:
    """Integration test: build distance model from actual data."""
    from src.position_priors import build_distance_model

    model = build_distance_model("data/rounds")
    assert len(model._profiles) > 0
