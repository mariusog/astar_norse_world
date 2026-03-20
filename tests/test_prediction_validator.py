"""Tests for pre-submission prediction validation."""

from __future__ import annotations

import numpy as np

from src.prediction_validator import validate_predictions
from src.terrain import InternalTerrain, Terrain

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grid(h: int = 10, w: int = 10) -> np.ndarray:
    """Create a grid with ocean border and mountains at (2,2)."""
    grid = np.full((h, w), InternalTerrain.PLAINS, dtype=np.int8)
    grid[0, :] = InternalTerrain.OCEAN
    grid[-1, :] = InternalTerrain.OCEAN
    grid[:, 0] = InternalTerrain.OCEAN
    grid[:, -1] = InternalTerrain.OCEAN
    grid[2, 2] = InternalTerrain.MOUNTAIN
    return grid


def _make_good_pred(grid: np.ndarray) -> np.ndarray:
    """Create a valid prediction matching the grid."""
    h, w = grid.shape
    pred = np.full((h, w, 6), 0.01, dtype=np.float64)

    for r in range(h):
        for c in range(w):
            cell = grid[r, c]
            if cell == InternalTerrain.OCEAN:
                cls = Terrain.EMPTY
            elif cell == InternalTerrain.MOUNTAIN:
                cls = Terrain.MOUNTAIN
            else:
                cls = Terrain.EMPTY
            pred[r, c, cls] = 0.95
    # Normalize each cell
    pred = pred / pred.sum(axis=2, keepdims=True)
    return pred


def _make_predictions(
    grid: np.ndarray,
    n_seeds: int = 5,
    noise_seed: int = 42,
) -> list[np.ndarray]:
    """Create n_seeds slightly different valid predictions."""
    rng = np.random.default_rng(noise_seed)
    base = _make_good_pred(grid)
    preds = []
    for _ in range(n_seeds):
        p = base.copy()
        noise = rng.uniform(-0.005, 0.005, size=p.shape)
        p += noise
        p = np.clip(p, 0.01, None)
        p /= p.sum(axis=2, keepdims=True)
        preds.append(p)
    return preds


# ---------------------------------------------------------------------------
# Tests: valid predictions pass
# ---------------------------------------------------------------------------


class TestValidPredictionsPass:
    def test_valid_predictions_no_errors(self) -> None:
        grid = _make_grid()
        grids = [grid] * 5
        preds = _make_predictions(grid)
        errors = validate_predictions(preds, grids)
        assert errors == [], f"Expected no errors, got: {errors}"


# ---------------------------------------------------------------------------
# Tests: shape checks
# ---------------------------------------------------------------------------


class TestShapeChecks:
    def test_wrong_shape_caught(self) -> None:
        grid = _make_grid()
        bad_pred = np.zeros((10, 10, 4))
        errors = validate_predictions([bad_pred], [grid])
        assert any("shape" in e.lower() for e in errors)

    def test_mismatched_grid_pred_size(self) -> None:
        grid = _make_grid(10, 10)
        pred = np.zeros((8, 8, 6))
        errors = validate_predictions([pred], [grid])
        assert any("shape" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Tests: uniform predictions caught
# ---------------------------------------------------------------------------


class TestUniformPredictionsCaught:
    def test_uniform_predictions_flagged(self) -> None:
        grid = _make_grid()
        uniform = np.full((10, 10, 6), 1.0 / 6.0)
        grids = [grid] * 5
        preds = [uniform.copy() for _ in range(5)]
        # Add tiny noise so they're not identical
        rng = np.random.default_rng(99)
        for p in preds:
            p += rng.uniform(-1e-6, 1e-6, p.shape)
            p /= p.sum(axis=2, keepdims=True)
        errors = validate_predictions(preds, grids)
        assert any("uniform" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Tests: unnormalized predictions caught
# ---------------------------------------------------------------------------


class TestUnnormalizedPredictionsCaught:
    def test_unnormalized_cells_flagged(self) -> None:
        grid = _make_grid()
        pred = _make_good_pred(grid)
        # Break normalization for several cells
        pred[3, 3, :] = 0.5  # sums to 3.0
        errors = validate_predictions([pred], [grid])
        assert any("normalized" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Tests: wrong static terrain caught
# ---------------------------------------------------------------------------


class TestStaticTerrainCaught:
    def test_ocean_wrong_class_flagged(self) -> None:
        grid = _make_grid()
        pred = _make_good_pred(grid)
        # Set ocean cells to predict settlement instead of EMPTY
        ocean_mask = grid == InternalTerrain.OCEAN
        pred[ocean_mask, Terrain.EMPTY] = 0.01
        pred[ocean_mask, Terrain.SETTLEMENT] = 0.95
        pred[ocean_mask] /= pred[ocean_mask].sum(axis=1, keepdims=True)
        errors = validate_predictions([pred], [grid])
        assert any("ocean" in e.lower() for e in errors)

    def test_mountain_wrong_class_flagged(self) -> None:
        grid = _make_grid()
        pred = _make_good_pred(grid)
        mtn_mask = grid == InternalTerrain.MOUNTAIN
        pred[mtn_mask, Terrain.MOUNTAIN] = 0.01
        pred[mtn_mask, Terrain.EMPTY] = 0.95
        pred[mtn_mask] /= pred[mtn_mask].sum(axis=1, keepdims=True)
        errors = validate_predictions([pred], [grid])
        assert any("mountain" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Tests: below-floor values caught
# ---------------------------------------------------------------------------


class TestBelowFloorCaught:
    def test_zero_probability_flagged(self) -> None:
        grid = _make_grid()
        pred = _make_good_pred(grid)
        pred[5, 5, 3] = 0.0  # below floor
        pred[5, 5] /= pred[5, 5].sum()
        errors = validate_predictions([pred], [grid])
        assert any("floor" in e.lower() for e in errors)

    def test_negative_probability_flagged(self) -> None:
        grid = _make_grid()
        pred = _make_good_pred(grid)
        pred[4, 4, 2] = -0.01
        errors = validate_predictions([pred], [grid])
        assert any("floor" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Tests: non-trivial (identical copies caught)
# ---------------------------------------------------------------------------


class TestNonTrivialCheck:
    def test_identical_predictions_flagged(self) -> None:
        grid = _make_grid()
        pred = _make_good_pred(grid)
        preds = [pred.copy() for _ in range(5)]
        errors = validate_predictions(preds, [grid] * 5)
        assert any("identical" in e.lower() for e in errors)
