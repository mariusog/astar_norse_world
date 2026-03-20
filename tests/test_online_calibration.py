"""Tests for online calibration module."""

import numpy as np

from src.constants import (
    NUM_PREDICTION_CLASSES,
    REGIME_AGGRESSIVE,
    REGIME_COLLAPSE,
    REGIME_SURVIVE,
)
from src.online_calibration import detect_regime, online_calibrate
from src.terrain import InternalTerrain


def _make_grid(size: int = 10) -> np.ndarray:
    """Create a test grid with mixed terrain."""
    grid = np.full((size, size), InternalTerrain.PLAINS, dtype=np.int8)
    grid[0, :] = InternalTerrain.OCEAN
    grid[:, 0] = InternalTerrain.OCEAN
    grid[5, 5] = InternalTerrain.SETTLEMENT
    grid[8, 8] = InternalTerrain.MOUNTAIN
    return grid


def _uniform_pred(h: int, w: int) -> np.ndarray:
    """Uniform prediction tensor."""
    return np.full((h, w, NUM_PREDICTION_CLASSES), 1.0 / NUM_PREDICTION_CLASSES)


class TestOnlineCalibrate:
    """Tests for online_calibrate function."""

    def test_too_few_observations_returns_input(self) -> None:
        grid = _make_grid()
        pred = _uniform_pred(10, 10)
        obs = np.full((10, 10, 6), np.nan)
        coverage = np.zeros((10, 10), dtype=bool)
        # Only 1 observed cell (< min threshold)
        obs[5, 5] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        coverage[5, 5] = True
        result = online_calibrate(pred, grid, obs, coverage)
        np.testing.assert_array_equal(result, pred)

    def test_sufficient_observations_modifies_prediction(self) -> None:
        grid = _make_grid()
        pred = _uniform_pred(10, 10)
        obs = np.full((10, 10, 6), np.nan)
        coverage = np.zeros((10, 10), dtype=bool)
        # Observe several plains cells
        for y in range(2, 6):
            for x in range(2, 6):
                obs[y, x] = [0.8, 0.1, 0.0, 0.0, 0.1, 0.0]
                coverage[y, x] = True
        result = online_calibrate(pred, grid, obs, coverage)
        # Should be normalized
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-6)
        # Plains cells should have shifted toward observed distribution
        plains_mask = grid == InternalTerrain.PLAINS
        diff = np.abs(result[plains_mask] - pred[plains_mask])
        assert diff.max() > 0.01

    def test_output_probabilities_valid(self) -> None:
        grid = _make_grid()
        pred = _uniform_pred(10, 10)
        obs = np.full((10, 10, 6), np.nan)
        coverage = np.zeros((10, 10), dtype=bool)
        for y in range(1, 8):
            for x in range(1, 8):
                obs[y, x] = [0.5, 0.2, 0.1, 0.05, 0.1, 0.05]
                coverage[y, x] = True
        result = online_calibrate(pred, grid, obs, coverage)
        assert (result >= 0).all()
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-6)


class TestDetectRegime:
    """Tests for detect_regime function."""

    def test_collapse_regime(self) -> None:
        grid = _make_grid()
        obs = np.full((10, 10, 6), np.nan)
        coverage = np.zeros((10, 10), dtype=bool)
        # All observed dynamic cells show mostly empty (collapse)
        for y in range(2, 8):
            for x in range(2, 8):
                obs[y, x] = [0.95, 0.01, 0.0, 0.02, 0.02, 0.0]
                coverage[y, x] = True
        regime = detect_regime(obs, coverage, grid)
        assert regime == REGIME_COLLAPSE

    def test_aggressive_regime(self) -> None:
        grid = _make_grid()
        obs = np.full((10, 10, 6), np.nan)
        coverage = np.zeros((10, 10), dtype=bool)
        # High settlement+port probability (aggressive)
        for y in range(2, 8):
            for x in range(2, 8):
                obs[y, x] = [0.3, 0.35, 0.1, 0.05, 0.15, 0.05]
                coverage[y, x] = True
        regime = detect_regime(obs, coverage, grid)
        assert regime == REGIME_AGGRESSIVE

    def test_survive_regime(self) -> None:
        grid = _make_grid()
        obs = np.full((10, 10, 6), np.nan)
        coverage = np.zeros((10, 10), dtype=bool)
        # Moderate settlement probability (survive)
        for y in range(2, 8):
            for x in range(2, 8):
                obs[y, x] = [0.6, 0.1, 0.05, 0.05, 0.15, 0.05]
                coverage[y, x] = True
        regime = detect_regime(obs, coverage, grid)
        assert regime == REGIME_SURVIVE

    def test_no_observations_returns_survive(self) -> None:
        grid = _make_grid()
        obs = np.full((10, 10, 6), np.nan)
        coverage = np.zeros((10, 10), dtype=bool)
        regime = detect_regime(obs, coverage, grid)
        assert regime == REGIME_SURVIVE
