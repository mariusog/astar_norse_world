"""Tests for GridPredictor protocol adapters."""

from __future__ import annotations

import numpy as np
import pytest

from src.constants import NUM_PREDICTION_CLASSES
from src.feature_predictor import FeatureGridPredictor
from src.ml_predictor import MLGridPredictor, train_model
from src.terrain import InternalTerrain
from src.unified_priors import PriorGridPredictor


@pytest.fixture
def small_grid() -> np.ndarray:
    grid = np.full((5, 5), InternalTerrain.PLAINS, dtype=np.int8)
    grid[2, 2] = InternalTerrain.SETTLEMENT
    grid[0, 0] = InternalTerrain.OCEAN
    grid[4, 4] = InternalTerrain.MOUNTAIN
    return grid


class TestPriorGridPredictor:
    def test_output_shape(self, small_grid: np.ndarray) -> None:
        priors = np.full((7, NUM_PREDICTION_CLASSES), 1.0 / NUM_PREDICTION_CLASSES)
        pred = PriorGridPredictor(priors)
        result = pred.predict_grid(small_grid)
        assert result.shape == (5, 5, NUM_PREDICTION_CLASSES)

    def test_normalized(self, small_grid: np.ndarray) -> None:
        priors = np.full((7, NUM_PREDICTION_CLASSES), 1.0 / NUM_PREDICTION_CLASSES)
        pred = PriorGridPredictor(priors)
        result = pred.predict_grid(small_grid)
        sums = result.sum(axis=2)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)


class TestFeatureGridPredictor:
    def test_output_shape(self, small_grid: np.ndarray) -> None:
        lookup = {(1, 0, 0): np.array([0.8, 0.05, 0.05, 0.05, 0.025, 0.025])}
        pred = FeatureGridPredictor(lookup)
        result = pred.predict_grid(small_grid)
        assert result.shape == (5, 5, NUM_PREDICTION_CLASSES)

    def test_normalized(self, small_grid: np.ndarray) -> None:
        lookup = {}
        pred = FeatureGridPredictor(lookup)
        result = pred.predict_grid(small_grid)
        sums = result.sum(axis=2)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)


class TestMLGridPredictor:
    def test_output_shape(self, small_grid: np.ndarray) -> None:
        # Train a trivial model on synthetic data
        from src.ml_predictor import extract_cell_features

        x = extract_cell_features(small_grid)
        y = np.random.default_rng(42).dirichlet([1] * NUM_PREDICTION_CLASSES, size=x.shape[0])
        model = train_model(x, y.astype(np.float32), seed=42)
        pred = MLGridPredictor(model)
        result = pred.predict_grid(small_grid)
        assert result.shape == (5, 5, NUM_PREDICTION_CLASSES)

    def test_normalized(self, small_grid: np.ndarray) -> None:
        from src.ml_predictor import extract_cell_features

        x = extract_cell_features(small_grid)
        y = np.random.default_rng(42).dirichlet([1] * NUM_PREDICTION_CLASSES, size=x.shape[0])
        model = train_model(x, y.astype(np.float32), seed=42)
        pred = MLGridPredictor(model)
        result = pred.predict_grid(small_grid)
        sums = result.sum(axis=2)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)
