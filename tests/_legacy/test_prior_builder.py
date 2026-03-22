"""Tests for multi-round terrain prior builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src._legacy.prior_builder import (
    NUM_INTERNAL_TYPES,
    _compute_round_weight,
    _normalize_priors,
    _uniform_priors,
    build_prior_prediction,
    build_terrain_priors,
    load_priors,
    save_priors,
)
from src.constants import NUM_PREDICTION_CLASSES, PROBABILITY_FLOOR
from src.terrain import InternalTerrain


class TestUniformPriors:
    """Tests for _uniform_priors."""

    def test_shape_is_correct(self) -> None:
        priors = _uniform_priors()
        assert priors.shape == (NUM_INTERNAL_TYPES, NUM_PREDICTION_CLASSES)

    def test_rows_sum_to_one(self) -> None:
        priors = _uniform_priors()
        for i in range(NUM_INTERNAL_TYPES):
            assert abs(priors[i].sum() - 1.0) < 1e-10


class TestComputeRoundWeight:
    """Tests for _compute_round_weight."""

    def test_most_recent_round_has_weight_one(self) -> None:
        w = _compute_round_weight(5, 5)
        assert abs(w - 1.0) < 1e-10

    def test_older_round_has_lower_weight(self) -> None:
        w_recent = _compute_round_weight(5, 5)
        w_old = _compute_round_weight(3, 5)
        assert w_old < w_recent

    def test_weight_is_positive(self) -> None:
        w = _compute_round_weight(1, 100)
        assert w > 0


class TestNormalizePriors:
    """Tests for _normalize_priors."""

    def test_rows_sum_to_one(self) -> None:
        counts = np.zeros(
            (NUM_INTERNAL_TYPES, NUM_PREDICTION_CLASSES),
            dtype=np.float64,
        )
        counts[0] = [10, 5, 3, 1, 1, 0]
        counts[1] = [0, 0, 0, 0, 0, 0]
        priors = _normalize_priors(counts)
        for i in range(priors.shape[0]):
            assert abs(priors[i].sum() - 1.0) < 1e-10

    def test_zero_row_gets_uniform(self) -> None:
        counts = np.zeros(
            (NUM_INTERNAL_TYPES, NUM_PREDICTION_CLASSES),
            dtype=np.float64,
        )
        priors = _normalize_priors(counts)
        expected = 1.0 / NUM_PREDICTION_CLASSES
        assert np.allclose(priors[0], expected, atol=0.01)

    def test_floor_applied(self) -> None:
        counts = np.zeros(
            (NUM_INTERNAL_TYPES, NUM_PREDICTION_CLASSES),
            dtype=np.float64,
        )
        counts[0] = [100, 0, 0, 0, 0, 0]
        priors = _normalize_priors(counts)
        assert np.all(priors >= PROBABILITY_FLOOR)


class TestSaveLoadPriors:
    """Tests for save_priors and load_priors."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        original = np.random.default_rng(42).random((NUM_INTERNAL_TYPES, NUM_PREDICTION_CLASSES))
        path = tmp_path / "priors.npy"
        save_priors(original, path)
        loaded = load_priors(path)
        np.testing.assert_array_almost_equal(loaded, original)


class TestBuildTerrainPriors:
    """Tests for build_terrain_priors."""

    def test_no_data_returns_uniform(self, tmp_path: Path) -> None:
        priors = build_terrain_priors(tmp_path)
        expected = 1.0 / NUM_PREDICTION_CLASSES
        assert np.allclose(priors, expected, atol=0.01)

    def test_single_round_produces_valid_priors(self, tmp_path: Path) -> None:
        # Create a minimal round with known data
        _create_test_round(tmp_path, round_id=1, seed=0)
        priors = build_terrain_priors(tmp_path)

        assert priors.shape == (NUM_INTERNAL_TYPES, NUM_PREDICTION_CLASSES)
        for i in range(NUM_INTERNAL_TYPES):
            assert abs(priors[i].sum() - 1.0) < 1e-10
            assert np.all(priors[i] >= PROBABILITY_FLOOR)

    def test_priors_reflect_ground_truth(self, tmp_path: Path) -> None:
        # Ocean stays ocean -> should have high EMPTY probability
        _create_test_round_ocean_stays(tmp_path, round_id=1)
        priors = build_terrain_priors(tmp_path)

        # Ocean (index 0) should have high EMPTY (class 0) probability
        ocean_prior = priors[InternalTerrain.OCEAN]
        assert ocean_prior[0] > 0.5  # EMPTY class should dominate


class TestBuildPriorPrediction:
    """Tests for build_prior_prediction."""

    def test_output_shape(self) -> None:
        grid = np.full((4, 5), InternalTerrain.PLAINS, dtype=np.int8)
        priors = _uniform_priors()
        pred = build_prior_prediction(grid, priors)
        assert pred.shape == (4, 5, NUM_PREDICTION_CLASSES)

    def test_rows_sum_to_one(self) -> None:
        grid = np.full((3, 3), InternalTerrain.OCEAN, dtype=np.int8)
        priors = _uniform_priors()
        pred = build_prior_prediction(grid, priors)
        sums = pred.sum(axis=2)
        assert np.allclose(sums, 1.0, atol=1e-10)

    def test_floor_applied(self) -> None:
        grid = np.full((3, 3), InternalTerrain.OCEAN, dtype=np.int8)
        priors = np.zeros(
            (NUM_INTERNAL_TYPES, NUM_PREDICTION_CLASSES),
            dtype=np.float64,
        )
        priors[:, 0] = 1.0  # All mass on class 0
        pred = build_prior_prediction(grid, priors)
        assert np.all(pred >= PROBABILITY_FLOOR)

    def test_uses_correct_terrain_prior(self) -> None:
        grid = np.array(
            [[InternalTerrain.OCEAN, InternalTerrain.FOREST]],
            dtype=np.int8,
        )
        priors = _uniform_priors()
        # Make ocean and forest priors distinct
        priors[InternalTerrain.OCEAN] = [0.9, 0.02, 0.02, 0.02, 0.02, 0.02]
        priors[InternalTerrain.FOREST] = [0.02, 0.02, 0.02, 0.02, 0.9, 0.02]

        pred = build_prior_prediction(grid, priors)
        # Ocean cell should have high class 0
        assert pred[0, 0, 0] > pred[0, 0, 4]
        # Forest cell should have high class 4
        assert pred[0, 1, 4] > pred[0, 1, 0]


# --- Test data helpers ---


def _create_test_round(data_dir: Path, round_id: int, seed: int) -> None:
    """Create a minimal test round with plains grid."""
    rounds_dir = data_dir / "rounds" / str(round_id)
    for s in range(5):
        seed_dir = rounds_dir / f"seed_{s}"
        seed_dir.mkdir(parents=True)
        grid = np.full((5, 5), InternalTerrain.PLAINS, dtype=np.int8)
        grid[0, :] = InternalTerrain.OCEAN
        np.save(seed_dir / "initial_grid.npy", grid)
        np.save(seed_dir / "ground_truth.npy", grid)


def _create_test_round_ocean_stays(data_dir: Path, round_id: int) -> None:
    """Create round where ocean cells remain ocean in GT."""
    rounds_dir = data_dir / "rounds" / str(round_id)
    for s in range(5):
        seed_dir = rounds_dir / f"seed_{s}"
        seed_dir.mkdir(parents=True)
        initial = np.full((5, 5), InternalTerrain.OCEAN, dtype=np.int8)
        gt = np.full((5, 5), InternalTerrain.OCEAN, dtype=np.int8)
        np.save(seed_dir / "initial_grid.npy", initial)
        np.save(seed_dir / "ground_truth.npy", gt)
