"""Tests for regime detection and regime-aware prior selection."""

from __future__ import annotations

import numpy as np

from src.constants import NUM_PREDICTION_CLASSES, PROBABILITY_FLOOR
from src.regime import (
    build_distance_priors,
    build_prediction,
    build_regime_priors,
    detect_regime_from_observations,
)
from src.terrain import InternalTerrain, Terrain

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_round(
    tmp_path,
    round_name: str,
    num_seeds: int = 1,
    settlement_count: int = 30,
) -> None:
    """Create a mock round directory with grid and ground truth data.

    Args:
        tmp_path: pytest tmp_path fixture.
        round_name: subdirectory name (e.g. 'round_1').
        num_seeds: how many seed subdirectories to create.
        settlement_count: number of settlement cells in GT (controls regime).
    """
    round_dir = tmp_path / round_name
    h, w = 10, 10

    for i in range(num_seeds):
        seed_dir = round_dir / f"seed_{i}"
        seed_dir.mkdir(parents=True)

        # Build initial grid: ocean border, plains interior, settlement at (5,5)
        grid = np.full((h, w), InternalTerrain.PLAINS, dtype=np.int8)
        grid[0, :] = InternalTerrain.OCEAN
        grid[-1, :] = InternalTerrain.OCEAN
        grid[:, 0] = InternalTerrain.OCEAN
        grid[:, -1] = InternalTerrain.OCEAN
        grid[2, 2] = InternalTerrain.MOUNTAIN
        grid[5, 5] = InternalTerrain.SETTLEMENT
        grid[3, 3] = InternalTerrain.FOREST
        np.save(seed_dir / "initial_grid.npy", grid)

        # Build ground truth: H x W x 6 one-hot-ish probabilities
        gt = np.full((h, w, NUM_PREDICTION_CLASSES), 0.01, dtype=np.float64)
        # Ocean -> class 0 (EMPTY)
        gt[grid == InternalTerrain.OCEAN, Terrain.EMPTY] = 0.95
        # Mountain -> class 5
        gt[grid == InternalTerrain.MOUNTAIN, Terrain.MOUNTAIN] = 0.95
        # Plains -> class 0
        gt[grid == InternalTerrain.PLAINS, Terrain.EMPTY] = 0.90
        # Forest -> class 4
        gt[grid == InternalTerrain.FOREST, Terrain.FOREST] = 0.90

        # Place settlement_count settlement cells in GT (argmax = SETTLEMENT)
        settlement_cells_placed = 0
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if settlement_cells_placed >= settlement_count:
                    break
                # Override any interior cell to have settlement as argmax in GT
                gt[r, c, :] = 0.01
                gt[r, c, Terrain.SETTLEMENT] = 0.90
                settlement_cells_placed += 1
            if settlement_cells_placed >= settlement_count:
                break

        gt = gt / gt.sum(axis=2, keepdims=True)
        np.save(seed_dir / "ground_truth.npy", gt)


# ---------------------------------------------------------------------------
# Tests: build_regime_priors
# ---------------------------------------------------------------------------


class TestBuildRegimePriors:
    def test_returns_survive_and_collapse_keys(self, tmp_path) -> None:
        _make_mock_round(tmp_path, "round_1", settlement_count=30)
        result = build_regime_priors(str(tmp_path))
        assert "survive" in result
        assert "collapse" in result

    def test_priors_are_normalized(self, tmp_path) -> None:
        _make_mock_round(tmp_path, "round_1", settlement_count=30)
        result = build_regime_priors(str(tmp_path))
        for regime_priors in result.values():
            for vec in regime_priors.values():
                assert vec.shape == (NUM_PREDICTION_CLASSES,)
                np.testing.assert_allclose(vec.sum(), 1.0, atol=1e-6)

    def test_returns_empty_dicts_for_missing_dir(self) -> None:
        result = build_regime_priors("/nonexistent/path")
        assert result == {"survive": {}, "collapse": {}}

    def test_survive_round_populates_survive_priors(self, tmp_path) -> None:
        # >10 settlement cells in GT -> survive regime
        _make_mock_round(tmp_path, "round_1", settlement_count=30)
        result = build_regime_priors(str(tmp_path))
        assert len(result["survive"]) > 0

    def test_collapse_round_populates_collapse_priors(self, tmp_path) -> None:
        # 0 settlement cells in GT -> collapse regime
        _make_mock_round(tmp_path, "round_1", settlement_count=0)
        result = build_regime_priors(str(tmp_path))
        assert len(result["collapse"]) > 0


# ---------------------------------------------------------------------------
# Tests: build_distance_priors
# ---------------------------------------------------------------------------


class TestBuildDistancePriors:
    def test_returns_survive_and_collapse_keys(self, tmp_path) -> None:
        _make_mock_round(tmp_path, "round_1", settlement_count=30)
        result = build_distance_priors(str(tmp_path))
        assert "survive" in result
        assert "collapse" in result

    def test_distance_priors_are_normalized(self, tmp_path) -> None:
        _make_mock_round(tmp_path, "round_1", settlement_count=30)
        result = build_distance_priors(str(tmp_path))
        for regime_priors in result.values():
            for vec in regime_priors.values():
                assert vec.shape == (NUM_PREDICTION_CLASSES,)
                np.testing.assert_allclose(vec.sum(), 1.0, atol=1e-6)

    def test_keys_are_terrain_distance_tuples(self, tmp_path) -> None:
        _make_mock_round(tmp_path, "round_1", settlement_count=30)
        result = build_distance_priors(str(tmp_path))
        for regime_priors in result.values():
            for key in regime_priors:
                assert isinstance(key, tuple)
                assert len(key) == 2

    def test_returns_empty_for_missing_dir(self) -> None:
        result = build_distance_priors("/nonexistent/path")
        assert result == {"survive": {}, "collapse": {}}


# ---------------------------------------------------------------------------
# Tests: detect_regime_from_observations
# ---------------------------------------------------------------------------


class TestDetectRegimeFromObservations:
    def test_empty_observations_returns_survive(self) -> None:
        assert detect_regime_from_observations([]) == "survive"

    def test_all_settlements_returns_survive(self) -> None:
        # All observations are settlement (class 1) -> fraction = 1.0 >= 0.3
        obs = [Terrain.SETTLEMENT] * 5
        assert detect_regime_from_observations(obs) == "survive"

    def test_no_settlements_returns_collapse(self) -> None:
        # All observations are empty (class 0) -> fraction = 0.0 < 0.3
        obs = [Terrain.EMPTY] * 5
        assert detect_regime_from_observations(obs) == "collapse"

    def test_mixed_above_threshold_returns_survive(self) -> None:
        # 2 out of 5 = 0.4 >= 0.3 -> survive
        obs = [Terrain.SETTLEMENT, Terrain.SETTLEMENT, Terrain.EMPTY, Terrain.FOREST, Terrain.RUIN]
        assert detect_regime_from_observations(obs) == "survive"

    def test_mixed_below_threshold_returns_collapse(self) -> None:
        # 1 out of 5 = 0.2 < 0.3 -> collapse
        obs = [Terrain.SETTLEMENT, Terrain.EMPTY, Terrain.EMPTY, Terrain.FOREST, Terrain.RUIN]
        assert detect_regime_from_observations(obs) == "collapse"

    def test_exactly_at_threshold_returns_survive(self) -> None:
        # 3 out of 10 = 0.3 >= 0.3 -> survive
        obs = [Terrain.SETTLEMENT] * 3 + [Terrain.EMPTY] * 7
        assert detect_regime_from_observations(obs) == "survive"


# ---------------------------------------------------------------------------
# Tests: build_prediction
# ---------------------------------------------------------------------------


class TestBuildPrediction:
    def _make_simple_grid(self) -> np.ndarray:
        """5x5 grid: ocean border, plains interior, mountain at (2,2)."""
        grid = np.full((5, 5), InternalTerrain.PLAINS, dtype=np.int8)
        grid[0, :] = InternalTerrain.OCEAN
        grid[-1, :] = InternalTerrain.OCEAN
        grid[:, 0] = InternalTerrain.OCEAN
        grid[:, -1] = InternalTerrain.OCEAN
        grid[2, 2] = InternalTerrain.MOUNTAIN
        return grid

    def _make_priors(self) -> dict[str, dict[int, np.ndarray]]:
        """Simple priors: ocean->EMPTY, mountain->MOUNTAIN, plains->EMPTY."""
        plains_vec = np.array([0.85, 0.02, 0.02, 0.02, 0.07, 0.02])
        ocean_vec = np.array([0.95, 0.01, 0.01, 0.01, 0.01, 0.01])
        mountain_vec = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.95])

        priors = {
            int(InternalTerrain.PLAINS): plains_vec / plains_vec.sum(),
            int(InternalTerrain.OCEAN): ocean_vec / ocean_vec.sum(),
            int(InternalTerrain.MOUNTAIN): mountain_vec / mountain_vec.sum(),
        }
        return {"survive": priors, "collapse": priors}

    def test_returns_correct_shape(self) -> None:
        grid = self._make_simple_grid()
        regime_priors = self._make_priors()
        pred = build_prediction(grid, "survive", regime_priors)
        assert pred.shape == (5, 5, NUM_PREDICTION_CLASSES)

    def test_probabilities_sum_to_one(self) -> None:
        grid = self._make_simple_grid()
        regime_priors = self._make_priors()
        pred = build_prediction(grid, "survive", regime_priors)
        np.testing.assert_allclose(pred.sum(axis=2), 1.0, atol=1e-6)

    def test_probability_floor_applied(self) -> None:
        grid = self._make_simple_grid()
        regime_priors = self._make_priors()
        pred = build_prediction(grid, "survive", regime_priors)
        # After renormalization, minimum may be slightly below floor
        # but should be close
        assert pred.min() >= PROBABILITY_FLOOR * 0.5

    def test_ocean_predicts_empty(self) -> None:
        grid = self._make_simple_grid()
        regime_priors = self._make_priors()
        pred = build_prediction(grid, "survive", regime_priors)
        ocean_mask = grid == InternalTerrain.OCEAN
        # Ocean cells should have high prob for class 0 (EMPTY)
        ocean_empty_probs = pred[ocean_mask, Terrain.EMPTY]
        assert ocean_empty_probs.min() > 0.8

    def test_mountain_predicts_mountain(self) -> None:
        grid = self._make_simple_grid()
        regime_priors = self._make_priors()
        pred = build_prediction(grid, "survive", regime_priors)
        mtn_mask = grid == InternalTerrain.MOUNTAIN
        mtn_probs = pred[mtn_mask, Terrain.MOUNTAIN]
        assert mtn_probs.min() > 0.8

    def test_with_distance_priors(self) -> None:
        grid = self._make_simple_grid()
        regime_priors = self._make_priors()
        # Minimal distance priors dict
        dist_priors: dict[str, dict[tuple[int, int], np.ndarray]] = {
            "survive": {},
            "collapse": {},
        }
        pred = build_prediction(
            grid,
            "survive",
            regime_priors,
            distance_priors=dist_priors,
            distance_blend=0.3,
        )
        assert pred.shape == (5, 5, NUM_PREDICTION_CLASSES)
        np.testing.assert_allclose(pred.sum(axis=2), 1.0, atol=1e-6)

    def test_unknown_regime_falls_back(self) -> None:
        grid = self._make_simple_grid()
        regime_priors = self._make_priors()
        # Use a regime key that doesn't exist; should fall back to 'survive'
        pred = build_prediction(grid, "unknown_regime", regime_priors)
        assert pred.shape == (5, 5, NUM_PREDICTION_CLASSES)
        np.testing.assert_allclose(pred.sum(axis=2), 1.0, atol=1e-6)
