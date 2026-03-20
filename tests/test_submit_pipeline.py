"""End-to-end tests for submit_v2 pipeline: observation blending,
validation checks, and prediction correctness.
"""

from __future__ import annotations

import numpy as np

from src.constants import (
    NUM_PREDICTION_CLASSES,
    PROBABILITY_FLOOR,
)
from src.observation import ObservationStore
from src.terrain import SERVER_TO_PRED_CLASS, InternalTerrain

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grid(h: int = 10, w: int = 10) -> np.ndarray:
    """Create a small test grid with mixed terrain."""
    grid = np.full((h, w), InternalTerrain.PLAINS, dtype=np.int8)
    grid[0, :] = InternalTerrain.OCEAN
    grid[-1, :] = InternalTerrain.OCEAN
    grid[:, 0] = InternalTerrain.OCEAN
    grid[:, -1] = InternalTerrain.OCEAN
    grid[3, 3] = InternalTerrain.SETTLEMENT
    grid[3, 4] = InternalTerrain.PORT
    grid[5, 5] = InternalTerrain.FOREST
    grid[6, 6] = InternalTerrain.MOUNTAIN
    grid[4, 3] = InternalTerrain.RUIN
    return grid


def _make_priors() -> np.ndarray:
    """Create non-uniform priors for 7 terrain types."""
    priors = np.full((7, NUM_PREDICTION_CLASSES), 1.0 / 6, dtype=np.float64)
    # Ocean: stays ocean
    priors[InternalTerrain.OCEAN] = [0.98, 0.004, 0.004, 0.004, 0.004, 0.004]
    # Plains: mostly empty
    priors[InternalTerrain.PLAINS] = [0.60, 0.10, 0.05, 0.05, 0.15, 0.05]
    # Settlement: likely settlement or ruin
    priors[InternalTerrain.SETTLEMENT] = [0.05, 0.50, 0.15, 0.20, 0.05, 0.05]
    # Mountain: stays mountain
    priors[InternalTerrain.MOUNTAIN] = [0.004, 0.004, 0.004, 0.004, 0.004, 0.98]
    # Forest: mostly forest
    priors[InternalTerrain.FOREST] = [0.15, 0.05, 0.05, 0.05, 0.65, 0.05]
    return priors


def _make_server_patch() -> np.ndarray:
    """Create a server-coded viewport patch (3x3)."""
    return np.array(
        [
            [10, 0, 0],  # ocean, empty/plains, empty/plains
            [0, 1, 2],  # empty/plains, settlement, port
            [0, 4, 5],  # empty/plains, forest, mountain
        ],
        dtype=np.int32,
    )


# ---------------------------------------------------------------------------
# T101: Observation blending end-to-end
# ---------------------------------------------------------------------------


class TestObservationBlending:
    """Test that server codes flow through ObservationStore correctly."""

    def test_server_codes_map_to_pred_classes(self) -> None:
        """Verify SERVER_TO_PRED_CLASS covers all server codes."""
        expected = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}
        for code, cls in expected.items():
            assert SERVER_TO_PRED_CLASS[code] == cls

    def test_observation_store_with_server_codes(self) -> None:
        """ObservationStore correctly accumulates server terrain codes."""
        store = ObservationStore(height=10, width=10, num_seeds=1)
        patch = _make_server_patch()
        store.add_observation(0, 1, 1, patch)

        probs = store.get_observed_probs(0)
        # Cell (1,1) got server code 10 (ocean) -> pred class 0
        assert probs[1, 1, 0] > 0.5
        # Cell (2,2) got server code 1 (settlement) -> pred class 1
        assert probs[2, 2, 1] > 0.5
        # Cell (2,3) got server code 2 (port) -> pred class 2
        assert probs[2, 3, 2] > 0.5

    def test_blend_improves_over_priors(self) -> None:
        """Blending observations with priors improves prediction."""
        from scripts.submit_v2 import _blend_observations

        grid = _make_grid()
        priors = _make_priors()
        h, w = grid.shape

        # Build prior-only prediction
        gi = np.clip(grid.astype(np.int32), 0, 6)
        prior_tensor = priors[gi].copy()

        # Add observation: settlement at (3,3) confirmed as settlement
        store = ObservationStore(height=h, width=w, num_seeds=1)
        patch = np.array([[1]], dtype=np.int32)  # server code 1 = settlement
        store.add_observation(0, 3, 3, patch)

        blended = _blend_observations(prior_tensor.copy(), store, 0)

        # Blended settlement cell should have higher class-1 prob
        prior_s1 = prior_tensor[3, 3, 1]
        blended_s1 = blended[3, 3, 1]
        assert blended_s1 > prior_s1

    def test_unobserved_cells_keep_priors(self) -> None:
        """Cells without observations keep their prior predictions."""
        from scripts.submit_v2 import _blend_observations

        priors = _make_priors()
        grid = _make_grid()
        gi = np.clip(grid.astype(np.int32), 0, 6)
        prior_tensor = priors[gi].copy()

        store = ObservationStore(height=10, width=10, num_seeds=1)
        # Only observe one cell
        store.add_observation(0, 3, 3, np.array([[1]], dtype=np.int32))

        blended = _blend_observations(prior_tensor.copy(), store, 0)
        # Unobserved cell (5,5) should match prior
        np.testing.assert_array_almost_equal(blended[5, 5], prior_tensor[5, 5])

    def test_multiple_observations_increase_weight(self) -> None:
        """More observations -> higher observation weight in blend."""
        from scripts.submit_v2 import _blend_observations

        priors = _make_priors()
        grid = _make_grid()
        gi = np.clip(grid.astype(np.int32), 0, 6)

        store1 = ObservationStore(height=10, width=10, num_seeds=1)
        store1.add_observation(0, 3, 3, np.array([[4]], dtype=np.int32))

        store5 = ObservationStore(height=10, width=10, num_seeds=1)
        for _ in range(5):
            store5.add_observation(0, 3, 3, np.array([[4]], dtype=np.int32))

        b1 = _blend_observations(priors[gi].copy(), store1, 0)
        b5 = _blend_observations(priors[gi].copy(), store5, 0)

        # 5 observations should push forest class higher than 1
        assert b5[3, 3, 4] > b1[3, 3, 4]



# ---------------------------------------------------------------------------
# T100: Distance priors integration
# ---------------------------------------------------------------------------


class TestDistancePriorIntegration:
    """Test that predict_from_priors uses distance priors when available."""

    def test_predict_with_dist_priors_differs(self) -> None:
        """Distance priors produce different results than flat priors."""
        from src.unified_priors import predict_from_priors

        grid = _make_grid()
        priors = _make_priors()

        # Create synthetic distance priors (7, 5, 6)
        # Make them different from flat priors
        dp = np.tile(priors[:, np.newaxis, :], (1, 5, 1))
        # Shift near-settlement bin to favor settlement outcome
        dp[:, 0, 1] += 0.1
        dp[:, 0] /= dp[:, 0].sum(axis=1, keepdims=True)

        pred_flat = predict_from_priors(grid, priors, None)
        pred_dist = predict_from_priors(grid, priors, dp)

        # Some non-static cells should differ
        dynamic = (grid != InternalTerrain.OCEAN) & (grid != InternalTerrain.MOUNTAIN)
        flat_dyn = pred_flat[dynamic]
        dist_dyn = pred_dist[dynamic]
        assert not np.allclose(flat_dyn, dist_dyn)

    def test_predict_without_dist_priors_uses_flat(self) -> None:
        """Without dist priors, falls back to flat terrain priors."""
        from src.unified_priors import predict_from_priors

        grid = _make_grid()
        priors = _make_priors()
        pred = predict_from_priors(grid, priors, None)

        # Check shape
        assert pred.shape == (10, 10, NUM_PREDICTION_CLASSES)
        # All rows should sum to ~1.0
        sums = pred.sum(axis=2)
        np.testing.assert_array_almost_equal(sums, 1.0, decimal=5)

    def test_static_overrides_applied(self) -> None:
        """Ocean and mountain cells get high-confidence overrides."""
        from src.unified_priors import predict_from_priors

        grid = _make_grid()
        priors = _make_priors()
        pred = predict_from_priors(grid, priors, None)

        # Ocean cells: class 0 should be high
        ocean_mask = grid == InternalTerrain.OCEAN
        assert pred[ocean_mask, 0].min() > 0.9

        # Mountain cells: class 5 should be high
        mtn_mask = grid == InternalTerrain.MOUNTAIN
        assert pred[mtn_mask, 5].min() > 0.9

    def test_floor_applied(self) -> None:
        """No probability value should be below the floor."""
        from src.unified_priors import predict_from_priors

        grid = _make_grid()
        priors = _make_priors()
        pred = predict_from_priors(grid, priors, None)
        assert pred.min() >= PROBABILITY_FLOOR - 1e-9
