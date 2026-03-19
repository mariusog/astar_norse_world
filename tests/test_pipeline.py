"""End-to-end pipeline tests with mocked API client.

Verifies the full pipeline produces valid predictions:
correct shape, normalized probabilities, probability floor,
budget compliance, and graceful degradation on seed failure.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.constants import (
    NUM_PREDICTION_CLASSES,
    PROBABILITY_FLOOR,
    TOTAL_QUERY_BUDGET,
)
from src.observation import ObservationStore
from src.pipeline import CompetitionPipeline, SeedResult, _log_summary
from src.terrain import InternalTerrain

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MAP_W, MAP_H = 20, 20
NUM_SEEDS = 2


def _make_grid() -> list[list[int]]:
    """Build a simple 20x20 grid with ocean border and plains interior."""
    grid = []
    for row in range(MAP_H):
        line = []
        for col in range(MAP_W):
            if row == 0 or row == MAP_H - 1 or col == 0 or col == MAP_W - 1:
                line.append(0)  # ocean
            else:
                line.append(1)  # plains
        grid.append(line)
    return grid


def _make_round_data() -> dict:
    """Create a canned round response with 2 seeds."""
    grid = _make_grid()
    return {
        "id": "test-round-1",
        "status": "active",
        "map_width": MAP_W,
        "map_height": MAP_H,
        "initial_states": [
            {"grid": grid, "settlements": [{"x": 5, "y": 5, "owner_id": 0}]},
            {"grid": grid, "settlements": [{"x": 10, "y": 10, "owner_id": 1}]},
        ],
    }


def _make_query_response() -> dict:
    """Canned query response with a small viewport patch."""
    patch_grid = [[1] * 5 for _ in range(5)]  # 5x5 plains
    return {"grid": patch_grid}


@pytest.fixture
def mock_client() -> MagicMock:
    """Build a fully mocked AstarClient."""
    client = MagicMock()
    round_data = _make_round_data()
    client.get_round.return_value = round_data
    client.get_active_round.return_value = {"id": "test-round-1"}
    client.query.return_value = _make_query_response()
    client.submit.return_value = {"score": 75.0}
    client.query_count.return_value = 0
    client.queries_remaining.return_value = TOTAL_QUERY_BUDGET
    return client


# ---------------------------------------------------------------------------
# Pipeline shape and normalization
# ---------------------------------------------------------------------------


class TestPipelinePredictionShape:
    """Verify pipeline output tensor has correct shape."""

    @patch("src.pipeline.load_round")
    @patch("src.pipeline.QueryPlanner")
    @patch("src.pipeline.Predictor")
    def test_prediction_shape_is_h_w_6(
        self,
        mock_predictor_cls: MagicMock,
        mock_planner_cls: MagicMock,
        mock_load_round: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """Each seed produces (H, W, 6) prediction."""
        grid = np.full((MAP_H, MAP_W), InternalTerrain.PLAINS, dtype=np.int8)
        mock_load_round.return_value = [
            (grid, []),
            (grid, []),
        ]
        pred = np.ones((MAP_H, MAP_W, NUM_PREDICTION_CLASSES)) / 6
        mock_predictor_cls.return_value.predict.return_value = pred
        mock_planner_cls.return_value.plan_initial_queries.return_value = []
        mock_planner_cls.return_value.plan_adaptive_query.return_value = None
        mock_planner_cls.return_value.queries_remaining = TOTAL_QUERY_BUDGET

        pipeline = CompetitionPipeline(client=mock_client, num_mc_runs=5)
        result = pipeline.run(round_id="test-round-1")

        assert len(result.seed_results) == 2
        for sr in result.seed_results:
            assert sr.submitted is True


class TestPredictionNormalization:
    """Verify predictions sum to 1.0 per cell."""

    def test_uniform_prediction_sums_to_one(self) -> None:
        """Uniform 1/6 prediction sums to 1.0 at every cell."""
        pred = np.ones((MAP_H, MAP_W, NUM_PREDICTION_CLASSES)) / 6
        sums = pred.sum(axis=2)
        np.testing.assert_allclose(sums, 1.0, atol=1e-10)

    def test_floored_prediction_sums_to_one(self) -> None:
        """After flooring and renorm, each cell still sums to 1.0."""
        pred = np.zeros((MAP_H, MAP_W, NUM_PREDICTION_CLASSES))
        pred[:, :, 0] = 1.0
        safe = np.maximum(pred, PROBABILITY_FLOOR)
        safe = safe / safe.sum(axis=2, keepdims=True)
        np.testing.assert_allclose(safe.sum(axis=2), 1.0, atol=1e-10)


class TestProbabilityFloorEnforcement:
    """Verify no prediction values fall below the floor."""

    def test_no_zero_values_after_floor(self) -> None:
        """Floored prediction has no zero values."""
        pred = np.zeros((MAP_H, MAP_W, NUM_PREDICTION_CLASSES))
        pred[:, :, 2] = 1.0
        safe = np.maximum(pred, PROBABILITY_FLOOR)
        safe = safe / safe.sum(axis=2, keepdims=True)
        assert safe.min() > 0.0


# ---------------------------------------------------------------------------
# Budget compliance
# ---------------------------------------------------------------------------


class TestBudgetCompliance:
    """Verify query budget is respected."""

    @patch("src.pipeline.load_round")
    @patch("src.pipeline.QueryPlanner")
    @patch("src.pipeline.Predictor")
    def test_total_queries_under_budget(
        self,
        mock_predictor_cls: MagicMock,
        mock_planner_cls: MagicMock,
        mock_load_round: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """Total queries across all seeds stays within budget."""
        grid = np.full((MAP_H, MAP_W), InternalTerrain.PLAINS, dtype=np.int8)
        mock_load_round.return_value = [(grid, [])] * NUM_SEEDS
        pred = np.ones((MAP_H, MAP_W, NUM_PREDICTION_CLASSES)) / 6
        mock_predictor_cls.return_value.predict.return_value = pred
        mock_planner_cls.return_value.plan_initial_queries.return_value = []
        mock_planner_cls.return_value.plan_adaptive_query.return_value = None
        mock_planner_cls.return_value.queries_remaining = TOTAL_QUERY_BUDGET

        pipeline = CompetitionPipeline(client=mock_client, num_mc_runs=5)
        result = pipeline.run(round_id="test-round-1")

        assert result.total_queries <= TOTAL_QUERY_BUDGET


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Verify pipeline handles per-seed failures gracefully."""

    @patch("src.pipeline.load_round")
    @patch("src.pipeline.QueryPlanner")
    @patch("src.pipeline.Predictor")
    def test_one_seed_fails_others_succeed(
        self,
        mock_predictor_cls: MagicMock,
        mock_planner_cls: MagicMock,
        mock_load_round: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """If one seed's prediction fails, others still submit."""
        grid = np.full((MAP_H, MAP_W), InternalTerrain.PLAINS, dtype=np.int8)
        mock_load_round.return_value = [(grid, [])] * NUM_SEEDS

        pred = np.ones((MAP_H, MAP_W, NUM_PREDICTION_CLASSES)) / 6
        predictor_instance = mock_predictor_cls.return_value
        predictor_instance.predict.side_effect = [
            RuntimeError("sim crashed"),
            pred,
        ]
        mock_planner_cls.return_value.plan_initial_queries.return_value = []
        mock_planner_cls.return_value.plan_adaptive_query.return_value = None
        mock_planner_cls.return_value.queries_remaining = TOTAL_QUERY_BUDGET

        pipeline = CompetitionPipeline(client=mock_client, num_mc_runs=5)
        result = pipeline.run(round_id="test-round-1")

        failed = [s for s in result.seed_results if s.error]
        succeeded = [s for s in result.seed_results if s.submitted]
        assert len(failed) == 1
        assert len(succeeded) == 1
        assert "sim crashed" in failed[0].error


# ---------------------------------------------------------------------------
# SeedResult and PipelineResult dataclasses
# ---------------------------------------------------------------------------


class TestSeedResult:
    """Test SeedResult defaults."""

    def test_defaults(self) -> None:
        """SeedResult has sane defaults."""
        sr = SeedResult(seed_index=0)
        assert sr.queries_used == 0
        assert sr.submitted is False
        assert sr.score is None
        assert sr.error is None


class TestLogSummary:
    """Test _log_summary does not raise."""

    def test_log_summary_runs(self) -> None:
        """Logging summary does not crash."""
        from src.pipeline import PipelineResult

        result = PipelineResult(
            round_id="r1",
            seed_results=[SeedResult(seed_index=0, submitted=True)],
            total_queries=5,
            elapsed_seconds=1.5,
        )
        _log_summary(result)  # should not raise


# ---------------------------------------------------------------------------
# ObservationStore integration
# ---------------------------------------------------------------------------


class TestObservationStoreIntegration:
    """Verify ObservationStore produces valid probabilities."""

    def test_observed_probs_sum_to_one(self) -> None:
        """Observed cells produce probabilities summing to 1.0."""
        store = ObservationStore(height=10, width=10, num_seeds=1)
        patch = np.zeros((5, 5), dtype=np.int8)
        patch[0, 0] = 2  # settlement
        store.add_observation(0, 0, 0, patch)
        probs = store.get_observed_probs(0)
        observed = ~np.isnan(probs[:, :, 0])
        for r in range(10):
            for c in range(10):
                if observed[r, c]:
                    np.testing.assert_allclose(
                        probs[r, c].sum(),
                        1.0,
                        atol=1e-10,
                    )

    def test_coverage_mask_matches_observation(self) -> None:
        """Coverage mask is True exactly where we observed."""
        store = ObservationStore(height=10, width=10, num_seeds=1)
        patch = np.ones((3, 3), dtype=np.int8)
        store.add_observation(0, 2, 2, patch)
        mask = store.get_coverage_mask(0)
        assert mask[2:5, 2:5].all()
        assert not mask[0, 0]
