"""Tests for submission pipeline (T22)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.constants import NUM_PREDICTION_CLASSES
from src.pipeline import (
    CompetitionPipeline,
    PipelineResult,
    SeedResult,
    _execute_single_query,
    _log_summary,
)
from src.query_strategy import QueryPlanner, Viewport
from src.terrain import InternalTerrain

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client() -> MagicMock:
    """Mock AstarClient with standard responses."""
    client = MagicMock()
    client.get_active_round.return_value = {"id": "round-1"}
    client.get_round.return_value = {
        "id": "round-1",
        "map_width": 10,
        "map_height": 10,
        "seeds_count": 2,
        "initial_states": [
            _make_initial_state(10, 10),
            _make_initial_state(10, 10),
        ],
    }
    # Query returns a small grid patch
    client.query.return_value = {
        "grid": [[1] * 5 for _ in range(5)],
    }
    client.submit.return_value = {"score": 0.85}
    client.queries_remaining.return_value = 50
    return client


def _make_initial_state(w: int, h: int) -> dict:
    """Create a minimal initial state dict."""
    grid = [[int(InternalTerrain.PLAINS)] * w for _ in range(h)]
    # Ocean border
    for col in range(w):
        grid[0][col] = int(InternalTerrain.OCEAN)
        grid[h - 1][col] = int(InternalTerrain.OCEAN)
    for row in range(h):
        grid[row][0] = int(InternalTerrain.OCEAN)
        grid[row][w - 1] = int(InternalTerrain.OCEAN)
    # One settlement
    grid[5][5] = int(InternalTerrain.SETTLEMENT)
    return {
        "grid": grid,
        "settlements": [{"x": 5, "y": 5, "owner_id": 0}],
    }


# ---------------------------------------------------------------------------
# SeedResult
# ---------------------------------------------------------------------------


class TestSeedResult:
    """Tests for SeedResult dataclass."""

    def test_defaults(self) -> None:
        r = SeedResult(seed_index=0)
        assert r.queries_used == 0
        assert r.submitted is False
        assert r.score is None
        assert r.error is None


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_defaults(self) -> None:
        r = PipelineResult(round_id="r1")
        assert r.seed_results == []
        assert r.total_queries == 0
        assert r.elapsed_seconds == 0.0


# ---------------------------------------------------------------------------
# _execute_single_query
# ---------------------------------------------------------------------------


class TestExecuteSingleQuery:
    """Tests for single query execution helper."""

    def test_records_observation(self, mock_client: MagicMock) -> None:
        from src.observation import ObservationStore

        store = ObservationStore(height=10, width=10, num_seeds=2)
        planner = QueryPlanner(map_width=10, map_height=10)
        vp = Viewport(
            seed_index=0,
            viewport_x=2,
            viewport_y=2,
            viewport_w=5,
            viewport_h=5,
        )
        result = _execute_single_query(
            mock_client,
            "round-1",
            vp,
            store,
            planner,
        )
        assert result == 1
        assert store.coverage_fraction(0) > 0

    def test_returns_zero_on_budget_error(self) -> None:
        from src.api_client import BudgetExhaustedError
        from src.observation import ObservationStore

        client = MagicMock()
        client.query.side_effect = BudgetExhaustedError("over budget")
        store = ObservationStore(height=10, width=10, num_seeds=2)
        planner = QueryPlanner(map_width=10, map_height=10)
        vp = Viewport(
            seed_index=0,
            viewport_x=0,
            viewport_y=0,
            viewport_w=5,
            viewport_h=5,
        )
        result = _execute_single_query(
            client,
            "round-1",
            vp,
            store,
            planner,
        )
        assert result == 0


# ---------------------------------------------------------------------------
# CompetitionPipeline
# ---------------------------------------------------------------------------


class TestCompetitionPipeline:
    """Tests for the full pipeline."""

    @patch("src.pipeline.Predictor")
    def test_run_processes_all_seeds(
        self,
        mock_predictor_cls: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        # Mock predictor to return a valid tensor
        mock_pred = MagicMock()
        mock_pred.predict.return_value = np.full(
            (10, 10, NUM_PREDICTION_CLASSES),
            1.0 / 6.0,
        )
        mock_predictor_cls.return_value = mock_pred

        pipeline = CompetitionPipeline(mock_client, num_mc_runs=2)
        result = pipeline.run(round_id="round-1")

        assert result.round_id == "round-1"
        assert len(result.seed_results) == 2

    @patch("src.pipeline.Predictor")
    def test_run_submits_successfully(
        self,
        mock_predictor_cls: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        mock_pred = MagicMock()
        mock_pred.predict.return_value = np.full(
            (10, 10, NUM_PREDICTION_CLASSES),
            1.0 / 6.0,
        )
        mock_predictor_cls.return_value = mock_pred

        pipeline = CompetitionPipeline(mock_client, num_mc_runs=2)
        result = pipeline.run(round_id="round-1")

        for sr in result.seed_results:
            assert sr.submitted is True
            assert sr.error is None

    @patch("src.pipeline.Predictor")
    def test_run_continues_on_seed_failure(
        self,
        mock_predictor_cls: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        # First seed fails, second succeeds
        call_count = [0]

        def side_effect(*_args: object, **_kwargs: object) -> np.ndarray:
            call_count[0] += 1
            if call_count[0] == 1:
                msg = "sim error"
                raise RuntimeError(msg)
            return np.full((10, 10, NUM_PREDICTION_CLASSES), 1.0 / 6.0)

        mock_pred = MagicMock()
        mock_pred.predict.side_effect = side_effect
        mock_predictor_cls.return_value = mock_pred

        pipeline = CompetitionPipeline(mock_client, num_mc_runs=2)
        result = pipeline.run(round_id="round-1")

        assert result.seed_results[0].error is not None
        assert result.seed_results[1].submitted is True

    def test_run_finds_active_round(
        self,
        mock_client: MagicMock,
    ) -> None:
        with patch("src.pipeline.Predictor") as mock_cls:
            mock_pred = MagicMock()
            mock_pred.predict.return_value = np.full(
                (10, 10, NUM_PREDICTION_CLASSES),
                1.0 / 6.0,
            )
            mock_cls.return_value = mock_pred

            pipeline = CompetitionPipeline(mock_client, num_mc_runs=2)
            result = pipeline.run()  # No round_id

            mock_client.get_active_round.assert_called_once()
            assert result.round_id == "round-1"


# ---------------------------------------------------------------------------
# _log_summary
# ---------------------------------------------------------------------------


class TestLogSummary:
    """Tests for pipeline summary logging."""

    def test_does_not_raise(self) -> None:
        result = PipelineResult(
            round_id="r1",
            seed_results=[
                SeedResult(seed_index=0, submitted=True),
                SeedResult(seed_index=1, error="fail"),
            ],
            total_queries=10,
            elapsed_seconds=5.5,
        )
        # Should log without error
        _log_summary(result)
