"""Tests for query budget optimizer (T20)."""

from __future__ import annotations

import numpy as np
import pytest

from src.constants import (
    QUERIES_PER_SEED_COVERAGE,
    TOTAL_QUERY_BUDGET,
    VIEWPORT_MAX_SIZE,
    VIEWPORT_MIN_SIZE,
)
from src.query_strategy import (
    QueryPlanner,
    Viewport,
    _axis_positions,
    _compute_tiling,
    _estimate_coverage,
)
from src.terrain import InternalTerrain

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def planner() -> QueryPlanner:
    """Standard 40x40 query planner."""
    return QueryPlanner(map_width=40, map_height=40)


@pytest.fixture
def small_planner() -> QueryPlanner:
    """Small 20x20 query planner for faster tests."""
    return QueryPlanner(map_width=20, map_height=20)


@pytest.fixture
def initial_grid_40x40() -> np.ndarray:
    """A 40x40 grid with settlements and varied terrain."""
    grid = np.full((40, 40), InternalTerrain.PLAINS, dtype=np.int8)
    grid[0, :] = InternalTerrain.OCEAN
    grid[-1, :] = InternalTerrain.OCEAN
    grid[:, 0] = InternalTerrain.OCEAN
    grid[:, -1] = InternalTerrain.OCEAN
    grid[10, 10] = InternalTerrain.SETTLEMENT
    grid[20, 20] = InternalTerrain.PORT
    grid[30, 30] = InternalTerrain.SETTLEMENT
    grid[15:20, 5:8] = InternalTerrain.MOUNTAIN
    return grid


# ---------------------------------------------------------------------------
# Viewport dataclass
# ---------------------------------------------------------------------------


class TestViewport:
    """Tests for Viewport dataclass."""

    def test_to_dict_returns_all_fields(self) -> None:
        vp = Viewport(seed_index=0, viewport_x=5, viewport_y=10, viewport_w=15, viewport_h=15)
        d = vp.to_dict()
        assert d == {
            "seed_index": 0,
            "viewport_x": 5,
            "viewport_y": 10,
            "viewport_w": 15,
            "viewport_h": 15,
        }

    def test_viewport_is_frozen(self) -> None:
        vp = Viewport(seed_index=0, viewport_x=0, viewport_y=0, viewport_w=10, viewport_h=10)
        with pytest.raises(AttributeError):
            vp.seed_index = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# QueryPlanner init and budget
# ---------------------------------------------------------------------------


class TestQueryPlannerBudget:
    """Tests for QueryPlanner budget tracking."""

    def test_initial_budget(self, planner: QueryPlanner) -> None:
        assert planner.queries_remaining == TOTAL_QUERY_BUDGET

    def test_record_query_decrements(self, planner: QueryPlanner) -> None:
        planner.record_query()
        assert planner.queries_remaining == TOTAL_QUERY_BUDGET - 1

    def test_budget_reaches_zero(self) -> None:
        planner = QueryPlanner(total_budget=3)
        for _ in range(3):
            planner.record_query()
        assert planner.queries_remaining == 0


# ---------------------------------------------------------------------------
# Initial coverage queries
# ---------------------------------------------------------------------------


class TestPlanInitialQueries:
    """Tests for initial coverage query planning."""

    def test_returns_correct_count(
        self,
        planner: QueryPlanner,
        initial_grid_40x40: np.ndarray,
    ) -> None:
        viewports = planner.plan_initial_queries(0, initial_grid_40x40)
        assert len(viewports) <= QUERIES_PER_SEED_COVERAGE

    def test_viewports_within_bounds(
        self,
        planner: QueryPlanner,
        initial_grid_40x40: np.ndarray,
    ) -> None:
        viewports = planner.plan_initial_queries(0, initial_grid_40x40)
        for vp in viewports:
            assert vp.viewport_x >= 0
            assert vp.viewport_y >= 0
            assert vp.viewport_x + vp.viewport_w <= 40
            assert vp.viewport_y + vp.viewport_h <= 40

    def test_viewport_dimensions_in_range(
        self,
        planner: QueryPlanner,
        initial_grid_40x40: np.ndarray,
    ) -> None:
        viewports = planner.plan_initial_queries(0, initial_grid_40x40)
        for vp in viewports:
            assert VIEWPORT_MIN_SIZE <= vp.viewport_w <= VIEWPORT_MAX_SIZE
            assert VIEWPORT_MIN_SIZE <= vp.viewport_h <= VIEWPORT_MAX_SIZE

    def test_coverage_exceeds_85_percent(
        self,
        planner: QueryPlanner,
        initial_grid_40x40: np.ndarray,
    ) -> None:
        viewports = planner.plan_initial_queries(0, initial_grid_40x40)
        tiles = [(vp.viewport_x, vp.viewport_y, vp.viewport_w, vp.viewport_h) for vp in viewports]
        coverage = _estimate_coverage(tiles, 40, 40)
        assert coverage > 0.85, f"Coverage {coverage:.1%} < 85%"

    def test_seed_index_propagated(
        self,
        planner: QueryPlanner,
        initial_grid_40x40: np.ndarray,
    ) -> None:
        viewports = planner.plan_initial_queries(3, initial_grid_40x40)
        for vp in viewports:
            assert vp.seed_index == 3


# ---------------------------------------------------------------------------
# Adaptive queries
# ---------------------------------------------------------------------------


class TestPlanAdaptiveQuery:
    """Tests for adaptive follow-up query planning."""

    def test_returns_viewport_when_budget_available(
        self,
        planner: QueryPlanner,
        initial_grid_40x40: np.ndarray,
    ) -> None:
        coverage = np.zeros((40, 40), dtype=bool)
        result = planner.plan_adaptive_query(0, coverage, initial_grid_40x40)
        assert result is not None
        assert isinstance(result, Viewport)

    def test_returns_none_when_budget_exhausted(
        self,
        initial_grid_40x40: np.ndarray,
    ) -> None:
        planner = QueryPlanner(total_budget=0)
        coverage = np.zeros((40, 40), dtype=bool)
        result = planner.plan_adaptive_query(0, coverage, initial_grid_40x40)
        assert result is None

    def test_targets_uncovered_areas(
        self,
        planner: QueryPlanner,
        initial_grid_40x40: np.ndarray,
    ) -> None:
        # Cover top-left, leave bottom-right uncovered
        coverage = np.ones((40, 40), dtype=bool)
        coverage[25:40, 25:40] = False
        result = planner.plan_adaptive_query(0, coverage, initial_grid_40x40)
        assert result is not None
        # Viewport should be near the uncovered region
        assert result.viewport_x >= 15 or result.viewport_y >= 15

    def test_adaptive_viewport_in_bounds(
        self,
        planner: QueryPlanner,
        initial_grid_40x40: np.ndarray,
    ) -> None:
        coverage = np.zeros((40, 40), dtype=bool)
        result = planner.plan_adaptive_query(0, coverage, initial_grid_40x40)
        assert result is not None
        assert result.viewport_x + result.viewport_w <= 40
        assert result.viewport_y + result.viewport_h <= 40

    def test_adaptive_viewport_dimensions_in_range(
        self,
        planner: QueryPlanner,
        initial_grid_40x40: np.ndarray,
    ) -> None:
        coverage = np.zeros((40, 40), dtype=bool)
        result = planner.plan_adaptive_query(0, coverage, initial_grid_40x40)
        assert result is not None
        assert VIEWPORT_MIN_SIZE <= result.viewport_w <= VIEWPORT_MAX_SIZE
        assert VIEWPORT_MIN_SIZE <= result.viewport_h <= VIEWPORT_MAX_SIZE


# ---------------------------------------------------------------------------
# Tiling helpers
# ---------------------------------------------------------------------------


class TestTilingHelpers:
    """Tests for internal tiling functions."""

    def test_axis_positions_small_map(self) -> None:
        positions = _axis_positions(10, 15)
        assert positions == [0]

    def test_axis_positions_exact_multiple(self) -> None:
        positions = _axis_positions(30, 15)
        assert len(positions) == 2
        assert positions[0] == 0

    def test_axis_positions_40(self) -> None:
        positions = _axis_positions(40, 15)
        assert len(positions) == 3
        # First tile starts at 0
        assert positions[0] == 0

    def test_compute_tiling_respects_budget(self) -> None:
        tiles = _compute_tiling(40, 40, max_queries=4)
        assert len(tiles) <= 4

    def test_compute_tiling_all_in_bounds(self) -> None:
        tiles = _compute_tiling(40, 40, max_queries=9)
        for x, y, w, h in tiles:
            assert x >= 0 and y >= 0
            assert x + w <= 40
            assert y + h <= 40
            assert VIEWPORT_MIN_SIZE <= w <= VIEWPORT_MAX_SIZE
            assert VIEWPORT_MIN_SIZE <= h <= VIEWPORT_MAX_SIZE

    def test_estimate_coverage_full(self) -> None:
        tiles = [(0, 0, 10, 10)]
        coverage = _estimate_coverage(tiles, 10, 10)
        assert coverage == pytest.approx(1.0)

    def test_estimate_coverage_partial(self) -> None:
        tiles = [(0, 0, 5, 5)]
        coverage = _estimate_coverage(tiles, 10, 10)
        assert coverage == pytest.approx(0.25)
