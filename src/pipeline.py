"""End-to-end competition pipeline from round load to submission.

Orchestrates: load round -> plan queries -> execute queries ->
predict -> submit for all 5 seeds.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.api_client import AstarClient, BudgetExhaustedError
from src.constants import (
    DEFAULT_MC_RUNS,
    TOTAL_QUERY_BUDGET,
)
from src.observation import ObservationStore
from src.predictor import Predictor
from src.query_strategy import QueryPlanner
from src.state_loader import load_round
from src.terrain import grid_to_prediction

logger = logging.getLogger(__name__)


@dataclass
class SeedResult:
    """Result of processing one seed."""

    seed_index: int
    queries_used: int = 0
    submitted: bool = False
    score: float | None = None
    error: str | None = None


@dataclass
class PipelineResult:
    """Result of running the full pipeline."""

    round_id: str
    seed_results: list[SeedResult] = field(default_factory=list)
    total_queries: int = 0
    elapsed_seconds: float = 0.0


class CompetitionPipeline:
    """End-to-end pipeline for one competition round.

    Steps per seed:
    1. Load initial state
    2. Plan and execute queries
    3. Generate prediction
    4. Submit prediction

    Args:
        client: Authenticated API client.
        num_mc_runs: Monte Carlo runs for prediction.
    """

    def __init__(
        self,
        client: AstarClient,
        num_mc_runs: int = DEFAULT_MC_RUNS,
    ) -> None:
        self._client = client
        self._num_mc_runs = num_mc_runs

    def run(self, round_id: str | None = None) -> PipelineResult:
        """Execute full pipeline for a round.

        Args:
            round_id: Specific round ID, or None to find active round.

        Returns:
            PipelineResult with per-seed outcomes.
        """
        start = time.monotonic()
        round_data = self._resolve_round(round_id)
        rid = round_data["id"]
        logger.info("Starting pipeline for round %s", rid)

        states = load_round(round_data)
        width = round_data.get("map_width", 40)
        height = round_data.get("map_height", 40)

        planner = QueryPlanner(
            map_width=width,
            map_height=height,
            total_budget=TOTAL_QUERY_BUDGET,
            num_seeds=len(states),
        )
        obs_store = ObservationStore(
            height=height,
            width=width,
            num_seeds=len(states),
        )

        result = PipelineResult(round_id=rid)
        for seed_idx in range(len(states)):
            seed_result = self._process_seed(
                rid,
                seed_idx,
                states[seed_idx],
                planner,
                obs_store,
                width,
                height,
            )
            result.seed_results.append(seed_result)
            result.total_queries += seed_result.queries_used

        result.elapsed_seconds = time.monotonic() - start
        _log_summary(result)
        return result

    def _resolve_round(
        self,
        round_id: str | None,
    ) -> dict[str, Any]:
        """Get round data from server."""
        if round_id:
            return self._client.get_round(round_id)
        active = self._client.get_active_round()
        if active is None:
            msg = "No active round found"
            raise RuntimeError(msg)
        return self._client.get_round(active["id"])

    def _process_seed(
        self,
        round_id: str,
        seed_idx: int,
        state: tuple[np.ndarray, list],
        planner: QueryPlanner,
        obs_store: ObservationStore,
        width: int,
        height: int,
    ) -> SeedResult:
        """Process one seed: query, predict, submit."""
        result = SeedResult(seed_index=seed_idx)
        grid, settlements = state

        try:
            queries_used = self._execute_queries(
                round_id,
                seed_idx,
                grid,
                planner,
                obs_store,
            )
            result.queries_used = queries_used

            prediction = self._generate_prediction(
                grid,
                settlements,
                obs_store,
                seed_idx,
            )
            response = self._client.submit(
                round_id,
                seed_idx,
                prediction,
            )
            result.submitted = True
            result.score = response.get("score")
            logger.info(
                "Seed %d submitted, score=%s",
                seed_idx,
                result.score,
            )
        except Exception as exc:
            result.error = str(exc)
            logger.error("Seed %d failed: %s", seed_idx, exc)

        return result

    def _execute_queries(
        self,
        round_id: str,
        seed_idx: int,
        grid: np.ndarray,
        planner: QueryPlanner,
        obs_store: ObservationStore,
    ) -> int:
        """Execute coverage and adaptive queries for one seed."""
        queries_used = 0
        viewports = planner.plan_initial_queries(seed_idx, grid)

        for vp in viewports:
            if planner.queries_remaining <= 0:
                break
            queries_used += _execute_single_query(
                self._client,
                round_id,
                vp,
                obs_store,
                planner,
            )

        # Adaptive queries with remaining budget
        coverage = obs_store.get_coverage_mask(seed_idx)
        adaptive_vp = planner.plan_adaptive_query(
            seed_idx,
            coverage,
            grid,
        )
        if adaptive_vp and planner.queries_remaining > 0:
            queries_used += _execute_single_query(
                self._client,
                round_id,
                adaptive_vp,
                obs_store,
                planner,
            )

        return queries_used

    def _generate_prediction(
        self,
        grid: np.ndarray,
        settlements: list,
        obs_store: ObservationStore,
        seed_idx: int,
    ) -> np.ndarray:
        """Generate prediction tensor for one seed."""
        predictor = Predictor(
            initial_grid=grid,
            settlements=settlements,
            observation_store=obs_store,
        )
        return predictor.predict(
            seed_index=seed_idx,
            num_mc_runs=self._num_mc_runs,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _execute_single_query(
    client: AstarClient,
    round_id: str,
    vp: Any,
    obs_store: ObservationStore,
    planner: QueryPlanner,
) -> int:
    """Execute one query and record the observation. Returns 1 or 0."""
    try:
        response = client.query(
            round_id,
            vp.seed_index,
            vp.viewport_x,
            vp.viewport_y,
            vp.viewport_w,
            vp.viewport_h,
        )
        planner.record_query()
        patch = np.array(response["grid"], dtype=np.int8)
        pred_patch = grid_to_prediction(patch)
        obs_store.add_observation(
            vp.seed_index,
            vp.viewport_x,
            vp.viewport_y,
            pred_patch,
        )
        return 1
    except BudgetExhaustedError:
        logger.warning("Budget exhausted during query")
        return 0


def _log_summary(result: PipelineResult) -> None:
    """Log a summary of the pipeline run."""
    submitted = sum(1 for s in result.seed_results if s.submitted)
    failed = sum(1 for s in result.seed_results if s.error)
    logger.info(
        "Pipeline complete: round=%s, seeds=%d/%d submitted, %d failed, %d queries, %.1fs",
        result.round_id,
        submitted,
        len(result.seed_results),
        failed,
        result.total_queries,
        result.elapsed_seconds,
    )
