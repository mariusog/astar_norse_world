"""Viewport capture helpers for active competition rounds.

Handles planning and executing viewport queries, recording observations,
and persisting them to disk.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.api_client import APIError, AstarClient, BudgetExhaustedError
from src.constants import TOTAL_QUERY_BUDGET
from src.query_strategy import QueryPlanner, Viewport

logger = logging.getLogger(__name__)


def capture_viewports(
    client: AstarClient,
    round_id: str,
    seeds_count: int,
    width: int,
    height: int,
    states: list,
    out_dir: Path,
) -> None:
    """Use viewport queries to observe final states (active rounds)."""
    logger.info("Starting viewport queries (budget: %d)", TOTAL_QUERY_BUDGET)
    planner = QueryPlanner(width, height)
    observations: dict[int, list[dict]] = {i: [] for i in range(seeds_count)}

    for seed_idx in range(seeds_count):
        budget_exhausted = _query_seed_viewports(
            client,
            round_id,
            seed_idx,
            states[seed_idx][0],
            planner,
            observations,
        )
        if budget_exhausted:
            break

    _save_observations(observations, out_dir)


def _query_seed_viewports(
    client: AstarClient,
    round_id: str,
    seed_idx: int,
    initial_grid: np.ndarray,
    planner: QueryPlanner,
    observations: dict[int, list[dict]],
) -> bool:
    """Query all viewports for one seed. Returns True if budget exhausted."""
    viewports = planner.plan_initial_queries(seed_idx, initial_grid)
    for vp in viewports:
        try:
            obs = _execute_viewport_query(client, round_id, vp)
            observations[seed_idx].append(obs)
            remaining = client.queries_remaining(round_id)
            logger.info(
                "Seed %d: queried (%d,%d) %dx%d -- %d remaining",
                seed_idx,
                vp.viewport_x,
                vp.viewport_y,
                vp.viewport_w,
                vp.viewport_h,
                remaining,
            )
        except BudgetExhaustedError:
            logger.warning("Budget exhausted at seed %d", seed_idx)
            return True
        except APIError as e:
            logger.warning("Query failed for seed %d: %s", seed_idx, e)
    return False


def _execute_viewport_query(
    client: AstarClient,
    round_id: str,
    vp: Viewport,
) -> dict:
    """Execute a single viewport query and return the observation dict."""
    result = client.query(
        round_id,
        vp.seed_index,
        vp.viewport_x,
        vp.viewport_y,
        vp.viewport_w,
        vp.viewport_h,
    )
    return {
        "seed_index": vp.seed_index,
        "viewport": {
            "x": vp.viewport_x,
            "y": vp.viewport_y,
            "w": vp.viewport_w,
            "h": vp.viewport_h,
        },
        "grid": result.get("grid"),
        "settlements": result.get("settlements"),
    }


def _save_observations(
    observations: dict[int, list[dict]],
    out_dir: Path,
) -> None:
    """Persist collected viewport observations to disk."""
    for seed_idx, obs_list in observations.items():
        if obs_list:
            seed_dir = out_dir / f"seed_{seed_idx}"
            seed_dir.mkdir(exist_ok=True)
            (seed_dir / "observations.json").write_text(
                json.dumps(obs_list, indent=2, default=str),
            )
    total_obs = sum(len(v) for v in observations.values())
    logger.info("Saved %d observations", total_obs)
