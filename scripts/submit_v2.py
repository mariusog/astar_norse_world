# LEGACY: Superseded by scripts/submit_v3.py. Kept for test compatibility.
"""V2 submission pipeline using feature-based per-cell predictions.

Replaces flat terrain priors with regime-adaptive feature lookup,
then blends with server observations.

Usage:
    python scripts/submit_v2.py --token <JWT> [--round-id <ID>]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any

import numpy as np

from src.adaptive_priors import build_adaptive_feature_lookup
from src.api_client import AstarClient, BudgetExhaustedError
from src.constants import (
    DEFAULT_MAP_HEIGHT,
    DEFAULT_MAP_WIDTH,
    OBS_CONFIDENCE_K,
    PROBABILITY_FLOOR,
    STATIC_TERRAIN_CONFIDENCE,
    TOTAL_QUERY_BUDGET,
)
from src.feature_predictor import FeatureLookup, predict_from_features
from src.observation import ObservationStore
from src.query_strategy import QueryPlanner
from src.state_loader import load_round
from src.terrain import InternalTerrain

logger = logging.getLogger(__name__)

_MAX_OBS_WEIGHT = 0.8


def main() -> int:
    """Run the v2 pipeline with feature-based predictions."""
    args = _parse_args()
    _setup_logging(args.verbose)
    client = AstarClient(token=args.token)
    round_data = _resolve_round(client, args.round_id)
    regime = _detect_regime(round_data)
    lookup = build_adaptive_feature_lookup(regime, args.data_dir)
    return _run_pipeline(client, round_data, lookup)


def _run_pipeline(
    client: AstarClient,
    round_data: dict[str, Any],
    lookup: FeatureLookup,
) -> int:
    """Execute full pipeline for all seeds."""
    start = time.monotonic()
    rid = round_data["id"]
    states = load_round(round_data)
    planner, obs_store = _build_components(round_data, len(states))
    ok_count = 0
    for seed_idx in range(len(states)):
        grid, _settlements = states[seed_idx]
        ok_count += _process_seed(
            client,
            rid,
            seed_idx,
            grid,
            planner,
            obs_store,
            lookup,
        )
    elapsed = time.monotonic() - start
    logger.info("V2 pipeline: %d/%d submitted in %.1fs", ok_count, len(states), elapsed)
    return 0 if ok_count == len(states) else 1


def _process_seed(
    client: AstarClient,
    round_id: str,
    seed_idx: int,
    grid: np.ndarray,
    planner: QueryPlanner,
    obs_store: ObservationStore,
    lookup: FeatureLookup,
) -> int:
    """Process one seed: query, predict, submit. Returns 1 on success."""
    try:
        _execute_queries(client, round_id, seed_idx, grid, planner, obs_store)
        prediction = _build_prediction(grid, seed_idx, obs_store, lookup)
        client.submit(round_id, seed_idx, prediction)
        logger.info("Seed %d submitted", seed_idx)
        return 1
    except Exception as exc:
        logger.error("Seed %d failed: %s", seed_idx, exc)
        return 0


def _build_prediction(
    grid: np.ndarray,
    seed_idx: int,
    obs_store: ObservationStore,
    lookup: FeatureLookup,
) -> np.ndarray:
    """Build prediction using feature lookup + observation blending."""
    base = predict_from_features(grid, lookup)
    base = _blend_observations(base, obs_store, seed_idx)
    base = _apply_static_overrides(base, grid)
    return _floor_and_normalize(base)


def _blend_observations(
    tensor: np.ndarray,
    obs_store: ObservationStore,
    seed_idx: int,
) -> np.ndarray:
    """Blend server observations into feature-based predictions."""
    obs_probs = obs_store.get_observed_probs(seed_idx)
    coverage = obs_store.get_coverage_mask(seed_idx)
    obs_counts = obs_store.observation_count(seed_idx)
    observed = coverage & ~np.isnan(obs_probs[:, :, 0])
    if not observed.any():
        return tensor
    result = tensor.copy()
    counts = obs_counts[observed].astype(np.float64)
    w_obs = (_MAX_OBS_WEIGHT * counts / (counts + OBS_CONFIDENCE_K))[:, np.newaxis]
    result[observed] = w_obs * obs_probs[observed] + (1.0 - w_obs) * tensor[observed]
    return result


def _apply_static_overrides(
    tensor: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Override ocean/mountain with near-certain probabilities."""
    result = tensor.copy()
    residual = 1.0 - STATIC_TERRAIN_CONFIDENCE
    per_class = residual / 5
    for terrain, cls_idx in [(InternalTerrain.OCEAN, 0), (InternalTerrain.MOUNTAIN, 5)]:
        mask = grid == terrain
        if mask.any():
            result[mask] = per_class
            result[mask, cls_idx] = STATIC_TERRAIN_CONFIDENCE
    return result


def _floor_and_normalize(tensor: np.ndarray) -> np.ndarray:
    """Apply probability floor and renormalize."""
    safe = np.maximum(tensor, PROBABILITY_FLOOR)
    return safe / safe.sum(axis=2, keepdims=True)


def _detect_regime(round_data: dict[str, Any]) -> str:
    """Detect round regime from metadata. Defaults to survive."""
    rnum = round_data.get("round_number", 1)
    if rnum in (3, 4):
        return "collapse"
    if rnum in (5, 6):
        return "aggressive"
    return "survive"


def _resolve_round(
    client: AstarClient,
    round_id: str | None,
) -> dict[str, Any]:
    """Get round data from server."""
    if round_id:
        return client.get_round(round_id)
    active = client.get_active_round()
    if active is None:
        msg = "No active round found"
        raise RuntimeError(msg)
    return client.get_round(active["id"])


def _build_components(
    round_data: dict[str, Any],
    num_seeds: int,
) -> tuple[QueryPlanner, ObservationStore]:
    """Create planner and observation store."""
    width = round_data.get("map_width", DEFAULT_MAP_WIDTH)
    height = round_data.get("map_height", DEFAULT_MAP_HEIGHT)
    planner = QueryPlanner(
        map_width=width,
        map_height=height,
        total_budget=TOTAL_QUERY_BUDGET,
        num_seeds=num_seeds,
    )
    obs_store = ObservationStore(height=height, width=width, num_seeds=num_seeds)
    return planner, obs_store


def _execute_queries(
    client: AstarClient,
    round_id: str,
    seed_idx: int,
    grid: np.ndarray,
    planner: QueryPlanner,
    obs_store: ObservationStore,
) -> None:
    """Execute coverage + adaptive queries for one seed."""
    viewports = planner.plan_initial_queries(seed_idx, grid)
    for vp in viewports:
        if planner.queries_remaining <= 0:
            break
        _do_query(client, round_id, vp, obs_store, planner)
    coverage = obs_store.get_coverage_mask(seed_idx)
    adaptive_vp = planner.plan_adaptive_query(seed_idx, coverage, grid)
    if adaptive_vp and planner.queries_remaining > 0:
        _do_query(client, round_id, adaptive_vp, obs_store, planner)


def _do_query(
    client: AstarClient,
    round_id: str,
    vp: Any,
    obs_store: ObservationStore,
    planner: QueryPlanner,
) -> None:
    """Execute one viewport query."""
    try:
        resp = client.query(
            round_id,
            vp.seed_index,
            vp.viewport_x,
            vp.viewport_y,
            vp.viewport_w,
            vp.viewport_h,
        )
        planner.record_query()
        patch = np.array(resp["grid"], dtype=np.int32)
        obs_store.add_observation(vp.seed_index, vp.viewport_x, vp.viewport_y, patch)
    except BudgetExhaustedError:
        logger.warning("Budget exhausted")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="V2 submission pipeline")
    parser.add_argument("--token", required=True, help="JWT token")
    parser.add_argument("--round-id", default=None, help="Round ID")
    parser.add_argument(
        "--data-dir",
        default="data/rounds",
        help="Historical data directory",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _setup_logging(verbose: bool) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


if __name__ == "__main__":
    sys.exit(main())
