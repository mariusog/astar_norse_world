"""Submit predictions for an active competition round.

Strategy:
1. Probe settlement cells to detect regime (survive vs collapse)
2. Build regime-weighted priors from historical data
3. Apply distance-aware refinements
4. Use remaining queries to observe dynamic cells
5. Blend observations into predictions
6. Submit all 5 seeds

Usage:
    python scripts/submit_round.py --token <JWT> [--round-id <ID>]
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from src.api_client import APIError, AstarClient, BudgetExhaustedError
from src.constants import (
    NUM_SEEDS,
    OBS_CONFIDENCE_K,
    PROBABILITY_FLOOR,
    TOTAL_QUERY_BUDGET,
)
from src.features import compute_settlement_distance
from src.regime import (
    build_prediction,
    build_regime_priors,
    detect_regime_from_observations,
)
from src.state_loader import load_round
from src.terrain import SERVER_TO_PRED_CLASS, InternalTerrain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROBE_VIEWPORT_SIZE = 5
OBSERVE_VIEWPORT_SIZE = 15
QUERIES_FOR_PROBES = 2  # per seed
SETTLEMENT_PROBE_RADIUS = 5


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit predictions")
    parser.add_argument("--token", required=True, help="JWT token")
    parser.add_argument("--round-id", help="Round ID (default: active)")
    parser.add_argument("--dry-run", action="store_true", help="Score locally, don't submit")
    args = parser.parse_args()

    client = AstarClient(args.token)
    round_id = args.round_id or _find_active_round(client)
    if not round_id:
        logger.error("No active round found")
        return

    round_data = client.get_round(round_id)
    states = load_round(round_data)
    w, h = round_data.get("map_width", 40), round_data.get("map_height", 40)
    logger.info("Round %s: %dx%d, %d seeds", round_id, w, h, len(states))

    # Load historical priors
    regime_priors = build_regime_priors("data/rounds")
    logger.info(
        "Loaded regime priors: survive=%d, collapse=%d terrain types",
        len(regime_priors.get("survive", {})),
        len(regime_priors.get("collapse", {})),
    )

    queries_per_seed = TOTAL_QUERY_BUDGET // NUM_SEEDS
    total_used = 0

    for seed_idx in range(len(states)):
        grid, _settlements = states[seed_idx]
        logger.info("--- Seed %d ---", seed_idx)

        # Phase 1: Probe settlements to detect regime
        regime, probe_used = _probe_regime(
            client,
            round_id,
            seed_idx,
            grid,
            h,
            w,
        )
        total_used += probe_used

        # Phase 2: Build prediction from priors
        pred = build_prediction(grid, regime, regime_priors)

        # Phase 3: Observe dynamic cells
        remaining = min(queries_per_seed - probe_used, TOTAL_QUERY_BUDGET - total_used)
        pred, obs_used = _observe_and_blend(
            client,
            round_id,
            seed_idx,
            grid,
            pred,
            remaining,
            h,
            w,
        )
        total_used += obs_used

        # Phase 4: Submit
        if args.dry_run:
            queries = probe_used + obs_used
            logger.info("Seed %d: DRY RUN (regime=%s, queries=%d)", seed_idx, regime, queries)
        else:
            _submit_seed(client, round_id, seed_idx, pred)

    logger.info("Total queries used: %d/%d", total_used, TOTAL_QUERY_BUDGET)


def _find_active_round(client: AstarClient) -> str | None:
    active = client.get_active_round()
    return active["id"] if active else None


def _probe_regime(
    client: AstarClient,
    round_id: str,
    seed_idx: int,
    grid: np.ndarray,
    h: int,
    w: int,
) -> tuple[str, int]:
    """Probe settlement cells to detect survive/collapse regime."""
    ys, xs = np.where((grid == InternalTerrain.SETTLEMENT) | (grid == InternalTerrain.PORT))
    settle_cells = list(zip(ys, xs, strict=True))
    if not settle_cells:
        return "collapse", 0

    # Pick 2 settlement cells to probe with small viewports
    obs_classes: list[int] = []
    queries_used = 0
    for cy, cx in settle_cells[:QUERIES_FOR_PROBES]:
        vy = max(0, min(h - PROBE_VIEWPORT_SIZE, cy - PROBE_VIEWPORT_SIZE // 2))
        vx = max(0, min(w - PROBE_VIEWPORT_SIZE, cx - PROBE_VIEWPORT_SIZE // 2))
        try:
            result = client.query(
                round_id,
                seed_idx,
                vx,
                vy,
                PROBE_VIEWPORT_SIZE,
                PROBE_VIEWPORT_SIZE,
            )
            queries_used += 1
            # Extract the class at the settlement cell
            local_y, local_x = cy - vy, cx - vx
            server_code = result["grid"][local_y][local_x]
            pred_class = SERVER_TO_PRED_CLASS.get(server_code, 0)
            obs_classes.append(pred_class)
            time.sleep(0.2)
        except (BudgetExhaustedError, APIError) as e:
            logger.warning("Probe failed: %s", e)
            break

    regime = detect_regime_from_observations(obs_classes)
    return regime, queries_used


def _observe_and_blend(
    client: AstarClient,
    round_id: str,
    seed_idx: int,
    grid: np.ndarray,
    pred: np.ndarray,
    budget: int,
    h: int,
    w: int,
) -> tuple[np.ndarray, int]:
    """Query dynamic cells and blend observations into prediction."""
    if budget <= 0:
        return pred, 0

    dist = compute_settlement_distance(grid)
    obs_counts = np.zeros((h, w), dtype=int)
    obs_sums = np.zeros((h, w, 6), dtype=float)
    queries_used = 0

    # Target viewports on areas near settlements
    targets = _plan_observation_viewports(grid, dist, budget, h, w)

    for vx, vy, vw, vh in targets:
        try:
            result = client.query(round_id, seed_idx, vx, vy, vw, vh)
            queries_used += 1
            _record_observation(result, vx, vy, vw, vh, obs_counts, obs_sums)
            time.sleep(0.2)
        except (BudgetExhaustedError, APIError) as e:
            logger.warning("Observation query failed: %s", e)
            break

    # Blend observations into prediction
    observed = obs_counts > 0
    if observed.any():
        pred = _blend_observations(pred, obs_counts, obs_sums, observed)

    logger.info(
        "Seed %d: %d observation queries, %.0f%% coverage",
        seed_idx,
        queries_used,
        observed.sum() / (h * w) * 100,
    )
    return pred, queries_used


def _plan_observation_viewports(
    grid: np.ndarray,
    dist: np.ndarray,
    budget: int,
    h: int,
    w: int,
) -> list[tuple[int, int, int, int]]:
    """Plan viewports targeting dynamic cells near settlements."""
    # Find settlement cluster centers
    settle_mask = (grid == InternalTerrain.SETTLEMENT) | (grid == InternalTerrain.PORT)
    settle_ys, settle_xs = np.where(settle_mask)

    viewports = []
    if len(settle_ys) == 0:
        # No settlements: tile with large viewports
        for vy in range(0, h, OBSERVE_VIEWPORT_SIZE):
            for vx in range(0, w, OBSERVE_VIEWPORT_SIZE):
                vw = min(OBSERVE_VIEWPORT_SIZE, w - vx)
                vh = min(OBSERVE_VIEWPORT_SIZE, h - vy)
                viewports.append((vx, vy, max(5, vw), max(5, vh)))
        return viewports[:budget]

    # Center viewports on settlement clusters with overlap
    for cy, cx in zip(settle_ys, settle_xs, strict=True):
        if len(viewports) >= budget:
            break
        vy = max(0, min(h - OBSERVE_VIEWPORT_SIZE, cy - OBSERVE_VIEWPORT_SIZE // 2))
        vx = max(0, min(w - OBSERVE_VIEWPORT_SIZE, cx - OBSERVE_VIEWPORT_SIZE // 2))
        viewports.append((vx, vy, OBSERVE_VIEWPORT_SIZE, OBSERVE_VIEWPORT_SIZE))

    # If we have budget left, add shifted viewports for overlap
    if len(viewports) < budget:
        for cy, cx in zip(settle_ys, settle_xs, strict=True):
            if len(viewports) >= budget:
                break
            # Offset by half viewport for overlap
            vy = max(0, min(h - OBSERVE_VIEWPORT_SIZE, cy - 3))
            vx = max(0, min(w - OBSERVE_VIEWPORT_SIZE, cx - 3))
            viewports.append((vx, vy, OBSERVE_VIEWPORT_SIZE, OBSERVE_VIEWPORT_SIZE))

    return viewports[:budget]


def _record_observation(
    result: dict,
    vx: int,
    vy: int,
    vw: int,
    vh: int,
    obs_counts: np.ndarray,
    obs_sums: np.ndarray,
) -> None:
    """Record a single viewport observation into counts arrays."""
    grid_data = result.get("grid", [])
    for row in range(min(vh, len(grid_data))):
        for col in range(min(vw, len(grid_data[row]) if row < len(grid_data) else 0)):
            gy, gx = vy + row, vx + col
            if 0 <= gy < obs_counts.shape[0] and 0 <= gx < obs_counts.shape[1]:
                server_code = grid_data[row][col]
                pred_class = SERVER_TO_PRED_CLASS.get(server_code, 0)
                obs_counts[gy, gx] += 1
                obs_sums[gy, gx, pred_class] += 1


def _blend_observations(
    pred: np.ndarray,
    obs_counts: np.ndarray,
    obs_sums: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    """Blend observations into prediction with count-scaled weights."""
    result = pred.copy()
    counts = obs_counts[observed].astype(float)
    obs_probs = (obs_sums[observed] + 0.01) / (counts[:, np.newaxis] + 0.06)
    w_obs = (0.8 * counts / (counts + OBS_CONFIDENCE_K))[:, np.newaxis]
    result[observed] = (1 - w_obs) * result[observed] + w_obs * obs_probs
    result = np.maximum(result, PROBABILITY_FLOOR)
    result = result / result.sum(axis=2, keepdims=True)
    return result


def _submit_seed(
    client: AstarClient,
    round_id: str,
    seed_idx: int,
    pred: np.ndarray,
) -> None:
    """Submit prediction for one seed."""
    try:
        resp = client.submit(round_id, seed_idx, pred)
        logger.info("Seed %d: SUBMITTED — %s", seed_idx, resp.get("status"))
    except APIError as e:
        logger.error("Seed %d: FAILED — %s", seed_idx, e)


if __name__ == "__main__":
    main()
