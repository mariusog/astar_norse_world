"""Submit predictions v2: survive priors + observation blending, no regime detection."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any

import numpy as np

from src.api_client import APIError, AstarClient
from src.constants import (
    DEFAULT_MAP_HEIGHT,
    DEFAULT_MAP_WIDTH,
    NUM_PREDICTION_CLASSES,
    NUM_SEEDS,
    OBS_CONFIDENCE_K,
    PROBABILITY_FLOOR,
    STATIC_TERRAIN_CONFIDENCE,
    TOTAL_QUERY_BUDGET,
)
from src.observation import ObservationStore
from src.state_loader import load_round
from src.terrain import InternalTerrain

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
_MAX_OBS_WEIGHT = 0.8
_VP_SIZE = 15
_VP_OFFSET = 5
_QUERY_DELAY = 0.2


def main() -> None:
    """Parse CLI args and run the full pipeline."""
    args = _parse_args()
    client = AstarClient(args.token)
    round_id, rd = _resolve_round(client, args.round_id)
    states = load_round(rd)
    h, w = rd.get("map_height", DEFAULT_MAP_HEIGHT), rd.get("map_width", DEFAULT_MAP_WIDTH)
    n_seeds = rd.get("seeds_count", NUM_SEEDS)
    logger.info("Round %s: %dx%d, %d seeds", round_id, w, h, n_seeds)
    priors = _load_survive_priors()
    viewports = _plan_all_queries(states, w, h, args.budget)
    obs = ObservationStore(h, w, n_seeds)
    if not args.dry_run:
        _execute_queries(client, round_id, viewports, obs)
    for si in range(n_seeds):
        grid, _ = states[si]
        pred = _build_prediction(grid, priors, obs, si)
        _submit_seed(client, round_id, si, pred, args.dry_run)
    logger.info("Pipeline complete for round %s", round_id)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit predictions v2")
    p.add_argument("--token", required=True, help="JWT auth token")
    p.add_argument("--round-id", default=None, help="Round ID (default: active)")
    p.add_argument("--budget", type=int, default=TOTAL_QUERY_BUDGET)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _resolve_round(client: AstarClient, rid: str | None) -> tuple[str, dict[str, Any]]:
    """Find the active round or fetch the specified one."""
    if rid:
        return rid, client.get_round(rid)
    active = client.get_active_round()
    if active is None:
        logger.error("No active round found")
        sys.exit(1)
    return active["id"], client.get_round(active["id"])


def _load_survive_priors() -> np.ndarray:
    """Load survive-weighted (7, 6) terrain priors from historical data."""
    from src.unified_priors import build_unified_priors

    priors, _dist = build_unified_priors("data/rounds")
    return priors


def _find_settlements(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return (x, y) of settlement and port cells."""
    ys, xs = np.where((grid == InternalTerrain.SETTLEMENT) | (grid == InternalTerrain.PORT))
    return list(zip(xs.tolist(), ys.tolist(), strict=True))


def _dedupe_centers(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Reduce settlement cells to spread-out cluster centers."""
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for cx, cy in sorted(cells, key=lambda c: (c[1], c[0])):
        key = (cx // _VP_OFFSET, cy // _VP_OFFSET)
        if key not in seen:
            seen.add(key)
            out.append((cx, cy))
    return out


def _viewports_around(
    cx: int, cy: int, w: int, h: int, max_n: int, existing: list
) -> list[dict[str, int]]:
    """Generate overlapping viewports around a center point."""
    half, vps = _VP_SIZE // 2, []
    for dy in [0, -_VP_OFFSET, _VP_OFFSET]:
        for dx in [0, -_VP_OFFSET, _VP_OFFSET]:
            if len(vps) >= max_n:
                return vps
            vx = max(0, min(cx - half + dx, w - _VP_SIZE))
            vy = max(0, min(cy - half + dy, h - _VP_SIZE))
            vp = {"x": vx, "y": vy, "w": _VP_SIZE, "h": _VP_SIZE}
            if vp not in existing and vp not in vps:
                vps.append(vp)
    return vps


def _plan_seed_viewports(grid: np.ndarray, w: int, h: int, budget: int) -> list[dict[str, int]]:
    """Plan overlapping viewports centered on settlement clusters."""
    cells = _find_settlements(grid)
    if not cells:
        return _tile_viewports(w, h, budget)
    vps: list[dict[str, int]] = []
    for cx, cy in _dedupe_centers(cells):
        if len(vps) >= budget:
            break
        vps.extend(_viewports_around(cx, cy, w, h, budget - len(vps), vps))
    return vps[:budget]


def _tile_viewports(w: int, h: int, budget: int) -> list[dict[str, int]]:
    """Fallback: tile the map when no settlements found."""
    vps: list[dict[str, int]] = []
    for y in range(0, h, _VP_SIZE):
        for x in range(0, w, _VP_SIZE):
            if len(vps) < budget:
                vw, vh = max(5, min(_VP_SIZE, w - x)), max(5, min(_VP_SIZE, h - y))
                vps.append({"x": x, "y": y, "w": vw, "h": vh})
    return vps[:budget]


def _plan_all_queries(
    states: list[tuple[np.ndarray, list]], w: int, h: int, budget: int
) -> list[dict[str, Any]]:
    """Plan viewports across all seeds within budget."""
    per_seed = budget // len(states)
    all_vps: list[dict[str, Any]] = []
    for si in range(len(states)):
        grid, _ = states[si]
        for vp in _plan_seed_viewports(grid, w, h, per_seed):
            all_vps.append({"seed_index": si, **vp})
    logger.info("Planned %d queries across %d seeds", len(all_vps), len(states))
    return all_vps[:budget]


def _execute_queries(
    client: AstarClient, round_id: str, vps: list[dict[str, Any]], obs: ObservationStore
) -> None:
    """Execute all queries and feed results into the observation store."""
    for i, vp in enumerate(vps):
        try:
            res = client.query(
                round_id,
                int(vp["seed_index"]),
                int(vp["x"]),
                int(vp["y"]),
                int(vp["w"]),
                int(vp["h"]),
            )
            grid_data = res.get("grid", [])
            if grid_data:
                patch = np.array(grid_data, dtype=np.int32)
                obs.add_observation(int(vp["seed_index"]), int(vp["x"]), int(vp["y"]), patch)
            logger.info("Query %d/%d seed %d", i + 1, len(vps), vp["seed_index"])
        except APIError as e:
            logger.warning("Query %d failed: %s", i + 1, e)
        time.sleep(_QUERY_DELAY)


def _build_prediction(
    grid: np.ndarray, priors: np.ndarray, obs: ObservationStore, seed_index: int
) -> np.ndarray:
    """Build H x W x 6: priors -> blend obs -> static overrides -> floor."""
    tensor = priors[np.clip(grid.astype(np.int32), 0, 6)].copy()
    tensor = _blend_observations(tensor, obs, seed_index)
    tensor = _apply_static_overrides(tensor, grid)
    return _floor_and_normalize(tensor)


def _blend_observations(tensor: np.ndarray, obs: ObservationStore, seed_index: int) -> np.ndarray:
    """Blend observation data with count-scaled weights."""
    obs_probs = obs.get_observed_probs(seed_index)
    mask = obs.get_coverage_mask(seed_index) & ~np.isnan(obs_probs[:, :, 0])
    if not mask.any():
        return tensor
    counts = obs.observation_count(seed_index)[mask].astype(np.float64)
    w = (_MAX_OBS_WEIGHT * counts / (counts + OBS_CONFIDENCE_K))[:, np.newaxis]
    result = tensor.copy()
    result[mask] = w * obs_probs[mask] + (1.0 - w) * tensor[mask]
    pct = float(mask.mean() * 100)
    logger.info("Seed %d: blended %d cells (%.1f%%)", seed_index, int(mask.sum()), pct)
    return result


def _apply_static_overrides(tensor: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Override ocean/mountain cells with near-certain probabilities."""
    result = tensor.copy()
    r = (1.0 - STATIC_TERRAIN_CONFIDENCE) / (NUM_PREDICTION_CLASSES - 1)
    for terrain, cls in [(InternalTerrain.OCEAN, 0), (InternalTerrain.MOUNTAIN, 5)]:
        m = grid == terrain
        if m.any():
            result[m] = r
            result[m, cls] = STATIC_TERRAIN_CONFIDENCE
    return result


def _floor_and_normalize(tensor: np.ndarray) -> np.ndarray:
    """Apply probability floor and renormalize each cell."""
    safe = np.maximum(tensor, PROBABILITY_FLOOR)
    return safe / safe.sum(axis=2, keepdims=True)


def _submit_seed(
    client: AstarClient, round_id: str, si: int, pred: np.ndarray, dry_run: bool
) -> None:
    """Submit prediction for one seed."""
    if dry_run:
        logger.info("DRY RUN: skip seed %d (%s)", si, pred.shape)
        return
    try:
        resp = client.submit(round_id, si, pred)
        logger.info("Submitted seed %d: %s", si, resp)
    except APIError as e:
        logger.error("Submit failed for seed %d: %s", si, e)


if __name__ == "__main__":
    main()
