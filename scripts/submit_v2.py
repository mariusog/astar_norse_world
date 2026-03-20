"""Submit predictions v2: survive priors + distance priors + observation blending."""

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
    NUM_SEEDS,
    OBS_CONFIDENCE_K,
    PROBABILITY_FLOOR,
    TOTAL_QUERY_BUDGET,
)
from src.observation import ObservationStore
from src.prediction_validator import backtest_check, validate_predictions
from src.state_loader import load_round
from src.terrain import InternalTerrain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
_MAX_OBS_WEIGHT = 0.8
_VP_SIZE = 15
_VP_OFFSET = 5
_QUERY_DELAY = 0.2
_DATA_DIR = "data/rounds"


def main() -> None:
    """Two-phase pipeline: probe regime, then observe with adapted priors."""
    args = _parse_args()
    client = AstarClient(args.token)
    round_id, rd = _resolve_round(client, args.round_id)
    states = load_round(rd)
    h = rd.get("map_height", DEFAULT_MAP_HEIGHT)
    w = rd.get("map_width", DEFAULT_MAP_WIDTH)
    n_seeds = rd.get("seeds_count", NUM_SEEDS)
    logger.info("Round %s: %dx%d, %d seeds", round_id, w, h, n_seeds)

    # Phase 1: Probe — 1 query per seed on densest settlement cluster
    obs = ObservationStore(h, w, n_seeds)
    probe_vps = _plan_probe_queries(states, w, h)
    probe_budget = len(probe_vps)
    if not args.dry_run:
        _execute_queries(client, round_id, probe_vps, obs)
    regime = _detect_regime_from_probes(states, obs, n_seeds)
    priors = _build_adaptive_priors(regime)
    logger.info("Phase 1 done: %d probes, regime=%s", probe_budget, regime)

    # Phase 2: Observe — remaining budget on settlement clusters
    obs_budget = args.budget - probe_budget
    obs_vps = _plan_all_queries(states, w, h, obs_budget)
    if not args.dry_run:
        _execute_queries(client, round_id, obs_vps, obs)
    logger.info("Phase 2 done: %d observation queries", len(obs_vps))

    # Build predictions, validate, submit
    predictions = _build_all_predictions(states, priors, obs)
    grids = [s[0] for s in states]
    errors = validate_predictions(predictions, grids)
    errors.extend(backtest_check(predictions, grids, _DATA_DIR))
    if errors and not args.force:
        for e in errors:
            logger.error("VALIDATION: %s", e)
        logger.error("Aborting. Use --force to override.")
        sys.exit(1)
    elif errors:
        for e in errors:
            logger.warning("VALIDATION (forced): %s", e)
    for si, pred in enumerate(predictions):
        _submit_seed(client, round_id, si, pred, args.dry_run)
    logger.info("Pipeline complete for round %s", round_id)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit predictions v2")
    p.add_argument("--token", required=True, help="JWT auth token")
    p.add_argument("--round-id", default=None)
    p.add_argument("--budget", type=int, default=TOTAL_QUERY_BUDGET)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="Submit even if validation fails")
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


_REGIME_AGGRESSIVE_THRESHOLD = 0.35
_REGIME_COLLAPSE_THRESHOLD = 0.05


def _plan_probe_queries(
    states: list[tuple[np.ndarray, list]],
    w: int,
    h: int,
) -> list[dict[str, Any]]:
    """Plan 1 probe query per seed: 15x15 on densest settlement cluster."""
    vps: list[dict[str, Any]] = []
    for si in range(len(states)):
        grid = states[si][0]
        center = _find_densest_settlement(grid)
        if center is None:
            continue
        cx, cy = center
        vx = max(0, min(cx - _VP_SIZE // 2, w - _VP_SIZE))
        vy = max(0, min(cy - _VP_SIZE // 2, h - _VP_SIZE))
        vps.append({"seed_index": si, "x": vx, "y": vy, "w": _VP_SIZE, "h": _VP_SIZE})
    return vps


def _find_densest_settlement(grid: np.ndarray) -> tuple[int, int] | None:
    """Find the settlement cell with most nearby settlement neighbors."""
    ys, xs = np.where((grid == InternalTerrain.SETTLEMENT) | (grid == InternalTerrain.PORT))
    if len(ys) == 0:
        return None
    best_idx, best_n = 0, 0
    for j in range(len(ys)):
        n = int(((np.abs(ys - ys[j]) <= _VP_SIZE) & (np.abs(xs - xs[j]) <= _VP_SIZE)).sum())
        if n > best_n:
            best_n = n
            best_idx = j
    return int(xs[best_idx]), int(ys[best_idx])


def _detect_regime_from_probes(
    states: list[tuple[np.ndarray, list]],
    obs: ObservationStore,
    n_seeds: int,
) -> str:
    """Count observed settlement survival across all probe viewports."""
    total_checked, total_survived = 0, 0
    for si in range(n_seeds):
        grid = states[si][0]
        obs_probs = obs.get_observed_probs(si)
        coverage = obs.get_coverage_mask(si)
        settle_mask = (grid == InternalTerrain.SETTLEMENT) | (grid == InternalTerrain.PORT)
        observed_settles = settle_mask & coverage
        if not observed_settles.any():
            continue
        # Check if observed settlement cells have high settlement probability
        ys, xs = np.where(observed_settles)
        for y, x in zip(ys, xs, strict=True):
            total_checked += 1
            probs = obs_probs[y, x]
            if not np.isnan(probs[0]) and (probs[1] + probs[2]) > 0.3:
                total_survived += 1

    rate = total_survived / max(total_checked, 1)
    if rate < _REGIME_COLLAPSE_THRESHOLD:
        regime = "collapse"
    elif rate > _REGIME_AGGRESSIVE_THRESHOLD:
        regime = "aggressive"
    else:
        regime = "survive"
    logger.info(
        "Regime: %s (rate=%.2f, %d/%d settlements survived in probes)",
        regime,
        rate,
        total_survived,
        total_checked,
    )
    return regime


def _build_adaptive_priors(regime: str) -> np.ndarray:
    """Build priors using only rounds matching the detected regime."""
    from src.adaptive_priors import build_adaptive_priors

    return build_adaptive_priors(regime, _DATA_DIR)


def _find_settlements(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return (x, y) of settlement and port cells."""
    ys, xs = np.where((grid == InternalTerrain.SETTLEMENT) | (grid == InternalTerrain.PORT))
    return list(zip(xs.tolist(), ys.tolist(), strict=True))


def _find_dense_clusters(
    cells: list[tuple[int, int]],
    max_clusters: int = 4,
) -> list[tuple[int, int]]:
    """Pick settlements with most nearby neighbors for max overlap."""
    if len(cells) <= max_clusters:
        return cells
    scored = []
    for cx, cy in cells:
        neighbors = sum(
            1 for ox, oy in cells if abs(ox - cx) <= _VP_SIZE and abs(oy - cy) <= _VP_SIZE
        )
        scored.append((neighbors, cx, cy))
    scored.sort(reverse=True)
    # Pick top clusters with minimum spacing
    out: list[tuple[int, int]] = []
    for _score, cx, cy in scored:
        if len(out) >= max_clusters:
            break
        too_close = any(abs(cx - ox) < _VP_OFFSET and abs(cy - oy) < _VP_OFFSET for ox, oy in out)
        if not too_close:
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
    for cx, cy in _find_dense_clusters(cells):
        if len(vps) >= budget:
            break
        vps.extend(_viewports_around(cx, cy, w, h, budget - len(vps), vps))
    return vps[:budget]


def _tile_viewports(w: int, h: int, budget: int) -> list[dict[str, int]]:
    """Fallback: tile the map center when no settlements found."""
    cx, cy = w // 2, h // 2
    return _viewports_around(cx, cy, w, h, budget, [])


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
    client: AstarClient,
    round_id: str,
    vps: list[dict[str, Any]],
    obs: ObservationStore,
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
                obs.add_observation(
                    int(vp["seed_index"]),
                    int(vp["x"]),
                    int(vp["y"]),
                    patch,
                )
            logger.info("Query %d/%d seed %d", i + 1, len(vps), vp["seed_index"])
        except APIError as e:
            logger.warning("Query %d failed: %s", i + 1, e)
        time.sleep(_QUERY_DELAY)


def _build_all_predictions(
    states: list[tuple[np.ndarray, list]],
    priors: np.ndarray,
    obs: ObservationStore,
) -> list[np.ndarray]:
    """Build predictions for all seeds."""
    return [_build_prediction(s[0], priors, obs, i) for i, s in enumerate(states)]


def _build_prediction(
    grid: np.ndarray,
    priors: np.ndarray,
    obs: ObservationStore,
    seed_index: int,
) -> np.ndarray:
    """Build H x W x 6: adaptive priors -> blend obs -> static -> floor."""
    h, w = grid.shape
    tensor = np.full((h, w, 6), 1.0 / 6)
    gi = np.clip(grid.astype(np.int32), 0, priors.shape[0] - 1)
    tensor = priors[gi].copy()
    tensor[grid == InternalTerrain.OCEAN] = [1, 0, 0, 0, 0, 0]
    tensor[grid == InternalTerrain.MOUNTAIN] = [0, 0, 0, 0, 0, 1]
    tensor = _blend_observations(tensor, obs, seed_index)
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
    logger.info(
        "Seed %d: blended %d cells (%.1f%%)",
        seed_index,
        int(mask.sum()),
        pct,
    )
    return result


def _floor_and_normalize(tensor: np.ndarray) -> np.ndarray:
    """Apply probability floor and renormalize each cell."""
    safe = np.maximum(tensor, PROBABILITY_FLOOR)
    return safe / safe.sum(axis=2, keepdims=True)


def _submit_seed(
    client: AstarClient,
    round_id: str,
    si: int,
    pred: np.ndarray,
    dry_run: bool,
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
