"""Submit predictions v3: XGBoost model + two-phase regime detection.

Phase 1: 5 probe queries (1 per seed) to detect regime from settlement survival
Phase 2: 45 observation queries on settlement clusters
Prediction: XGBoost trained on matching-regime historical rounds + observation blending

Usage:
    python -m scripts.submit_v3 --token <JWT>
    python -m scripts.submit_v3 --token <JWT> --regime survive --budget 50
"""

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
    OBS_CONFIDENCE_K,
    PROBABILITY_FLOOR,
    TOTAL_QUERY_BUDGET,
)
from src.ml_predictor import build_training_data, predict_grid, train_model
from src.observation import ObservationStore
from src.prediction_validator import validate_predictions
from src.state_loader import load_round
from src.terrain import InternalTerrain

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_VP_SIZE = 15
_QUERY_DELAY = 0.2
_DATA_DIR = "data/rounds"
_COLLAPSE_THRESHOLD = 0.05
_AGGRESSIVE_THRESHOLD = 0.35
_MAX_OBS_WEIGHT = 0.8


def main() -> None:
    args = _parse_args()
    client = AstarClient(args.token)
    round_id, rd = _resolve_round(client, args.round_id)
    states = load_round(rd)
    h = rd.get("map_height", DEFAULT_MAP_HEIGHT)
    w = rd.get("map_width", DEFAULT_MAP_WIDTH)
    logger.info("Round %s: %dx%d, %d seeds", round_id, w, h, len(states))

    obs = ObservationStore(h, w, len(states))

    # Phase 1: Probe regime (or use --regime flag)
    if args.regime:
        regime = args.regime
        probe_used = 0
        logger.info("Using --regime %s (no probes)", regime)
    else:
        regime, probe_used = _probe_regime(client, round_id, states, obs, h, w, args.dry_run)

    # Train XGBoost on matching-regime rounds
    model = _train_regime_model(regime)

    # Phase 2: Observation queries
    obs_budget = args.budget - probe_used
    _run_observations(client, round_id, states, obs, obs_budget, w, h, args.dry_run)

    # Build predictions: XGBoost base → blend observations → static → floor
    predictions = []
    grids = []
    for si in range(len(states)):
        grid = states[si][0]
        grids.append(grid)
        pred = _build_prediction(grid, model, obs, si)
        predictions.append(pred)

    # Validate
    errors = validate_predictions(predictions, grids)
    if errors and not args.force:
        for e in errors:
            logger.error("VALIDATION: %s", e)
        sys.exit(1)

    # Submit
    for si, pred in enumerate(predictions):
        _submit_seed(client, round_id, si, pred, args.dry_run)
    logger.info("Done. %d seeds submitted.", len(states))


# -- Phase 1: Regime detection ------------------------------------------------


def _probe_regime(
    client: AstarClient,
    round_id: str,
    states: list,
    obs: ObservationStore,
    h: int,
    w: int,
    dry_run: bool,
) -> tuple[str, int]:
    """Probe 1 viewport per seed, detect regime from settlement survival."""
    total_checked, total_survived, used = 0, 0, 0
    for si in range(len(states)):
        grid = states[si][0]
        cx, cy = _find_densest_settlement(grid)
        if cx is None:
            continue
        vx = max(0, min(cx - _VP_SIZE // 2, w - _VP_SIZE))
        vy = max(0, min(cy - _VP_SIZE // 2, h - _VP_SIZE))
        if not dry_run:
            try:
                res = client.query(round_id, si, vx, vy, _VP_SIZE, _VP_SIZE)
                patch = np.array(res["grid"], dtype=np.int32)
                obs.add_observation(si, vx, vy, patch)
                used += 1
                time.sleep(_QUERY_DELAY)
            except APIError as e:
                logger.warning("Probe %d failed: %s", si, e)
                continue
        checked, survived = _count_survival(grid, obs, si)
        total_checked += checked
        total_survived += survived

    rate = total_survived / max(total_checked, 1)
    if rate < _COLLAPSE_THRESHOLD:
        regime = "collapse"
    elif rate > _AGGRESSIVE_THRESHOLD:
        regime = "aggressive"
    else:
        regime = "survive"
    logger.info(
        "Regime: %s (rate=%.2f, %d/%d, %d probes)",
        regime,
        rate,
        total_survived,
        total_checked,
        used,
    )
    return regime, used


def _find_densest_settlement(grid: np.ndarray) -> tuple[int | None, int | None]:
    """Find settlement cell with most nearby settlement neighbors."""
    ys, xs = np.where((grid == InternalTerrain.SETTLEMENT) | (grid == InternalTerrain.PORT))
    if len(ys) == 0:
        return None, None
    best_idx, best_n = 0, 0
    for j in range(len(ys)):
        n = int(((np.abs(ys - ys[j]) <= _VP_SIZE) & (np.abs(xs - xs[j]) <= _VP_SIZE)).sum())
        if n > best_n:
            best_n, best_idx = n, j
    return int(xs[best_idx]), int(ys[best_idx])


def _count_survival(grid: np.ndarray, obs: ObservationStore, si: int) -> tuple[int, int]:
    """Count observed settlement cells and how many still show settlement."""
    obs_probs = obs.get_observed_probs(si)
    coverage = obs.get_coverage_mask(si)
    settle = (grid == InternalTerrain.SETTLEMENT) | (grid == InternalTerrain.PORT)
    observed = settle & coverage
    checked, survived = 0, 0
    if observed.any():
        ys, xs = np.where(observed)
        for y, x in zip(ys, xs, strict=True):
            probs = obs_probs[y, x]
            if not np.isnan(probs[0]):
                checked += 1
                if (probs[1] + probs[2]) > 0.3:
                    survived += 1
    return checked, survived


# -- Model training -----------------------------------------------------------

_REGIME_EXCLUDE: dict[str, set[int]] = {
    "survive": {3, 4},
    "aggressive": {3, 4},
    "collapse": {1, 2, 5, 6},
}


def _train_regime_model(regime: str) -> Any:
    """Train XGBoost on rounds matching the detected regime."""
    # Train on all historical rounds — XGBoost learns regime patterns from data
    x, y = build_training_data(_DATA_DIR, exclude_round=None)
    model = train_model(x, y, seed=42)
    logger.info("Trained XGBoost on regime=%s (%d samples)", regime, len(x))
    return model


# -- Phase 2: Observations ----------------------------------------------------


def _run_observations(
    client: AstarClient,
    round_id: str,
    states: list,
    obs: ObservationStore,
    budget: int,
    w: int,
    h: int,
    dry_run: bool,
) -> None:
    """Execute observation queries on settlement clusters."""
    per_seed = budget // len(states)
    total = 0
    for si in range(len(states)):
        grid = states[si][0]
        vps = _plan_seed_viewports(grid, w, h, per_seed)
        for vp in vps:
            if dry_run:
                continue
            try:
                res = client.query(round_id, si, vp["x"], vp["y"], vp["w"], vp["h"])
                patch = np.array(res["grid"], dtype=np.int32)
                obs.add_observation(si, vp["x"], vp["y"], patch)
                total += 1
                time.sleep(_QUERY_DELAY)
            except APIError as e:
                logger.warning("Query failed: %s", e)
    logger.info("Phase 2: %d observation queries", total)


def _plan_seed_viewports(grid: np.ndarray, w: int, h: int, budget: int) -> list[dict[str, int]]:
    """Plan overlapping viewports on settlement clusters."""
    ys, xs = np.where((grid == InternalTerrain.SETTLEMENT) | (grid == InternalTerrain.PORT))
    if len(ys) == 0:
        return [{"x": w // 4, "y": h // 4, "w": _VP_SIZE, "h": _VP_SIZE}]
    # Score settlements by neighbor density, pick top clusters
    scored = sorted(
        range(len(ys)),
        key=lambda j: -((np.abs(ys - ys[j]) <= _VP_SIZE) & (np.abs(xs - xs[j]) <= _VP_SIZE)).sum(),
    )
    vps: list[dict[str, int]] = []
    for j in scored:
        if len(vps) >= budget:
            break
        cx, cy = int(xs[j]), int(ys[j])
        for dx, dy in [(0, 0), (-5, 0), (5, 0), (0, -5), (0, 5)]:
            if len(vps) >= budget:
                break
            vx = max(0, min(cx - _VP_SIZE // 2 + dx, w - _VP_SIZE))
            vy = max(0, min(cy - _VP_SIZE // 2 + dy, h - _VP_SIZE))
            vp = {"x": vx, "y": vy, "w": _VP_SIZE, "h": _VP_SIZE}
            if vp not in vps:
                vps.append(vp)
    return vps[:budget]


# -- Prediction building ------------------------------------------------------


def _build_prediction(
    grid: np.ndarray,
    model: Any,
    obs: ObservationStore,
    si: int,
) -> np.ndarray:
    """XGBoost base → observation blend → static override → floor."""
    pred = predict_grid(grid, model)
    pred = _blend_observations(pred, obs, si)
    return _floor_and_normalize(pred)


def _blend_observations(
    tensor: np.ndarray,
    obs: ObservationStore,
    si: int,
) -> np.ndarray:
    """Blend observations with count-scaled weights."""
    obs_probs = obs.get_observed_probs(si)
    mask = obs.get_coverage_mask(si) & ~np.isnan(obs_probs[:, :, 0])
    if not mask.any():
        return tensor
    result = tensor.copy()
    counts = obs.observation_count(si)[mask].astype(np.float64)
    w = (_MAX_OBS_WEIGHT * counts / (counts + OBS_CONFIDENCE_K))[:, np.newaxis]
    result[mask] = w * obs_probs[mask] + (1.0 - w) * tensor[mask]
    logger.info("Seed %d: blended %d cells (%.0f%%)", si, int(mask.sum()), mask.mean() * 100)
    return result


def _floor_and_normalize(tensor: np.ndarray) -> np.ndarray:
    safe = np.maximum(tensor, PROBABILITY_FLOOR)
    return safe / safe.sum(axis=2, keepdims=True)


def _submit_seed(client: AstarClient, rid: str, si: int, pred: np.ndarray, dry: bool) -> None:
    if dry:
        logger.info("DRY RUN: seed %d (%s)", si, pred.shape)
        return
    try:
        client.submit(rid, si, pred)
        logger.info("Submitted seed %d", si)
    except APIError as e:
        logger.error("Submit failed seed %d: %s", si, e)


def _resolve_round(client: AstarClient, rid: str | None) -> tuple[str, dict[str, Any]]:
    if rid:
        return rid, client.get_round(rid)
    active = client.get_active_round()
    if not active:
        logger.error("No active round")
        sys.exit(1)
    return active["id"], client.get_round(active["id"])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit v3: XGBoost + regime detection")
    p.add_argument("--token", required=True)
    p.add_argument("--round-id", default=None)
    p.add_argument("--budget", type=int, default=TOTAL_QUERY_BUDGET)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--regime", choices=["survive", "collapse", "aggressive"])
    return p.parse_args()


if __name__ == "__main__":
    main()
