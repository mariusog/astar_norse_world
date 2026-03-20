"""Observation-focused query planner.

Places 15x15 observation viewports to maximize coverage of dynamic
cells. Uses settlement clustering to center viewports on areas of
highest prediction uncertainty.
"""

from __future__ import annotations

import logging

import numpy as np

from src.constants import (
    NUM_SEEDS,
    TOTAL_QUERY_BUDGET,
    VIEWPORT_MAX_SIZE,
)
from src.query_strategy import Viewport

logger = logging.getLogger(__name__)

# All viewports are max size for maximum coverage per query
VP_SIZE = VIEWPORT_MAX_SIZE

# Queries per seed (budget / seeds)
QUERIES_PER_SEED = TOTAL_QUERY_BUDGET // NUM_SEEDS


def plan_queries(
    grid: np.ndarray,
    dynamic_mask: np.ndarray,
    budget: int = TOTAL_QUERY_BUDGET,
    num_seeds: int = NUM_SEEDS,
) -> list[list[Viewport]]:
    """Plan observation viewports for all seeds.

    Places 15x15 viewports to maximize overlap on dynamic cells.
    All queries are observation viewports (no probes).

    Args:
        grid: H x W array of InternalTerrain values.
        dynamic_mask: H x W boolean mask of dynamic cells.
        budget: Total query budget across all seeds.
        num_seeds: Number of seeds.

    Returns:
        List of num_seeds lists, each containing Viewport objects.
    """
    h, w = grid.shape
    per_seed = budget // num_seeds
    result: list[list[Viewport]] = []

    for seed_idx in range(num_seeds):
        vps = _plan_seed_viewports(dynamic_mask, seed_idx, per_seed, w, h)
        result.append(vps)
        _log_coverage(vps, dynamic_mask, seed_idx, w, h)

    return result


def _plan_seed_viewports(
    dynamic_mask: np.ndarray,
    seed_idx: int,
    max_queries: int,
    map_w: int,
    map_h: int,
) -> list[Viewport]:
    """Plan viewports for a single seed targeting dynamic cells."""
    vw = min(VP_SIZE, map_w)
    vh = min(VP_SIZE, map_h)
    clusters = _find_dynamic_clusters(dynamic_mask, map_w, map_h)
    viewports: list[Viewport] = []
    covered = np.zeros_like(dynamic_mask, dtype=bool)

    for _ in range(max_queries):
        vp = _best_viewport(dynamic_mask, covered, clusters, map_w, map_h)
        if vp is None:
            break
        viewports.append(
            Viewport(
                seed_index=seed_idx,
                viewport_x=vp[0],
                viewport_y=vp[1],
                viewport_w=vw,
                viewport_h=vh,
            )
        )
        _mark_covered(covered, vp[0], vp[1], map_w, map_h)

    return viewports


def _find_dynamic_clusters(
    dynamic_mask: np.ndarray,
    map_w: int,
    map_h: int,
) -> list[tuple[int, int]]:
    """Find cluster centers of dynamic cells using grid sampling.

    Returns list of (x, y) center positions sorted by dynamic
    cell density (highest first).
    """
    step = VP_SIZE // 2
    centers: list[tuple[int, int, int]] = []

    for y in range(0, map_h, step):
        for x in range(0, map_w, step):
            y0 = max(0, y - VP_SIZE // 2)
            y1 = min(map_h, y + VP_SIZE // 2)
            x0 = max(0, x - VP_SIZE // 2)
            x1 = min(map_w, x + VP_SIZE // 2)
            count = int(dynamic_mask[y0:y1, x0:x1].sum())
            if count > 0:
                centers.append((x, y, count))

    centers.sort(key=lambda c: c[2], reverse=True)
    return [(c[0], c[1]) for c in centers]


def _best_viewport(
    dynamic_mask: np.ndarray,
    covered: np.ndarray,
    clusters: list[tuple[int, int]],
    map_w: int,
    map_h: int,
) -> tuple[int, int] | None:
    """Select viewport position covering most uncovered dynamic cells."""
    best_pos: tuple[int, int] | None = None
    best_score = 0

    candidates = _generate_candidates(clusters, map_w, map_h)

    for x, y in candidates:
        score = _viewport_score(dynamic_mask, covered, x, y, map_w, map_h)
        if score > best_score:
            best_score = score
            best_pos = (x, y)

    if best_score == 0:
        return _fallback_viewport(covered, map_w, map_h)

    return best_pos


def _generate_candidates(
    clusters: list[tuple[int, int]],
    map_w: int,
    map_h: int,
) -> list[tuple[int, int]]:
    """Generate candidate viewport positions from cluster centers."""
    vw = min(VP_SIZE, map_w)
    vh = min(VP_SIZE, map_h)
    candidates: list[tuple[int, int]] = []
    offsets = [0, -vw // 3, vw // 3]

    for cx, cy in clusters:
        for dx in offsets:
            for dy in offsets:
                x = _clamp(cx - vw // 2 + dx, 0, map_w - vw)
                y = _clamp(cy - vh // 2 + dy, 0, map_h - vh)
                candidates.append((x, y))

    return list(set(candidates))


def _viewport_score(
    dynamic_mask: np.ndarray,
    covered: np.ndarray,
    x: int,
    y: int,
    map_w: int,
    map_h: int,
) -> int:
    """Count uncovered dynamic cells in a viewport at (x, y)."""
    x1 = min(x + VP_SIZE, map_w)
    y1 = min(y + VP_SIZE, map_h)
    region_dyn = dynamic_mask[y:y1, x:x1]
    region_cov = covered[y:y1, x:x1]
    return int((region_dyn & ~region_cov).sum())


def _mark_covered(
    covered: np.ndarray,
    x: int,
    y: int,
    map_w: int,
    map_h: int,
) -> None:
    """Mark cells in viewport as covered."""
    x1 = min(x + VP_SIZE, map_w)
    y1 = min(y + VP_SIZE, map_h)
    covered[y:y1, x:x1] = True


def _fallback_viewport(
    covered: np.ndarray,
    map_w: int,
    map_h: int,
) -> tuple[int, int] | None:
    """Find a viewport covering the most uncovered cells overall."""
    vw = min(VP_SIZE, map_w)
    vh = min(VP_SIZE, map_h)
    best_pos: tuple[int, int] | None = None
    best_count = 0
    step = max(1, vw)

    for y in range(0, max(1, map_h - vh + 1), step):
        for x in range(0, max(1, map_w - vw + 1), step):
            x1 = min(x + vw, map_w)
            y1 = min(y + vh, map_h)
            count = int((~covered[y:y1, x:x1]).sum())
            if count > best_count:
                best_count = count
                best_pos = (x, y)

    return best_pos


def _log_coverage(
    viewports: list[Viewport],
    dynamic_mask: np.ndarray,
    seed_idx: int,
    map_w: int,
    map_h: int,
) -> None:
    """Log coverage statistics for planned viewports."""
    covered = np.zeros_like(dynamic_mask, dtype=bool)
    for vp in viewports:
        x1 = min(vp.viewport_x + vp.viewport_w, map_w)
        y1 = min(vp.viewport_y + vp.viewport_h, map_h)
        covered[vp.viewport_y : y1, vp.viewport_x : x1] = True

    dyn_total = int(dynamic_mask.sum())
    dyn_covered = int((dynamic_mask & covered).sum()) if dyn_total > 0 else 0
    frac = dyn_covered / dyn_total if dyn_total > 0 else 1.0

    logger.info(
        "Seed %d: %d viewports, %d/%d dynamic cells covered (%.0f%%)",
        seed_idx,
        len(viewports),
        dyn_covered,
        dyn_total,
        frac * 100,
    )


def _clamp(value: int, lo: int, hi: int) -> int:
    """Clamp value to [lo, hi]."""
    return max(lo, min(value, hi))
