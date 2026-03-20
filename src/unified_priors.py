"""Unified survive-weighted prior builder.

Aggregates ground truth across all historical rounds per terrain type,
weighting survive rounds 3x and collapse rounds 1x. Includes distance-
to-settlement refinement for more accurate per-cell predictions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.constants import (
    COLLAPSE_WEIGHT,
    DIST_BIN_EDGES,
    NUM_INTERNAL_TYPES,
    NUM_PREDICTION_CLASSES,
    NUM_SEEDS,
    STATIC_TERRAIN_CONFIDENCE,
    SURVIVE_ROUNDS,
    SURVIVE_WEIGHT,
)
from src.features import compute_settlement_distance
from src.prior_builder import _apply_floor_to_row
from src.terrain import InternalTerrain

logger = logging.getLogger(__name__)


def build_unified_priors(data_dir: str | Path) -> np.ndarray:
    """Aggregate GT across all rounds per terrain type (survive-weighted).

    Returns shape (NUM_INTERNAL_TYPES, NUM_PREDICTION_CLASSES).
    """
    rounds = _load_round_metadata(data_dir)
    if not rounds:
        return _uniform(NUM_INTERNAL_TYPES, NUM_PREDICTION_CLASSES)

    accum = np.zeros((NUM_INTERNAL_TYPES, NUM_PREDICTION_CLASSES), dtype=np.float64)
    total = np.zeros(NUM_INTERNAL_TYPES, dtype=np.float64)

    for rnum, rd in rounds.items():
        w = SURVIVE_WEIGHT if rnum in SURVIVE_ROUNDS else COLLAPSE_WEIGHT
        _accum_round(rd, w, accum, total)

    return _normalize_1d(accum, total)


def build_distance_priors(data_dir: str | Path) -> np.ndarray:
    """Build P(class | terrain, distance_to_settlement) priors.

    Returns shape (NUM_INTERNAL_TYPES, num_bins, NUM_PREDICTION_CLASSES).
    """
    rounds = _load_round_metadata(data_dir)
    nb = len(DIST_BIN_EDGES) - 1
    if not rounds:
        return _uniform(NUM_INTERNAL_TYPES * nb, NUM_PREDICTION_CLASSES).reshape(
            NUM_INTERNAL_TYPES, nb, NUM_PREDICTION_CLASSES
        )

    accum = np.zeros((NUM_INTERNAL_TYPES, nb, NUM_PREDICTION_CLASSES), dtype=np.float64)
    total = np.zeros((NUM_INTERNAL_TYPES, nb), dtype=np.float64)

    for rnum, rd in rounds.items():
        w = SURVIVE_WEIGHT if rnum in SURVIVE_ROUNDS else COLLAPSE_WEIGHT
        _accum_round_dist(rd, w, accum, total)

    return _normalize_2d(accum, total)


def predict_from_priors(
    grid: np.ndarray,
    priors: np.ndarray,
    dist_priors: np.ndarray | None = None,
) -> np.ndarray:
    """Apply priors + distance refinement + static overrides + floor.

    Returns H x W x NUM_PREDICTION_CLASSES probability tensor.
    """
    gi = np.clip(grid.astype(np.int32), 0, NUM_INTERNAL_TYPES - 1)
    pred = _lookup_dist(gi, dist_priors) if dist_priors is not None else priors[gi].copy()
    _static_overrides(pred, gi)
    _apply_floor(pred)
    return pred


def save_priors(
    priors: np.ndarray,
    path: str | Path,
    dist_priors: np.ndarray | None = None,
) -> None:
    """Persist priors to disk as .npz archive."""
    data: dict[str, np.ndarray] = {"priors": priors}
    if dist_priors is not None:
        data["dist_priors"] = dist_priors
    np.savez(path, **data)  # type: ignore[arg-type]
    logger.info("Saved priors to %s", path)


def load_priors(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load priors from disk. Returns (priors, dist_priors or None)."""
    archive = np.load(path)
    p = archive["priors"]
    dp = archive.get("dist_priors")
    logger.info("Loaded priors from %s", path)
    return p, dp


# -- Data loading -----------------------------------------------------------


def _load_round_metadata(data_dir: str | Path) -> dict[int, Path]:
    """Discover rounds and map round_number -> directory path."""
    rounds_dir = Path(data_dir)
    if not rounds_dir.exists():
        return {}
    result: dict[int, Path] = {}
    for child in rounds_dir.iterdir():
        rj_path = child / "round.json"
        if not child.is_dir() or not rj_path.exists():
            continue
        with open(rj_path) as f:
            rj = json.load(f)
        rnum = rj.get("round_number", -1)
        if rnum > 0:
            result[rnum] = child
    return result


def _load_seed(rd: Path, si: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Load initial grid and GT for one seed. Returns None if missing."""
    sd = rd / f"seed_{si}"
    ig_p, gt_p = sd / "initial_grid.npy", sd / "ground_truth.npy"
    if not ig_p.exists() or not gt_p.exists():
        return None
    return np.load(ig_p), np.load(gt_p)


# -- Accumulation -----------------------------------------------------------


def _accum_round(rd: Path, weight: float, accum: np.ndarray, total: np.ndarray) -> None:
    """Add one round's data to terrain-type accumulators."""
    for si in range(NUM_SEEDS):
        pair = _load_seed(rd, si)
        if pair is None:
            continue
        ig, gt = pair
        for t in range(NUM_INTERNAL_TYPES):
            mask = ig == t
            n = int(mask.sum())
            if n == 0:
                continue
            accum[t] += weight * gt[mask].sum(axis=0)  # type: ignore[index]
            total[t] += weight * n


def _accum_round_dist(rd: Path, weight: float, accum: np.ndarray, total: np.ndarray) -> None:
    """Add one round's data to distance-aware accumulators."""
    nb = len(DIST_BIN_EDGES) - 1
    for si in range(NUM_SEEDS):
        pair = _load_seed(rd, si)
        if pair is None:
            continue
        ig, gt = pair
        dist = compute_settlement_distance(ig)
        for t in range(NUM_INTERNAL_TYPES):
            for b in range(nb):
                lo, hi = DIST_BIN_EDGES[b], DIST_BIN_EDGES[b + 1]
                mask = (ig == t) & (dist >= lo) & (dist < hi)
                n = int(mask.sum())
                if n == 0:
                    continue
                accum[t, b] += weight * gt[mask].sum(axis=0)  # type: ignore[index]
                total[t, b] += weight * n


# -- Prediction helpers -----------------------------------------------------


def _lookup_dist(gi: np.ndarray, dp: np.ndarray) -> np.ndarray:
    """Look up distance-aware priors for each cell."""
    h, w = gi.shape
    dist = compute_settlement_distance(gi.astype(np.int8))
    pred = np.zeros((h, w, NUM_PREDICTION_CLASSES), dtype=np.float64)
    for b in range(len(DIST_BIN_EDGES) - 1):
        lo, hi = DIST_BIN_EDGES[b], DIST_BIN_EDGES[b + 1]
        dm = (dist >= lo) & (dist < hi)
        if dm.any():
            pred[dm] = dp[gi[dm], b]
    return pred


def _static_overrides(pred: np.ndarray, gi: np.ndarray) -> None:
    """Override predictions for static terrain (ocean, mountain)."""
    conf = STATIC_TERRAIN_CONFIDENCE
    res = (1.0 - conf) / (NUM_PREDICTION_CLASSES - 1)
    for mask, cls in [
        (gi == InternalTerrain.OCEAN, 0),
        (gi == InternalTerrain.MOUNTAIN, 5),
    ]:
        pred[mask] = res
        pred[mask, cls] = conf


def _apply_floor(pred: np.ndarray) -> None:
    """Apply probability floor with iterative clamp-and-redistribute."""
    c = pred.shape[2]
    flat = pred.reshape(-1, c)
    for i in range(flat.shape[0]):
        flat[i] = _apply_floor_to_row(flat[i])


# -- Normalization ----------------------------------------------------------


def _normalize_1d(accum: np.ndarray, total: np.ndarray) -> np.ndarray:
    """Normalize 1D accumulated counts to probability distributions."""
    r = np.full_like(accum, 1.0 / NUM_PREDICTION_CLASSES)
    for t in range(accum.shape[0]):
        if total[t] > 0:
            r[t] = accum[t] / total[t]
    return r


def _normalize_2d(accum: np.ndarray, total: np.ndarray) -> np.ndarray:
    """Normalize 2D accumulated counts."""
    r = np.full_like(accum, 1.0 / NUM_PREDICTION_CLASSES)
    for t in range(accum.shape[0]):
        for b in range(accum.shape[1]):
            if total[t, b] > 0:
                r[t, b] = accum[t, b] / total[t, b]
    return r


def _uniform(rows: int, cols: int) -> np.ndarray:
    """Return uniform distribution array."""
    return np.full((rows, cols), 1.0 / cols, dtype=np.float64)
