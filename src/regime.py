"""Regime detection and regime-aware prior selection.

The server simulation has two distinct regimes:
- 'survive': settlements persist through 50 years (~30 final settlements)
- 'collapse': all settlements collapse to ruins/empty/forest (0 final settlements)

Detecting the regime from a few viewport observations allows us to pick
the correct prior set, which is worth 15+ score points on hard rounds.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.constants import NUM_PREDICTION_CLASSES, PROBABILITY_FLOOR
from src.features import compute_settlement_distance
from src.terrain import InternalTerrain, Terrain

logger = logging.getLogger(__name__)

# Regime detection: how many initial settlement cells to probe
REGIME_PROBE_QUERIES = 2
# If this fraction of probed settlement cells are still settlements, it's "survive"
REGIME_SURVIVE_THRESHOLD = 0.3


def build_regime_priors(
    data_dir: str = "data/rounds",
) -> dict[str, dict[int, np.ndarray]]:
    """Build separate terrain priors for survive and collapse regimes.

    Returns:
        Dict with keys 'survive' and 'collapse', each mapping
        InternalTerrain int -> probability vector (shape 6).
    """
    round_dirs = _discover_rounds(data_dir)
    survive_vecs: dict[int, list[np.ndarray]] = {}
    collapse_vecs: dict[int, list[np.ndarray]] = {}

    for rd in round_dirs:
        regime = _classify_round_regime(rd)
        target = survive_vecs if regime == "survive" else collapse_vecs
        _accumulate_priors(rd, target)

    return {
        "survive": _average_priors(survive_vecs),
        "collapse": _average_priors(collapse_vecs),
    }


def build_distance_priors(
    data_dir: str = "data/rounds",
) -> dict[str, dict[tuple[int, int], np.ndarray]]:
    """Build distance-conditioned priors per regime.

    Returns:
        Dict with keys 'survive' and 'collapse', each mapping
        (terrain_type, distance_bin) -> probability vector.
    """
    round_dirs = _discover_rounds(data_dir)
    survive_dist: dict[tuple[int, int], list[np.ndarray]] = {}
    collapse_dist: dict[tuple[int, int], list[np.ndarray]] = {}

    for rd in round_dirs:
        regime = _classify_round_regime(rd)
        target = survive_dist if regime == "survive" else collapse_dist
        _accumulate_distance_priors(rd, target)

    return {
        "survive": _average_dist_priors(survive_dist),
        "collapse": _average_dist_priors(collapse_dist),
    }


def detect_regime_from_observations(
    observation_classes: list[int],
) -> str:
    """Detect regime from observed terrain classes at settlement cells.

    Args:
        observation_classes: list of prediction class indices observed
            at cells that were initially settlements.

    Returns:
        'survive' or 'collapse'.
    """
    if not observation_classes:
        return "survive"  # default assumption

    settlement_count = sum(1 for c in observation_classes if c == Terrain.SETTLEMENT)
    fraction = settlement_count / len(observation_classes)
    regime = "survive" if fraction >= REGIME_SURVIVE_THRESHOLD else "collapse"
    logger.info(
        "Regime detection: %d/%d settlements survived -> %s",
        settlement_count,
        len(observation_classes),
        regime,
    )
    return regime


def build_prediction(
    grid: np.ndarray,
    regime: str,
    regime_priors: dict[str, dict[int, np.ndarray]],
    distance_priors: dict[str, dict[tuple[int, int], np.ndarray]] | None = None,
    distance_blend: float = 0.3,
) -> np.ndarray:
    """Build H x W x 6 prediction using regime-aware priors.

    Args:
        grid: Initial terrain grid (H x W, InternalTerrain values).
        regime: 'survive' or 'collapse'.
        regime_priors: Output of build_regime_priors().
        distance_priors: Optional output of build_distance_priors().
        distance_blend: Weight for distance-conditioned adjustment.

    Returns:
        H x W x 6 probability tensor.
    """
    priors = regime_priors.get(regime, regime_priors.get("survive", {}))
    pred = _apply_terrain_priors(grid, priors)

    if distance_priors and regime in distance_priors:
        pred = _apply_distance_refinement(
            pred,
            grid,
            distance_priors[regime],
            distance_blend,
        )

    pred = _apply_static_overrides(pred, grid)
    pred = np.maximum(pred, PROBABILITY_FLOOR)
    pred = pred / pred.sum(axis=2, keepdims=True)
    return pred


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _discover_rounds(data_dir: str) -> list[Path]:
    """Find all round directories with ground truth data."""
    base = Path(data_dir)
    if not base.exists():
        return []
    return [
        d
        for d in sorted(base.iterdir())
        if d.is_dir() and (d / "seed_0" / "ground_truth.npy").exists()
    ]


def _classify_round_regime(round_dir: Path) -> str:
    """Determine if a round is survive or collapse from GT data."""
    total_settlements = 0
    count = 0
    for i in range(5):
        gt_path = round_dir / f"seed_{i}" / "ground_truth.npy"
        if not gt_path.exists():
            continue
        gt = np.load(gt_path)
        total_settlements += int((gt.argmax(axis=2) == Terrain.SETTLEMENT).sum())
        count += 1
    avg = total_settlements / max(count, 1)
    return "survive" if avg > 10 else "collapse"


def _accumulate_priors(
    round_dir: Path,
    target: dict[int, list[np.ndarray]],
) -> None:
    """Collect GT vectors per terrain type from one round."""
    for i in range(5):
        gt_path = round_dir / f"seed_{i}" / "ground_truth.npy"
        grid_path = round_dir / f"seed_{i}" / "initial_grid.npy"
        if not gt_path.exists() or not grid_path.exists():
            continue
        gt = np.load(gt_path)
        grid = np.load(grid_path)
        for t in InternalTerrain:
            if t.value > 6:
                continue
            mask = grid == t
            if mask.sum() > 0:
                target.setdefault(int(t), []).append(gt[mask].mean(axis=0))


def _accumulate_distance_priors(
    round_dir: Path,
    target: dict[tuple[int, int], list[np.ndarray]],
) -> None:
    """Collect distance-conditioned GT vectors from one round."""
    for i in range(5):
        gt_path = round_dir / f"seed_{i}" / "ground_truth.npy"
        grid_path = round_dir / f"seed_{i}" / "initial_grid.npy"
        if not gt_path.exists() or not grid_path.exists():
            continue
        gt = np.load(gt_path)
        grid = np.load(grid_path)
        dist = compute_settlement_distance(grid)
        for t in (
            InternalTerrain.PLAINS,
            InternalTerrain.FOREST,
            InternalTerrain.SETTLEMENT,
            InternalTerrain.PORT,
        ):
            for d in range(1, 8):
                mask = (grid == t) & (dist == d)
                if mask.sum() > 3:
                    key = (int(t), min(d, 5))
                    target.setdefault(key, []).append(gt[mask].mean(axis=0))


def _average_priors(
    vecs: dict[int, list[np.ndarray]],
) -> dict[int, np.ndarray]:
    """Average collected probability vectors per terrain type."""
    result = {}
    for key, v_list in vecs.items():
        avg = np.mean(v_list, axis=0)
        avg = np.maximum(avg, PROBABILITY_FLOOR)
        result[key] = avg / avg.sum()
    return result


def _average_dist_priors(
    vecs: dict[tuple[int, int], list[np.ndarray]],
) -> dict[tuple[int, int], np.ndarray]:
    """Average collected distance-conditioned probability vectors."""
    result = {}
    for key, v_list in vecs.items():
        avg = np.mean(v_list, axis=0)
        avg = np.maximum(avg, PROBABILITY_FLOOR)
        result[key] = avg / avg.sum()
    return result


def _apply_terrain_priors(
    grid: np.ndarray,
    priors: dict[int, np.ndarray],
) -> np.ndarray:
    """Apply flat terrain priors to build initial prediction."""
    h, w = grid.shape
    pred = np.full((h, w, NUM_PREDICTION_CLASSES), 1.0 / NUM_PREDICTION_CLASSES)
    for t_val, prior in priors.items():
        mask = grid == t_val
        if mask.any():
            pred[mask] = prior
    return pred


def _apply_distance_refinement(
    pred: np.ndarray,
    grid: np.ndarray,
    dist_priors: dict[tuple[int, int], np.ndarray],
    weight: float,
) -> np.ndarray:
    """Blend distance-conditioned priors into the prediction."""
    result = pred.copy()
    dist = compute_settlement_distance(grid)
    for (t_val, d_bin), prior in dist_priors.items():
        for d in range(1, 8):
            if min(d, 5) != d_bin:
                continue
            mask = (grid == t_val) & (dist == d)
            if mask.any():
                result[mask] = (1 - weight) * result[mask] + weight * prior
    return result


def _apply_static_overrides(
    pred: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Set near-certain probabilities for static terrain."""
    result = pred.copy()
    ocean = grid == InternalTerrain.OCEAN
    if ocean.any():
        result[ocean] = PROBABILITY_FLOOR
        result[ocean, Terrain.EMPTY] = 1.0

    mountain = grid == InternalTerrain.MOUNTAIN
    if mountain.any():
        result[mountain] = PROBABILITY_FLOOR
        result[mountain, Terrain.MOUNTAIN] = 1.0

    return result
