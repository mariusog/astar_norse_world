"""Build per-regime adaptive priors from historical round data."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Regime -> which historical rounds to use
_REGIME_ROUNDS: dict[str, tuple[int, ...]] = {
    "collapse": (3, 4),
    "aggressive": (1, 2, 5, 6),
    "survive": (1, 2, 5),
}


def build_adaptive_priors(regime: str, data_dir: str = "data/rounds") -> np.ndarray:
    """Build (7, 6) terrain priors from rounds matching the detected regime.

    Falls back to all rounds if no matching rounds have data.
    """
    round_priors = _load_all_round_priors(data_dir)
    target_rounds = _REGIME_ROUNDS.get(regime, ())
    selected = [r for r in target_rounds if r in round_priors]
    if not selected:
        selected = list(round_priors.keys())
    result = np.mean([round_priors[r] for r in selected], axis=0)
    logger.info("Adaptive priors for '%s' from rounds %s", regime, selected)
    return result


def _load_all_round_priors(data_dir: str) -> dict[int, np.ndarray]:
    """Load per-round flat terrain priors from disk."""
    rounds_dir = Path(data_dir)
    result: dict[int, np.ndarray] = {}
    if not rounds_dir.exists():
        return result
    for rd in sorted(rounds_dir.iterdir()):
        if not rd.is_dir():
            continue
        rj = rd / "round.json"
        if not rj.exists():
            continue
        rdata = json.loads(rj.read_text())
        rnum = rdata.get("round_number", 0)
        if rnum <= 0:
            continue
        accum = np.zeros((7, 6))
        count = np.zeros(7)
        for i in range(5):
            gt_p = rd / f"seed_{i}" / "ground_truth.npy"
            gr_p = rd / f"seed_{i}" / "initial_grid.npy"
            if not gt_p.exists() or not gr_p.exists():
                continue
            gt, gr = np.load(gt_p), np.load(gr_p)
            for t in range(7):
                mask = gr == t
                if mask.sum() > 0:
                    accum[t] += gt[mask].sum(axis=0)
                    count[t] += mask.sum()
        priors = np.zeros((7, 6))
        for t in range(7):
            priors[t] = accum[t] / count[t] if count[t] > 0 else 1 / 6
        result[rnum] = priors
    return result
